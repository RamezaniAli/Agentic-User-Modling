


from __future__ import annotations
import json
import uuid
import os
from IPython.display import Image, display

from typing_extensions import TypedDict, Annotated, Literal, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

from config import BASE_URL, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, MODEL_TEMP
from agent.memory_manager import MemoryManager
from agent.utils import safe_json_parse
import re

# --------------------------------------------------------------------------- #
# 1 ---- State definition
# --------------------------------------------------------------------------- #

class ReactState(TypedDict):
    ### always present on entry:
    user_id: str
    user_information: dict 
    ### Persona node:
    persona: Optional[str]
    ### ReAct Agent:
    react_messages: Annotated[list[AnyMessage], add_messages]
    ### finish react node:
    predicted_rating: Optional[float]
    predicted_review: Optional[str]
    retrieved_interactions: Optional[dict]  # interactions retrieved by ReAct agent


# --------------------------------------------------------------------------- #
# 2 ---- Shared resources
# --------------------------------------------------------------------------- #

memory = MemoryManager(embedding_model=EMBEDDING_MODEL_NAME)

# Add this import and initialization
from agent.tools import set_memory
set_memory(memory)  # Initialize the global memory in tools

llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    temperature=MODEL_TEMP,
    base_url=BASE_URL,
)

# --------------------------------------------------------------------------- #
# 3 ---- Nodes
# --------------------------------------------------------------------------- #

def persona_node(state: ReactState) -> ReactState:
    """Retrieve or create user persona."""
    user_id = state["user_id"]
    
    # Try to get existing persona
    persona_result = memory.get_persona(user_id=user_id)
    
    if persona_result:
        persona_text = persona_result['content']
        lookup_msg = SystemMessage(
            content=f"User exists in database, persona retrieved.\nPersona: {persona_text}"
        )
    else:
        # Create new persona from demographics
        info = state.get("user_information")

        if isinstance(info, str):
            persona_text = info 
        else:
            prompt = [
                SystemMessage(content="You are a helpful assistant that generates user personas based on user information. Create concise, realistic personas."),
                HumanMessage(content=f"Create a 2-3 sentence persona from these user information: {info}"),
            ]
            resp = llm.invoke(prompt)
            persona_text = resp.content
            
        # Save new persona using the tool
        success = memory.add_chunk(
            text=persona_text,
            metadata={"user_id": user_id},
            chunk_type="persona",
        )
        lookup_msg = SystemMessage(
            content=f"User was not in database, persona created from user information.\nPersona: {persona_text}"
        )


    return {
        "react_messages": [lookup_msg],
        "persona": persona_text,
    }


def react_node(state: ReactState) -> ReactState:
    """ReAct agent that produces rating, review and retrieved interactions."""

    current_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(current_dir, "prompts", "react_prompt.txt")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        sys_prompt = f.read()
    sys_msg = SystemMessage(content=sys_prompt)

    react_messages = [sys_msg] + state["react_messages"]
    react_result = react_llm_with_tools.invoke(react_messages)

    return {"react_messages": [react_result]}


def finish_react_node(state: ReactState) -> ReactState:
    """Process the ReAct agent's output and extract rating, review, and interactions."""
    
    react_messages = state["react_messages"]
    last_msg = react_messages[-1]
    
    # Check if the last message contains structured output
    if hasattr(last_msg, 'content'):
        content = last_msg.content
    else:
        print("ERROR: Last react message has no content.")
        print(f"Message type: {type(last_msg)}")
        print(f"Message attributes: {dir(last_msg)}")
        raise ValueError("Last react message has no content.")

    # Try to parse the content using our robust parser
    required_keys = ["rating", "review", "retrieved_interactions"]
    result = safe_json_parse(content, required_keys)
    
    if result is None:
        print("ERROR: Could not parse any valid JSON/dict from content.")
        print(f"Content preview: {content[:500]}...")
        
        # Last resort: try to extract rating and review from text
        rating_match = re.search(r"rating[\"']?\s*[:=]\s*([0-9.]+)", content, re.IGNORECASE)
        review_match = re.search(r"review[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']", content, re.IGNORECASE)
        
        predicted_rating = float(rating_match.group(1)) if rating_match else 2.5
        predicted_review = review_match.group(1) if review_match else "Could not extract review from response."
        retrieved_interactions = {"self": [], "peer": []}
        
        print(f"Fallback extraction: rating={predicted_rating}, review='{predicted_review}'")
    else:
        # Successfully parsed result
        try:
            # Extract and validate fields
            predicted_rating = float(result.get("rating", 2.5))
            predicted_review = str(result.get("review", "Unable to generate review."))
            retrieved_interactions = result.get("retrieved_interactions", {"self": [], "peer": []})

            # Validate rating range
            if not (0.0 <= predicted_rating <= 5.0):
                print(f"WARNING: Rating {predicted_rating} out of range, clamping to 0-5")
                predicted_rating = max(0.0, min(5.0, predicted_rating))

            # Ensure retrieved_interactions has the right structure
            if not isinstance(retrieved_interactions, dict):
                retrieved_interactions = {"self": [], "peer": []}
            if "self" not in retrieved_interactions:
                retrieved_interactions["self"] = []
            if "peer" not in retrieved_interactions:
                retrieved_interactions["peer"] = []

            print(f"Successfully extracted: rating={predicted_rating}, review_length={len(predicted_review)}")

        except Exception as e:
            print(f"Error processing parsed result: {e}")
            predicted_rating = 2.5
            predicted_review = "Error in processing parsed result."
            retrieved_interactions = {"self": [], "peer": []}

    return {
        "persona": state.get('persona', 'Unknown user'),
        "predicted_rating": predicted_rating,
        "predicted_review": predicted_review,
        "retrieved_interactions": retrieved_interactions,
    }

# --------------------------------------------------------------------------- #
# 3 ---- tools
# --------------------------------------------------------------------------- #

from agent.tools import *

react_tools = [
    get_interactions_tool,
    get_similar_users_persona_tool,
]

react_llm_with_tools = llm.bind_tools(react_tools)
react_tools_node = ToolNode(tools=react_tools, messages_key="react_messages")

# --------------------------------------------------------------------------- #
# 3 ---- Conditional Edges for Agents to Tools
# --------------------------------------------------------------------------- #

def react_tools_condition(state: ReactState) -> Literal["react_tools", "finish_react_node"]:
    """Route from react_node to either tools or finish based on tool calls."""
    react_messages = state.get("react_messages", [])
    if not react_messages:
        return "finish_react_node"
    
    last_message = react_messages[-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "react_tools"
    return "finish_react_node"

# --------------------------------------------------------------------------- #
# Graph assembly with proper conditional routing
# --------------------------------------------------------------------------- #

def build_react_graph():

    # Create the graph
    graph_builder = StateGraph(ReactState)

    # Add all nodes
    graph_builder.add_node("persona_node", persona_node)
    graph_builder.add_node("react_node", react_node)
    graph_builder.add_node("react_tools", react_tools_node)
    graph_builder.add_node("finish_react_node", finish_react_node)

    # Add edges
    graph_builder.add_edge(START, "persona_node") 
    graph_builder.add_edge("persona_node", "react_node")

    # Conditional edge for react_node
    graph_builder.add_conditional_edges(
        "react_node", 
        react_tools_condition  # This routes to either "react_tools" or "finish_react_node"
    )

    # After tools, go back to react_node
    graph_builder.add_edge("react_tools", "react_node")

    # After finishing react, go to reflect
    graph_builder.add_edge("finish_react_node", END)

    # Compile the graph
    graph = graph_builder.compile()

    return graph

def visualize_graph(graph):
    """Visualize the graph using Mermaid."""
    display(Image(graph.get_graph().draw_mermaid_png()))