


from __future__ import annotations
import json
import uuid
import os
from IPython.display import Image, display

from typing_extensions import TypedDict, Annotated, Literal, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

from config import BASE_URL, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, MODEL_TEMP
from agent.memory_manager import MemoryManager
import re

# --------------------------------------------------------------------------- #
# 1 ---- State definition
# --------------------------------------------------------------------------- #

class AgentState(TypedDict):
    ### always present on entry:
    user_id: str
    user_information: dict 
    item_id: str
    item_information: str
    true_rating: float
    true_review: str
    ### Persona node:
    persona: Optional[str]
    ### ReAct Agent:
    react_messages: Annotated[list[AnyMessage], add_messages]
    ### finish react node:
    predicted_rating: Optional[float]
    predicted_review: Optional[str]
    retrieved_interactions: Optional[dict]  # interactions retrieved by ReAct agent
    ### Reflect Agent:
    reflect_messages: Annotated[list[AnyMessage], add_messages]
    updated_interaction: Optional[str]
    updated_retrieved_interactions: Optional[list]
    updated_persona: Optional[str]



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

def persona_node(state: AgentState) -> AgentState:
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
        info = state.get("user_information", "No user_information provided")
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
            chunk_id=f"persona_{user_id}_{uuid.uuid4()}"
        )
        
        if success:
            lookup_msg = SystemMessage(
                content=f"User was not in database, persona created from user information.\nPersona: {persona_text}"
            )
        else:
            lookup_msg = SystemMessage(
                content=f"Error saving persona, but created from user information.\nPersona: {persona_text}"
            )

    return {
        "react_messages": [lookup_msg],
        "persona": persona_text,
    }


def react_node(state: AgentState) -> AgentState:
    """ReAct agent that produces rating, review and retrieved interactions."""

    current_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(current_dir, "prompts", "react_prompt.txt")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        sys_prompt = f.read()
    sys_msg = SystemMessage(content=sys_prompt)

    react_messages = [sys_msg] + state["react_messages"]
    react_result = react_llm_with_tools.invoke(react_messages)

    return {"react_messages": [react_result]}


def finish_react_node(state: AgentState) -> AgentState:
    """Process the ReAct agent's output and extract rating, review, and interactions."""
    
    react_messages = state["react_messages"]
    last_msg = react_messages[-1]
    
    # Check if the last message contains structured output
    if hasattr(last_msg, 'content'):
        content = last_msg.content
    else:
        raise ValueError("Last react message has no content.")

    try:
        # Extract JSON block using regex
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if not match:
            raise ValueError("No JSON block found in content.")

        json_block = match.group(1)

        # Parse extracted JSON block
        result = json.loads(json_block)

        # Extract fields
        predicted_rating = result.get("rating")
        predicted_review = result.get("review")
        retrieved_interactions = result.get("retrieved_interactions", [])

    except json.JSONDecodeError:
        raise ValueError("Error in JSON Decoding.")

    current_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(current_dir, "prompts", "reflect_human_prompt.txt")

    with open(prompt_path, "r", encoding="utf-8") as file:
        template = file.read()
    filled_prompt = template.format(
        user_id=state["user_id"],
        user_information=state["user_information"],
        item_id=state["item_id"],
        item_information=state["item_information"],
        true_rating=state["true_rating"],
        true_review=state["true_review"],
        predicted_rating=predicted_rating,
        predicted_review=predicted_review,
        persona=state.get("persona", "null"),
        retrieved_interactions=retrieved_interactions)
    first_reflect_message = HumanMessage(content=filled_prompt)

    return {
        "reflect_messages": [first_reflect_message],
        "predicted_rating": predicted_rating,
        "predicted_review": predicted_review,
        "retrieved_interactions": retrieved_interactions,
    }


def reflect_node(state: AgentState) -> AgentState:
    """Reflect on the ReAct agent's output and update persona and interactions."""

    current_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(current_dir, "prompts", "reflect_prompt.txt")

    with open(prompt_path, "r", encoding="utf-8") as f:
        sys_prompt = f.read()
    sys_msg = SystemMessage(content=sys_prompt)
    
    reflect_messages = [sys_msg] + state["reflect_messages"]
    reflect_result = reflect_llm_with_tools.invoke(reflect_messages)

    return {"reflect_messages": [reflect_result]}


# --------------------------------------------------------------------------- #
# 3 ---- tools
# --------------------------------------------------------------------------- #

from agent.tools import *

react_tools = [
    get_persona_tool,
    get_interactions_tool,
    get_similar_users_persona_tool,
]

reflect_tools = [
    update_persona_tool,
    update_interaction_tool,
    add_new_interaction_tool,
]

react_llm_with_tools = llm.bind_tools(react_tools)
reflect_llm_with_tools = llm.bind_tools(reflect_tools)

react_tools_node = ToolNode(tools=react_tools, messages_key="react_messages")
reflect_tools_node = ToolNode(tools=reflect_tools, messages_key="reflect_messages")

# --------------------------------------------------------------------------- #
# 3 ---- Conditional Edges for Agents to Tools
# --------------------------------------------------------------------------- #

def react_tools_condition(state: AgentState) -> Literal["react_tools", "finish_react_node"]:
    """Route from react_node to either tools or finish based on tool calls."""
    react_messages = state.get("react_messages", [])
    if not react_messages:
        return "finish_react_node"
    
    last_message = react_messages[-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "react_tools"
    return "finish_react_node"



def reflect_tools_condition(state: AgentState) -> Literal["reflect_tools", "__end__"]:
    """Route from reflect_node to either tools or end based on tool calls."""
    reflect_messages = state.get("reflect_messages", [])
    if not reflect_messages:
        return "__end__"
    
    last_message = reflect_messages[-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "reflect_tools"
    return "__end__"

# --------------------------------------------------------------------------- #
# Graph assembly with proper conditional routing
# --------------------------------------------------------------------------- #

def build_graph():

    # Create the graph
    graph_builder = StateGraph(AgentState)

    # Add all nodes
    graph_builder.add_node("persona_node", persona_node)
    graph_builder.add_node("react_node", react_node)
    graph_builder.add_node("react_tools", react_tools_node)
    graph_builder.add_node("finish_react_node", finish_react_node)
    graph_builder.add_node("reflect_node", reflect_node)
    graph_builder.add_node("reflect_tools", reflect_tools_node)

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
    graph_builder.add_edge("finish_react_node", "reflect_node")

    # Conditional edge for reflect_node
    graph_builder.add_conditional_edges(
        "reflect_node", 
        reflect_tools_condition  # This routes to either "reflect_tools" or END
    )

    # After reflect tools, go back to reflect_node
    graph_builder.add_edge("reflect_tools", "reflect_node")

    # Compile the graph
    graph = graph_builder.compile()

    return graph

def visualize_graph(graph):
    """Visualize the graph using Mermaid."""
    display(Image(graph.get_graph().draw_mermaid_png()))