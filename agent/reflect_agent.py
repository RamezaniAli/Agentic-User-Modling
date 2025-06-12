


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

class ReflectState(TypedDict):

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

def reflect_node(state: ReflectState) -> ReflectState:
    """Reflect on the ReAct agent's output and update persona and interactions."""

    current_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(current_dir, "prompts", "reflect_prompt.txt")

    with open(prompt_path, "r", encoding="utf-8") as f:
        sys_prompt = f.read()
    sys_msg = SystemMessage(content=sys_prompt)
    
    reflect_messages = [sys_msg] + state["reflect_messages"]
    reflect_result = reflect_llm_with_tools.invoke(reflect_messages)

    return {"reflect_messages": [reflect_result]}

def finish_reflect_node(state: ReflectState) -> ReflectState:
    """Process the Reflect agent's output and extract updates."""
    
    reflect_messages = state["reflect_messages"]
    last_msg = reflect_messages[-1]
    
    # Check if the last message contains structured output
    if hasattr(last_msg, 'content'):
        content = last_msg.content
    else:
        print("ERROR: Last reflect message has no content.")
        raise ValueError("Last reflect message has no content.")
    
    # Try to parse the content using our robust parser
    required_keys = ["updated_interaction", "updated_retrieved_interactions", "updated_persona"]
    result = safe_json_parse(content, required_keys)
    
    if result is None:
        print("ERROR: Could not parse any valid JSON/dict from reflect content.")
        print(f"Content preview: {content[:500]}...")
        
        # Provide default values
        updated_interaction = "Error in reflection - could not parse response"
        updated_retrieved_interactions = ["No updates needed"]
        updated_persona = "No updates needed"
    else:
        # Successfully parsed result
        updated_interaction = result.get("updated_interaction", "Default interaction summary")
        updated_retrieved_interactions = result.get("updated_retrieved_interactions", ["No updates needed"])
        updated_persona = result.get("updated_persona", "No updates needed")

        print(f"Successfully extracted reflect results")

    return {
        "updated_interaction": updated_interaction,
        "updated_retrieved_interactions": updated_retrieved_interactions,
        "updated_persona": updated_persona,
    }



# --------------------------------------------------------------------------- #
# 3 ---- tools
# --------------------------------------------------------------------------- #

from agent.tools import *

reflect_tools = [
    update_persona_tool,
    update_interaction_tool,
    add_new_interaction_tool,
]

reflect_llm_with_tools = llm.bind_tools(reflect_tools)
reflect_tools_node = ToolNode(tools=reflect_tools, messages_key="reflect_messages")

# --------------------------------------------------------------------------- #
# 3 ---- Conditional Edges for Agents to Tools
# --------------------------------------------------------------------------- #


def reflect_tools_condition(state: ReflectState) -> Literal["reflect_tools", "finish_reflect_node"]:
    """Route from reflect_node to either tools or end based on tool calls."""
    reflect_messages = state.get("reflect_messages", [])
    if not reflect_messages:
        return "finish_reflect_node"
    
    last_message = reflect_messages[-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "reflect_tools"
    return "finish_reflect_node"

# --------------------------------------------------------------------------- #
# Graph assembly with proper conditional routing
# --------------------------------------------------------------------------- #

def build_reflect_graph():

    # Create the graph
    graph_builder = StateGraph(ReflectState)

    # Add all nodes
    graph_builder.add_node("reflect_node", reflect_node)
    graph_builder.add_node("finish_reflect_node", finish_reflect_node)
    graph_builder.add_node("reflect_tools", reflect_tools_node)

    # Add edges
    graph_builder.add_edge(START, "reflect_node") 

    # Conditional edge for reflect_node
    graph_builder.add_conditional_edges(
        "reflect_node", 
        reflect_tools_condition  # This routes to either "reflect_tools" or END
    )

    # After reflect tools, go back to reflect_node
    graph_builder.add_edge("reflect_tools", "reflect_node")

    # After finishing react, go to reflect
    graph_builder.add_edge("finish_reflect_node", END)

    # Compile the graph
    graph = graph_builder.compile()

    return graph

def visualize_graph(graph):
    """Visualize the graph using Mermaid."""
    display(Image(graph.get_graph().draw_mermaid_png()))