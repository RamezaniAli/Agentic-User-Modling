from __future__ import annotations
import json
import uuid
import os
from IPython.display import Image, display

from typing_extensions import TypedDict, Annotated, Literal, Optional, List, Dict
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

from agentic_rag.config import BASE_URL, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, MODEL_TEMP
from agent.memory_manager import MemoryManager
from agent.utils import safe_json_parse

# --------------------------------------------------------------------------- #
# 1 ---- State definition
# --------------------------------------------------------------------------- #

class ReflectState(TypedDict):

    ### Reflect Agent:
    user_id: str
    int_id: str
    item_information: str
    rating: float
    review: str

    retrieved_chunk_ids: dict  # Chunk IDs from ReAct agent
    ### Retrieved interactions (full content):
    retrieved_interactions: Optional[dict]    

    ### Agent messages and outputs:
    reflect_messages: Annotated[list[AnyMessage], add_messages]
    updated_retrieved_interactions: Optional[list]
    updated_persona: Optional[str]

# --------------------------------------------------------------------------- #
# 2 ---- Shared resources
# --------------------------------------------------------------------------- #

memory = MemoryManager(embedding_model=EMBEDDING_MODEL_NAME)

# Add this import and initialization
# from agent.tools import set_memory
# set_memory(memory)  # Initialize the global memory in tools

llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    temperature=MODEL_TEMP,
    base_url=BASE_URL,
)

# --------------------------------------------------------------------------- #
# 3 ---- tools
# --------------------------------------------------------------------------- #

@tool
def update_persona_tool(user_id: str, persona_content: str) -> bool:
    """
    Update or create a user's persona with new content.

    Args:
        user_id (str): The unique identifier of the user.
        persona_content (str): The new persona content as natural language description.

    Returns:
        bool: True if update/creation successful, False otherwise.
    """
    if memory is None:
        raise RuntimeError("Memory not initialized. Call set_memory() first.")
    
    try:
        metadata = {
            "user_id": user_id,
            "updated": 'True'
        }
        
        # Check if persona already exists for this user
        existing_persona = memory.get_persona(user_id=user_id)
        if existing_persona:
            # Update existing persona
            updates = [{
                "chunk_id": existing_persona['chunk_id'],
                "text": persona_content,
                "metadata": metadata,
                "chunk_type": "persona"
            }]
            return memory.update_chunk(updates)
        else:
            # Create new persona
            persona_chunk_id = f"{user_id}_00" 
            memory.add_chunk(
                text=persona_content,
                metadata=metadata,
                chunk_type="persona",
                chunk_id=persona_chunk_id
            )
            return True
        
    except Exception as e:
        print(f"Error updating persona for user {user_id}: {e}")
        return False

@tool
def update_interaction_tool(interactions: List[Dict]) -> bool:
    """
    Update one or multiple interaction records.
    Agent can provide a list with one element for single update, or multiple elements for batch update.

    Args:
        interactions (List[Dict]): List of interaction updates. Each dict should contain:
            - chunk_id (str, required): Unique identifier of the retrieved chunk.
            - note (str, required): A brief summary describing the chunk's influence in this prediction task.
            - success_score (str, required): A score from "0.0" to "1.0" representing the chunk's contribution.

    Returns:
        bool: True if all interaction updates successful, False otherwise.
    """
    if memory is None:
        raise RuntimeError("Memory not initialized. Call set_memory() first.")
    
    try:
        if not interactions or not isinstance(interactions, list):
            print("interactions must be a non-empty list")
            return False
        
        updates = []
        
        for interaction in interactions:
            chunk_id = interaction.get('chunk_id')
            note = interaction.get('note')
            success_score = interaction.get('success_score')

            if not chunk_id or not note:
                print(f"Skipping interaction: chunk_id and note are required. Got: {interaction}")
                continue
                        
            existing_interaction = memory.get_chunk_by_id(chunk_id=chunk_id)
            if existing_interaction:
                # Create a copy of existing metadata (excluding 'content')
                existing_metadata = {k: v for k, v in existing_interaction.items() if k != 'content'}
                
                # Create new feedback entry
                new_feedback_entry = {'note': note, 'success_score': success_score}
                
                # Handle feed_back - serialize as JSON string for vector DB compatibility
                if 'feed_back' in existing_metadata:
                    try:
                        # Try to deserialize existing feedback (if it's a JSON string)
                        if isinstance(existing_metadata['feed_back'], str):
                            existing_feedback = json.loads(existing_metadata['feed_back'])
                        else:
                            existing_feedback = existing_metadata['feed_back']
                        
                        # Ensure it's a list
                        if not isinstance(existing_feedback, list):
                            existing_feedback = [existing_feedback]
                        
                        # Append new feedback
                        existing_feedback.append(new_feedback_entry)
                        
                    except (json.JSONDecodeError, TypeError):
                        # If deserialization fails, start fresh
                        print(f"Warning: Could not parse existing feedback for {chunk_id}, starting fresh")
                        existing_feedback = [new_feedback_entry]
                else:
                    # No existing feedback
                    existing_feedback = [new_feedback_entry]
                
                # Serialize feedback list as JSON string for vector DB
                existing_metadata['feed_back'] = json.dumps(existing_feedback)
                
                # Mark as updated
                existing_metadata['updated'] = 'True'
                
                # Ensure all metadata values are vector DB compatible
                sanitized_metadata = {}
                for key, value in existing_metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        sanitized_metadata[key] = value
                    else:
                        # Convert complex types to JSON strings
                        sanitized_metadata[key] = json.dumps(value) if value is not None else None
                
                metadata = sanitized_metadata
                
            else:
                print(f"Interaction with chunk_id {chunk_id} not found. Skipping update.")
                continue

            # Prepare update object
            update_obj = {
                "chunk_id": chunk_id,
                "text": existing_interaction['content'],
                "metadata": metadata,
                "chunk_type": "interaction"
            }
            
            updates.append(update_obj)
        
        if updates:
            result = memory.update_chunk(updates)
            if result:
                print(f"Successfully updated {len(updates)} interactions")
                return True
            else:
                print("Memory update_chunk returned False")
                return False
        else:
            print("No valid interaction updates to process")
            return False
        
    except Exception as e:
        print(f"Error updating interactions: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
        return False

reflect_tools = [
    update_persona_tool,
    update_interaction_tool,
]

reflect_llm_with_tools = llm.bind_tools(reflect_tools)
reflect_tools_node = ToolNode(tools=reflect_tools, messages_key="reflect_messages")

# --------------------------------------------------------------------------- #
# 4 ---- Nodes
# --------------------------------------------------------------------------- #

def retrieve_chunks_node(state: ReflectState) -> ReflectState:
    """Retrieve full interaction content using chunk IDs from ReAct agent."""
    
    retrieved_chunk_ids = state.get("retrieved_chunk_ids", {"self": [], "peer": []})
    retrieved_interactions = {"self": [], "peer": []}
    
    # Retrieve self interactions
    for chunk_id in retrieved_chunk_ids.get("self", []):
        try:
            chunk_data = memory.get_chunk_by_id(chunk_id)
            if chunk_data:
                retrieved_interactions["self"].append(chunk_data)
            else:
                print(f"Warning: Could not retrieve chunk {chunk_id}")
        except Exception as e:
            print(f"Error retrieving chunk {chunk_id}: {e}")
    
    # Retrieve peer interactions
    for chunk_id in retrieved_chunk_ids.get("peer", []):
        try:
            chunk_data = memory.get_chunk_by_id(chunk_id)
            if chunk_data:
                retrieved_interactions["peer"].append(chunk_data)
            else:
                print(f"Warning: Could not retrieve chunk {chunk_id}")
        except Exception as e:
            print(f"Error retrieving chunk {chunk_id}: {e}")
    
    print(f"Retrieved {len(retrieved_interactions['self'])} self interactions and {len(retrieved_interactions['peer'])} peer interactions")

    retrieve_msg = SystemMessage(
            content=f"Retrieved {retrieved_interactions['self']} self interactions and {retrieved_interactions['peer']} peer interactions."
        )
    
    return {"retrieved_interactions": retrieved_interactions,
            "reflect_messages":[retrieve_msg]}

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

    metadata = {
    "user_id": state['user_id'],
    "true_rating": state['rating'],
    "true_review": state['review'],
    "updated": 'False'
    }


    memory.add_chunk(
        text=state['item_information'],
        metadata=metadata,
        chunk_type="interaction",
        chunk_id=state['int_id']
    )
    
    reflect_messages = state["reflect_messages"]
    last_msg = reflect_messages[-1]
    
    # Check if the last message contains structured output
    if hasattr(last_msg, 'content'):
        content = last_msg.content
    else:
        print("ERROR: Last reflect message has no content.")
        raise ValueError("Last reflect message has no content.")
    
    # Try to parse the content using our robust parser
    required_keys = ["updated_retrieved_interactions", "updated_persona"]
    result = safe_json_parse(content, required_keys)
    
    if result is None:
        print("ERROR: Could not parse any valid JSON/dict from reflect content.")
        print(f"Content preview: {content[:500]}...")
        
        # Provide default values
        updated_retrieved_interactions = ["No updates needed"]
        updated_persona = "No updates needed"
    else:
        # Successfully parsed result
        updated_retrieved_interactions = result.get("updated_retrieved_interactions", ["No updates needed"])
        updated_persona = result.get("updated_persona", "No updates needed")

        print(f"Successfully extracted reflect results")

    return {
        "updated_retrieved_interactions": updated_retrieved_interactions,
        "updated_persona": updated_persona,
    }


# --------------------------------------------------------------------------- #
# 5 ---- Conditional Edges for Agents to Tools
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
    graph_builder.add_node("retrieve_chunks_node", retrieve_chunks_node)
    graph_builder.add_node("reflect_node", reflect_node)
    graph_builder.add_node("finish_reflect_node", finish_reflect_node)
    graph_builder.add_node("reflect_tools", reflect_tools_node)

    # Add edges
    graph_builder.add_edge(START, "retrieve_chunks_node") 
    graph_builder.add_edge("retrieve_chunks_node", "reflect_node")

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