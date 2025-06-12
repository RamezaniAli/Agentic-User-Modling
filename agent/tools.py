from typing import List, Dict, Optional
from langchain.tools import tool
from config import TOP_K_PERSONA, TOP_K_INTERACTION
from agent.memory_manager import MemoryManager
import uuid

# Global memory instance - will be set by graph_agent.py
memory: Optional[MemoryManager] = None

def set_memory(memory_instance: MemoryManager):
    """Set the global memory instance."""
    global memory
    memory = memory_instance

@tool
def get_persona_tool(user_id: str) -> Optional[str]:
    """
    Retrieve the stored persona of a user as a short natural language description.

    Args:
        user_id (str): The unique identifier of the user.

    Returns:
        str: A natural language summary of the user's persona, including demographic and behavioral characteristics.
             Returns None if no persona found for the user.
    """
    if memory is None:
        raise RuntimeError("Memory not initialized. Call set_memory() first.")
    
    persona = memory.get_persona(user_id=user_id)
    if persona:
        return persona['content']
    return None


@tool
def get_interactions_tool(user_id: str, query: str) -> List[Dict]:
    """
    Retrieve the user's past interactions that are semantically relevant to the given query (typically the item description).

    Args:
        user_id (str): The unique identifier of the user.
        query (str): A description of the target item (used for similarity search).

    Returns:
        List[dict]: A list of dictionaries, each representing an interaction with the following keys:
            - content (str): Text content of the interaction, including title, rating, and snippet
            - chunk_type (str): Type of the chunk, e.g., "interaction"
            - user_id (str): ID of the user who interacted
            - item_id (str): ID of the item involved in the interaction
            - chunk_id (str): Unique identifier for the chunk
    """
    if memory is None:
        raise RuntimeError("Memory not initialized. Call set_memory() first.")
    
    return memory.get_interactions(user_id=user_id, query=query, k=TOP_K_INTERACTION)


@tool
def get_similar_users_persona_tool(persona_content: str) -> List[Dict]:
    """
    Retrieve personas of users who are semantically similar to the given persona content.

    Args:
        persona_content (str): A natural language description of the current user's persona.

    Returns:
        List[dict]: A list of similar user personas with the following keys:
            - user_id (str): ID of the similar user
            - persona (str): Natural language description of that user's persona
    """
    if memory is None:
        raise RuntimeError("Memory not initialized. Call set_memory() first.")
    
    try:
        similar_users = memory.vstore.similarity_search(
            query=persona_content,
            k=TOP_K_PERSONA,
            filter={"chunk_type": {"$eq": "persona"}},
        )
        return [
            {
                "user_id": p.metadata.get("user_id", ""),
                "persona": p.page_content,
            }
            for p in similar_users
        ]
    except Exception as e:
        print(f"Error retrieving similar personas: {e}")
        return []


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
            persona_chunk_id = f"persona_{user_id}_{uuid.uuid4()}" 
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
def add_new_interaction_tool(
    user_id: str,
    item_id: str,
    rating: float,
    interaction_content: str
) -> bool:
    """
    Add a new interaction record for a user.

    Args:
        user_id (str): The unique identifier of the user.
        item_id (str): The unique identifier of the item.
        rating (float): Rating given by the user for the interaction (0.0 to 5.0).
        interaction_content (str): Text content of the interaction (title, review, snippet, etc.).

    Returns:
        bool: True if interaction added successfully, False otherwise.
    """
    if memory is None:
        raise RuntimeError("Memory not initialized. Call set_memory() first.")
    
    try:
        metadata = {
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating,
            "updated": 'False'
        }
        
        # Generate unique chunk_id for the interaction
        interaction_chunk_id = f"interaction_{user_id}_{item_id}_{uuid.uuid4()}"
        
        memory.add_chunk(
            text=interaction_content,
            metadata=metadata,
            chunk_type="interaction",
            chunk_id=interaction_chunk_id
        )
        
        return True
        
    except Exception as e:
        print(f"Error adding interaction for user {user_id} and item {item_id}: {e}")
        return False


@tool
def update_interaction_tool(interactions: List[Dict]) -> bool:
    """
    Update one or multiple interaction records.
    Agent can provide a list with one element for single update, or multiple elements for batch update.

    Args:
        interactions (List[Dict]): List of interaction updates. Each dict should contain:
            - chunk_id (str, required): Unique identifier of the interaction chunk to update
            - content (str, required): New text content of the interaction

    Examples:
        Single interaction update:
        [{"chunk_id": "interaction_123", "content": "Updated review text"}]
        
        Multiple interaction updates:
        [
            {"chunk_id": "interaction_123", "content": "Great movie!"},
            {"chunk_id": "interaction_456", "content": "Not bad"}
        ]

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
            content = interaction.get('content')
            
            if not chunk_id or not content:
                print(f"Skipping interaction: chunk_id and content are required. Got: {interaction}")
                continue
            
            # Prepare metadata for this interaction
            metadata = {
                "updated": 'True'
            }
            
            existing_interaction = memory.get_chunk_by_id(chunk_id=chunk_id)
            if existing_interaction:
                # If the interaction exists, we can use its metadata
                existing_metadata = {k: v for k, v in existing_interaction.items() if k != 'content'}
                metadata.update(existing_metadata)
            else:
                print(f"Interaction with chunk_id {chunk_id} not found. Skipping update.")
                continue

            # Prepare update object
            update_obj = {
                "chunk_id": chunk_id,
                "text": content,
                "metadata": metadata,
                "chunk_type": "interaction"
            }
            
            updates.append(update_obj)
        
        if updates:
            return memory.update_chunk(updates)
        else:
            print("No valid interaction updates to process")
            return False
        
    except Exception as e:
        print(f"Error updating interactions: {e}")
        return False