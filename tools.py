from typing import List, Dict, Optional
from langchain.tools import tool
from agentic_rag.config import TOP_K_PERSONA, TOP_K_INTERACTION
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
def add_new_interaction_tool(
    user_id: str,
    int_id: str,
    rating: float,
    review: str,
    item_information: str
) -> bool:
    """
    Add a new interaction record for a user.

    Args:
        user_id (str): The unique identifier of the user.
        rating (float): Rating given by the user for the interaction (0.0 to 5.0).
        review (str): 
        interaction_content (str): Text content of the interaction (title, review, snippet, etc.).

    Returns:
        bool: True if interaction added successfully, False otherwise.
    """
    if memory is None:
        raise RuntimeError("Memory not initialized. Call set_memory() first.")
    
    try:
        metadata = {
            "user_id": user_id,
            "true_rating": rating,
            "true_review": review,
            "updated": 'False'
        }
        
        
        memory.add_chunk(
            text=item_information,
            metadata=metadata,
            chunk_type="interaction",
            chunk_id=int_id
        )
        
        return True
        
    except Exception as e:
        print(f"Error adding interaction for user {user_id} and item {int_id}: {e}")
        return False


@tool
def update_interaction_tool(interactions: List[Dict]) -> bool:
    """
    Update one or multiple interaction records.
    Agent can provide a list with one element for single update, or multiple elements for batch update.

Args:
    interactions (List[Dict]): List of interaction updates. Each dict should contain:
        - chunk_id (str, required): Unique identifier of the retrieved chunk.
        - note (str, required): A brief summary describing the chunk’s influence in this prediction task. It should reflect:
            - the type of item involved (e.g., "crime novel", "fitness tracker")
            - and how well this chunk contributed to predicting the correct user feedback (review or rating).
        - success_score (str, required): A score from "0.0" to "1.0" (in 0.1 increments) representing the chunk’s contribution to prediction success.
            Higher scores indicate greater usefulness; lower scores indicate misleading or irrelevant influence.

Examples:

    # Feedback for a single retrieved chunk (from a prediction on a crime novel)
    [
        {
            "chunk_id": "retrieved_chunk_001",
            "note": "This chunk reflected typical user reactions to crime novels and matched the tone of the target review.",
            "success_score": "0.9"
        }
    ]

    # Multiple retrieved chunks evaluated after a prediction task on a romantic novel
    [
        {
            "chunk_id": "retrieved_chunk_002",
            "note": "This chunk emphasized emotional themes common in romantic fiction, aligning well with the predicted review.",
            "success_score": "0.8"
        },
        {
            "chunk_id": "retrieved_chunk_003",
            "note": "This chunk was about detective stories and confused the sentiment prediction for the romance novel.",
            "success_score": "0.2"
        }
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
            note = interaction.get('note')
            success_score = interaction.get('success_score')

            
            if not chunk_id or not note:
                print(f"Skipping interaction: chunk_id and content are required. Got: {interaction}")
                continue
                        
            existing_interaction = memory.get_chunk_by_id(chunk_id=chunk_id)
            if existing_interaction:
                # If the interaction exists, we can use its metadata
                existing_metadata = {k: v for k, v in existing_interaction.items() if k != 'content'}
                if existing_metadata['feed_back']:
                    existing_metadata['feed_back'].append({'note': note, 'success_score': success_score})
                    metadata = {
                        "updated": 'True'
                    }
                else:
                    metadata = {
                        "updated": 'True',
                        "feed_back": [{'note': note, 'success_score': success_score}]
                    }

                metadata.update(existing_metadata)
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
            return memory.update_chunk(updates)
        else:
            print("No valid interaction updates to process")
            return False
        
    except Exception as e:
        print(f"Error updating interactions: {e}")
        return False