

from typing import List, Optional, Dict
from langchain_chroma  import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from config import CHROMA_DB_PATH, TOP_K_PERSONA, TOP_K_INTERACTION, BASE_URL, EMBEDDING_MODEL_NAME
from langchain.schema import Document
import os
import chromadb
from chromadb.config import Settings
import uuid

class MemoryManager:
    """
    Wrapper for ChromaDB vector store using LangChain and Google Gemini embeddings.
    Provides methods for persona management and interaction retrieval.
    """

    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        base_url: str = BASE_URL,
        collection_name: str = "memory_collection"
    ):
        # Initialize embeddings client; relies on GOOGLE_API_KEY in environment
        # self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.embeddings = OpenAIEmbeddings(model=embedding_model,
                                           base_url=base_url)
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Ensure directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize LangChain Chroma wrapper
        self.vstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=db_path
        )

    def add_chunk(
        self,
        text: str,
        metadata: Dict,
        chunk_type: str,
        chunk_id: Optional[str] = None
    ) -> None:
        """
        Add a new chunk (persona or interaction) to ChromaDB.

        Args:
            text: chunk content
            metadata: extra metadata including 'user_id'
            chunk_type: 'persona' or 'interaction'
            chunk_id: unique identifier for the chunk
        """
        full_meta = {
            "chunk_type": chunk_type, 
            **metadata
        }
        
        # Generate chunk_id if not provided
        if chunk_id is None:
            chunk_id = f"{chunk_type}_{metadata.get('user_id', 'unknown')}_{uuid.uuid4()}"
        
        full_meta["chunk_id"] = chunk_id
        
        doc = Document(page_content=text, metadata=full_meta)
        self.vstore.add_documents([doc], ids=[chunk_id])

    def update_chunk(
        self,
        updates: List[Dict]
    ) -> bool:
        """
        Update one or multiple existing chunks in ChromaDB.
        Automatically handles single or batch updates based on list length.
        
        Args:
            updates: List of update dictionaries. Each dict should contain:
                - chunk_id (str, required): unique identifier of the chunk to update
                - text (str, required): new content for the chunk
                - metadata (dict, optional): new metadata to merge with existing
                - chunk_type (str, optional): chunk type if different from existing
        
        Examples:
            # Single update
            update_chunk([{"chunk_id": "id1", "text": "new content"}])
            
            # Multiple updates  
            update_chunk([
                {"chunk_id": "id1", "text": "content 1", "metadata": {"rating": 4.0}},
                {"chunk_id": "id2", "text": "content 2"}
            ])

        Returns:
            bool: True if all updates successful, False otherwise
        """
        if not updates or not isinstance(updates, list):
            print("updates must be a non-empty list")
            return False
        
        try:
            # Use batch processing regardless of single or multiple updates
            # This keeps the interface consistent and leverages ChromaDB's efficiency
            documents = []
            ids = []
            
            for update in updates:
                chunk_id = update.get('chunk_id')
                text = update.get('text')
                
                if not chunk_id or not text:
                    print(f"Skipping update: chunk_id and text are required. Got: {update}")
                    continue
                
                # Get existing chunk to preserve metadata
                existing_chunk = self.get_chunk_by_id(chunk_id)
                if not existing_chunk:
                    print(f"Chunk {chunk_id} not found, skipping")
                    continue
                
                # Start with existing metadata (excluding 'content' key)
                existing_metadata = {k: v for k, v in existing_chunk.items() if k != 'content'}
                
                # Use existing or provided chunk_type
                chunk_type = update.get('chunk_type', existing_metadata.get('chunk_type'))
                
                # Merge with new metadata if provided
                new_metadata = update.get('metadata', {})
                if new_metadata:
                    existing_metadata.update(new_metadata)
                
                # Build final metadata
                full_meta = {
                    **existing_metadata,
                    "chunk_type": chunk_type,
                    "chunk_id": chunk_id,
                }
                
                doc = Document(
                    page_content=text,
                    metadata=full_meta,
                    id=chunk_id
                )
                
                documents.append(doc)
                ids.append(chunk_id)
            
            if documents:
                # Use batch update for efficiency (works for both single and multiple)
                self.vstore.update_documents(ids=ids, documents=documents)
                return True
            else:
                print("No valid updates to process")
                return False
                
        except Exception as e:
            print(f"Error updating chunks: {e}")
            return False

    def get_persona(
        self,
        user_id: Optional[str] = None,
        fallback_persona: Optional[str] = None,
        k: int = TOP_K_PERSONA
    ) -> Optional[Dict]:
        """
        Retrieve a persona chunk.
        1. If user_id is provided, return that exact user's persona or None if not found.
        2. If not found and fallback_persona provided, search other users' personas semantically.
        """

        if user_id:
            # Exact search for specific user's persona
            try:
                results = self.vstore.get(
                    where={"$and": [
                        {"chunk_type": {"$eq": "persona"}},
                        {"user_id": {"$eq": user_id}}
                    ]}
                )
                
                if results['documents']:
                    # Return the first matching persona
                    doc_content = results['documents'][0]
                    doc_metadata = results['metadatas'][0]
                    return {"content": doc_content, **doc_metadata}
                
                # If exact user persona not found and fallback provided, search other users
                if fallback_persona:
                    results = self.vstore.similarity_search(
                        query=fallback_persona,
                        k=k,
                        filter={"$and": [
                            {"chunk_type": {"$eq": "persona"}},
                            {"user_id": {"$ne": user_id}}  # Exclude the original user
                        ]}
                    )
                    
                    if results:
                        result = results[0]
                        return {"content": result.page_content, **result.metadata}
                
                return None
                
            except Exception as e:
                print(f"Error retrieving persona for user {user_id}: {e}")
                return None

        elif fallback_persona:
            # Semantic search across all personas when no specific user_id
            try:
                results = self.vstore.similarity_search(
                    query=fallback_persona,
                    k=k,
                    filter={"chunk_type": {"$eq": "persona"}}
                )
                
                if results:
                    return  [{"content": r.page_content, **r.metadata} for r in results]
                
                return None
                
            except Exception as e:
                print(f"Error retrieving persona with fallback: {e}")
                return None

        return None

    def get_interactions(
        self,
        user_id: str,
        query: str,
        k: int = TOP_K_INTERACTION
    ) -> List[Dict]:
        """
        Retrieve top-k interaction chunks for a user matching the query.

        Returns:
            List of dicts with 'content' and metadata.
        """
        try:
            docs = self.vstore.similarity_search(
                query=query,
                k=k,
                filter={"$and": [
                    {"user_id": {"$eq": user_id}},
                    {"chunk_type": {"$eq": "interaction"}}
                ]}
            )
            return [{"content": d.page_content, **d.metadata} for d in docs]
        
        except Exception as e:
            print(f"Error retrieving interactions for user {user_id}: {e}")
            return []
        

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a specific chunk by its ID.

        Args:
            chunk_id: unique identifier of the chunk

        Returns:
            Dict with content and metadata, or None if not found
        """
        try:
            results = self.vstore.get(ids=[chunk_id])
            
            if results['documents']:
                doc_content = results['documents'][0]
                doc_metadata = results['metadatas'][0]
                return {"content": doc_content, **doc_metadata}
            
            return None
            
        except Exception as e:
            print(f"Error retrieving chunk {chunk_id}: {e}")
            return None

def set_memory(memory_instance: MemoryManager):
    """Set the global memory instance."""
    global memory
    memory = memory_instance

def memory_initalize(input_json, chunk_type):

    if chunk_type == 'persona':
        memory.add_chunk(text= input_json['user_information'],
                         chunk_type = chunk_type,
                         chunk_id = f"{input_json['user_id']}_00",
                         metadata = {'user_id':input_json['user_id'],
                                     'updated':"False"})
        
    elif chunk_type == 'interaction':
        memory.add_chunk(text= input_json['item_information'],
                         chunk_type = chunk_type,
                         chunk_id = input_json['int_id'],
                         metadata = {'user_id':input_json['user_id'],
                                     'true_rating':input_json['true_rating'],
                                     'true_review':input_json['true_review'],
                                     'updated':"False"})