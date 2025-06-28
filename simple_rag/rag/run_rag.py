import json
import os
import re

from typing import List, Dict, Any
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from simple_rag.rag_config import BASE_URL, LLM_MODEL_NAME, MODEL_TEMP, TOP_K_INTERACTION
from memory.memory_manager import MemoryManager, set_memory
from utils.utils import safe_json_parse


def load_jsonl(file_path: str) -> List[Dict[Any, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[Any, Any]], file_path: str) -> None:
    """Save data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def create_batches(data: List[Dict[Any, Any]], batch_size: int) -> List[List[Dict[Any, Any]]]:
    """Split data into batches of specified size."""
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches

def initialize_llm_client() -> ChatOpenAI:
    """Initialize the LLM client."""
    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=MODEL_TEMP,
        base_url=BASE_URL,
    )


def load_prompts() -> tuple[str, str]:
    """Load system and human prompt templates from files."""
    current_dir = os.path.dirname(__file__)
    
    # Load system prompt
    sys_prompt_path = os.path.join(current_dir, "prompts", "sys_prompt.txt")
    with open(sys_prompt_path, "r", encoding="utf-8") as f:
        sys_prompt = f.read()
    
    # Load human prompt
    human_prompt_path = os.path.join(current_dir, "prompts", "human_prompt.txt")
    with open(human_prompt_path, "r", encoding="utf-8") as f:
        human_prompt = f.read()
    
    return sys_prompt, human_prompt


def simple_predict(interaction: Dict[Any, Any], memory: MemoryManager, llm_client: ChatOpenAI, debug: bool = False) -> Dict[Any, Any]:
    """
    Simple prediction function that retrieves relevant interactions and calls LLM for prediction.
    
    Args:
        interaction: Input interaction data
        memory: MemoryManager instance
        llm_client: LLM client for making predictions
        debug: Whether to print debug information
    
    Returns:
        Dictionary with prediction results
    """
    
    # Retrieve relevant interactions from memory
    query = interaction["item_information"]
    user_id = interaction["user_id"]
    
    retrieved_interactions = memory.get_interactions(
        user_id=user_id,
        query=query,
        k=TOP_K_INTERACTION
    )
    
    if debug:
        print(f"  Retrieved {len(retrieved_interactions)} interactions for user {user_id}")
    
    
    # Load prompts
    sys_prompt, human_prompt_template = load_prompts()
    
    # Prepare context for prompts

    filled_human_prompt = human_prompt_template.format(
        user_information=interaction["user_information"],
        item_information=interaction["item_information"]
    )
    
    # Create messages
    sys_msg = SystemMessage(content=sys_prompt)
    human_msg = HumanMessage(content=filled_human_prompt)
    
    try:
        # Call LLM
        response = llm_client.invoke([sys_msg, human_msg])
        
        if debug:
            print(f"  LLM Response: {response.content[:200]}...")
        
        # Parse LLM response
        required_keys = ["predicted_review", "predicted_rating"]
        result = safe_json_parse(response.content, required_keys)
        
        if result is None:
            print("ERROR: Could not parse valid JSON from LLM response.")
            print(f"Response preview: {response.content[:500]}...")
            
            # Provide default values
            predicted_rating = 2.5
            predicted_review = "Could not generate prediction due to parsing error."
        else:
            # Successfully parsed result
            predicted_rating = result.get("predicted_rating", 2.5)
            predicted_review = result.get("predicted_review", "No review generated.")
            
            if debug:
                print(f"  ✓ Successfully parsed LLM response - Rating: {predicted_rating}")
    
    except Exception as e:
        print(f"ERROR: LLM call failed: {e}")
        predicted_rating = 2.5
        predicted_review = f"Prediction failed due to error: {str(e)}"
    
    # Combine original interaction data with prediction results
    result = {
        'user_id': interaction["user_id"],
        'user_information': interaction["user_information"],
        'int_id': interaction["int_id"],
        'item_information': interaction["item_information"],
        'true_rating': interaction["true_rating"],
        'true_review': interaction["true_review"],
        'predicted_rating': predicted_rating,
        'predicted_review': predicted_review,
        'retrieved_interactions': retrieved_interactions
    }
    
    return result


def add_interactions_to_memory(batch_interactions: List[Dict[Any, Any]], memory: MemoryManager, debug: bool = False) -> List[Dict[Any, Any]]:
    """
    Add batch of interactions to memory using add_chunk method.
    
    Args:
        batch_interactions: List of processed interactions
        memory: MemoryManager instance
        debug: Whether to print debug information
    
    Returns:
        List of interactions with memory addition status
    """
    
    results = []
    
    for interaction in batch_interactions:
        try:
            # Add interaction to memory
            memory.add_chunk(
                text=interaction['item_information'],
                chunk_type='interaction',
                chunk_id=interaction['int_id'],
                metadata={
                    'user_id': interaction['user_id'],
                    'true_rating': interaction['true_rating'],
                    'true_review': interaction['true_review'],
                }
            )
            
            # Add memory addition status
            result = {
                **interaction,
                'memory_added': True
            }
            
            if debug:
                print(f"  ✓ Added interaction {interaction['int_id']} to memory")
                
        except Exception as e:
            result = {
                **interaction,
                'memory_added': False,
                'memory_error': str(e)
            }
            
            if debug:
                print(f"  ✗ Failed to add interaction {interaction['int_id']} to memory: {e}")
        
        results.append(result)
    
    return results


def run_simple_baseline(
    input_jsonl_path: str,
    output_jsonl_path: str,
    memory_instance: MemoryManager,
    llm_client: ChatOpenAI = None,
    batch_size: int = 5,
    debug: bool = False,
    write_every_n_batches: int = 5,
    progress_log_path: str = "simple_baseline_progress.txt",
    memory_log_path: str = "memory_additions.txt"
) -> None:
    """
    Run simple baseline agent in batches.
    
    Args:
        input_jsonl_path: Path to input JSONL file
        output_jsonl_path: Path to output JSONL file
        memory_instance: MemoryManager instance
        llm_client: LLM client for predictions (will be initialized if None)
        batch_size: Number of interactions per batch
        debug: Whether to print debug information
        write_every_n_batches: How often to write intermediate results
        progress_log_path: Path to progress log file
        memory_log_path: Path to memory additions log file
    """
    
    # Initialize LLM client if not provided
    if llm_client is None:
        print("Initializing LLM client...")
        llm_client = initialize_llm_client()
    
    # Set global memory instance
    set_memory(memory_instance)
    
    print(f"Loading data from {input_jsonl_path}...")
    all_interactions = load_jsonl(input_jsonl_path)
    print(f"Loaded {len(all_interactions)} interactions")

    batches = create_batches(all_interactions, batch_size)
    print(f"Created {len(batches)} batches of size {batch_size}")

    # Determine starting batch index
    start_batch = 0
    if os.path.exists(progress_log_path):
        with open(progress_log_path, "r") as f:
            content = f.read().strip()
            if content:
                try:
                    start_batch = int(content)
                    print(f"Resuming from batch {start_batch + 1}...")
                except ValueError:
                    print("⚠️ Invalid number in progress log. Starting from scratch.")
            else:
                print("📄 Progress log is empty. Starting from scratch.")

    all_final_results = []
    memory_additions_count = 0

    for batch_idx, batch in tqdm(enumerate(batches), desc="Processing batches", total=len(batches)):
        if batch_idx < start_batch:
            continue

        print(f"\n{'='*50}")
        print(f"Processing Batch {batch_idx + 1}/{len(batches)} ({len(batch)} interactions)")
        print(f"{'='*50}")

        # Phase 1: Simple Prediction
        print("Phase 1: Simple Prediction...")
        batch_predictions = []
        for idx, interaction in tqdm(enumerate(batch), desc="Making predictions", total=len(batch)):
            if debug:
                print(f"  Processing interaction {idx + 1}/{len(batch)} (User: {interaction['user_id']}, Interaction: {interaction['int_id']})")
            
            try:
                prediction_result = simple_predict(interaction, memory_instance, llm_client, debug=debug)
                batch_predictions.append(prediction_result)
                
                if debug:
                    print(f"    ✓ Prediction completed - Predicted rating: {prediction_result['predicted_rating']}")
                    
            except Exception as e:
                print(f"    ✗ Error in prediction: {e}")
                batch_predictions.append({**interaction, 'prediction_error': str(e)})

        # Phase 2: Add to Memory
        print("\nPhase 2: Adding to Memory...")
        try:
            batch_final_results = add_interactions_to_memory(batch_predictions, memory_instance, debug=debug)
            all_final_results.extend(batch_final_results)
            
            # Count successful memory additions
            successful_additions = sum(1 for r in batch_final_results if r.get('memory_added', False))
            memory_additions_count += successful_additions
            
            print(f"  ✓ Batch {batch_idx + 1} memory additions completed ({successful_additions}/{len(batch)} successful)")

            # Stats
            valid_predictions = [r for r in batch_final_results if 'predicted_rating' in r and r['predicted_rating'] is not None]
            if valid_predictions:
                avg_pred = sum(float(r['predicted_rating']) for r in valid_predictions) / len(valid_predictions)
                avg_true = sum(float(r['true_rating']) for r in valid_predictions) / len(valid_predictions)
                print(f"  📊 Batch {batch_idx + 1} Stats:")
                print(f"      Average Predicted Rating: {avg_pred:.2f}")
                print(f"      Average True Rating: {avg_true:.2f}")
                print(f"      Prediction Error (MAE): {abs(avg_pred - avg_true):.2f}")
                
        except Exception as e:
            print(f"  ✗ Error in memory addition: {e}")
            batch_final_results = [{**interaction, 'memory_error': str(e)} for interaction in batch_predictions]
            all_final_results.extend(batch_final_results)

        # Save progress log
        with open(progress_log_path, "w") as f:
            f.write(str(batch_idx + 1))

        # Save memory additions log
        with open(memory_log_path, "w") as f:
            f.write(f"Total interactions added to memory: {memory_additions_count}\n")
            f.write(f"Last processed batch: {batch_idx + 1}\n")

        # Write intermediate results every N batches
        if (batch_idx + 1) % write_every_n_batches == 0:
            print(f"💾 Writing intermediate results to {output_jsonl_path}...")
            existing = load_jsonl(output_jsonl_path) if os.path.exists(output_jsonl_path) else []
            save_jsonl(existing + all_final_results, output_jsonl_path)
            all_final_results = []

    # Final write
    if all_final_results:
        print(f"💾 Writing remaining {len(all_final_results)} results...")
        existing = load_jsonl(output_jsonl_path) if os.path.exists(output_jsonl_path) else []
        save_jsonl(existing + all_final_results, output_jsonl_path)

    print("✓ Simple baseline processing completed successfully!")
    print(f"📊 Final Stats:")
    print(f"    Total interactions added to memory: {memory_additions_count}")


def run_single_simple_interaction(input_json: dict, memory_instance: MemoryManager, llm_client: ChatOpenAI = None) -> dict:
    """
    Legacy function to process a single interaction with simple baseline (for backward compatibility).
    """
    # Initialize LLM client if not provided
    if llm_client is None:
        llm_client = initialize_llm_client()
    
    # Process prediction
    prediction_result = simple_predict(input_json, memory_instance, llm_client)
    
    # Add to memory
    memory_results = add_interactions_to_memory([prediction_result], memory_instance)
    
    return memory_results[0]
