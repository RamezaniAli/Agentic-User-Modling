import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from baselines.direct_context.direct_config import BASE_URL, LLM_MODEL_NAME, MODEL_TEMP
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


def append_jsonl(data: List[Dict[Any, Any]], file_path: str) -> None:
    """Append data to JSONL file."""
    with open(file_path, 'a', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def get_last_n_lines_from_jsonl(file_path: str, n: int = 5) -> List[Dict[Any, Any]]:
    """Get the last n lines from a JSONL file."""
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Get last n non-empty lines
        last_lines = []
        for line in reversed(lines):
            line = line.strip()
            if line:
                last_lines.append(json.loads(line))
                if len(last_lines) >= n:
                    break
        
        # Reverse to get chronological order
        return list(reversed(last_lines))
    
    except Exception as e:
        print(f"Error reading last {n} lines from {file_path}: {e}")
        return []


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
    
    # Try to load existing human prompt or create default
    human_prompt_path = os.path.join(current_dir, "prompts", "human_prompt.txt")
    with open(human_prompt_path, "r", encoding="utf-8") as f:
        human_prompt = f.read()

    return sys_prompt, human_prompt



def format_previous_interactions(interactions: List[Dict[Any, Any]]) -> str:
    """Format previous interactions for the prompt."""
    if not interactions:
        return "No previous interactions available."
    
    formatted = []
    for i, interaction in enumerate(interactions, 1):

        formatted.append(f"Interaction {i}:")
        formatted.append(f"  Item: {interaction['item_information']}")
        formatted.append(f"  Rating: {interaction.get('true_rating')}")
        formatted.append(f"  Review: {interaction.get('true_review')}")
        formatted.append("")
    
    return "\n".join(formatted)


def direct_predict(
    current_interaction: Dict[Any, Any],
    previous_interactions: List[Dict[Any, Any]],
    llm_client: ChatOpenAI,
    debug: bool = False
) -> Dict[Any, Any]:
    """
    Direct prediction function that uses only the last 5 interactions.
    
    Args:
        current_interaction: Current interaction data to predict
        previous_interactions: List of previous interactions (max 5)
        llm_client: LLM client for making predictions
        debug: Whether to print debug information
    
    Returns:
        Dictionary with prediction results
    """
    
    # Take only last 5 interactions
    last_5_interactions = previous_interactions[-5:] if len(previous_interactions) >= 5 else previous_interactions
    
    if debug:
        print(f"  Using {len(last_5_interactions)} previous interactions for prediction")
    
    # Load prompts
    sys_prompt, human_prompt_template = load_prompts()
    
    # Format previous interactions
    formatted_previous = format_previous_interactions(last_5_interactions)
    
    # Fill the human prompt template
    filled_human_prompt = human_prompt_template.format(
        user_information=current_interaction["user_information"],
        previous_interactions=formatted_previous,
        item_information=current_interaction["item_information"]
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
        'user_id': current_interaction["user_id"],
        'user_information': current_interaction["user_information"],
        'int_id': current_interaction["int_id"],
        'item_information': current_interaction["item_information"],
        'true_rating': current_interaction["true_rating"],
        'true_review': current_interaction["true_review"],
        'predicted_rating': predicted_rating,
        'predicted_review': predicted_review,
        'previous_interactions_count': len(last_5_interactions)
    }
    
    return result


def create_batches(data: List[Dict[Any, Any]], batch_size: int) -> List[List[Dict[Any, Any]]]:
    """Split data into batches of specified size."""
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches


def initialize_output_file(input_jsonl_path: str, output_jsonl_path: str, start_index: int, debug: bool = False) -> None:
    """
    Initialize the output file with the first start_index interactions.
    
    Args:
        input_jsonl_path: Path to input JSONL file
        output_jsonl_path: Path to output JSONL file
        start_index: Number of initial interactions to copy
        debug: Whether to print debug information
    """
    
    if os.path.exists(output_jsonl_path):
        existing_data = load_jsonl(output_jsonl_path)
        if len(existing_data) >= start_index:
            if debug:
                print(f"Output file already initialized with {len(existing_data)} interactions")
            return
    
    # Load initial interactions
    all_interactions = load_jsonl(input_jsonl_path)
    
    if len(all_interactions) < start_index:
        raise ValueError(f"Input file has only {len(all_interactions)} interactions, cannot initialize with {start_index}")
    
    # Copy first start_index interactions to output file
    initial_interactions = all_interactions[:start_index]
    
    # Convert to output format (keeping true values as both true and predicted for initial interactions)
    formatted_initial = []
    for interaction in initial_interactions:
        formatted_interaction = {
            'user_id': interaction["user_id"],
            'user_information': interaction["user_information"],
            'int_id': interaction["int_id"],
            'item_information': interaction["item_information"],
            'true_rating': interaction["true_rating"],
            'true_review': interaction["true_review"],
            'predicted_rating': interaction["true_rating"],  # Use true as predicted for initial
            'predicted_review': interaction["true_review"],  # Use true as predicted for initial
            'previous_interactions_count': 0,
            'initial_interaction': True
        }
        formatted_initial.append(formatted_interaction)
    
    # Save to output file
    save_jsonl(formatted_initial, output_jsonl_path)
    
    if debug:
        print(f"Initialized output file with {len(formatted_initial)} interactions")


def run_direct_baseline(
    input_jsonl_path: str,
    output_jsonl_path: str,
    start_index: int = 20,
    llm_client: ChatOpenAI = None,
    batch_size: int = 5,
    debug: bool = False,
    progress_log_path: str = "direct_baseline_progress.txt"
) -> None:
    """
    Run direct baseline method on a JSONL file with batch processing.
    
    Args:
        input_jsonl_path: Path to input JSONL file
        output_jsonl_path: Path to output JSONL file
        start_index: Index to start predictions from (20 or 50)
        llm_client: LLM client for predictions (will be initialized if None)
        batch_size: Number of interactions per batch
        debug: Whether to print debug information
        progress_log_path: Path to progress log file
    """
    
    # Initialize LLM client if not provided
    if llm_client is None:
        print("Initializing LLM client...")
        llm_client = initialize_llm_client()
    
    print(f"Loading data from {input_jsonl_path}...")
    all_interactions = load_jsonl(input_jsonl_path)
    print(f"Loaded {len(all_interactions)} interactions")
    
    if len(all_interactions) <= start_index:
        print(f"Error: File has only {len(all_interactions)} interactions, cannot start from index {start_index}")
        return
    
    # Initialize output file with first start_index interactions
    print(f"Initializing output file with first {start_index} interactions...")
    initialize_output_file(input_jsonl_path, output_jsonl_path, start_index, debug)
    
    # Get interactions to process (from start_index onwards)
    interactions_to_process = all_interactions[start_index:]
    print(f"Processing {len(interactions_to_process)} interactions starting from index {start_index}")
    
    # Create batches
    batches = create_batches(interactions_to_process, batch_size)
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
    
    for batch_idx, batch in tqdm(enumerate(batches), desc="Processing batches", total=len(batches)):
        if batch_idx < start_batch:
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing Batch {batch_idx + 1}/{len(batches)} ({len(batch)} interactions)")
        print(f"{'='*50}")
        
        # Get last 5 interactions from output file for context
        previous_interactions = get_last_n_lines_from_jsonl(output_jsonl_path, 5)
        
        if debug:
            print(f"  Retrieved {len(previous_interactions)} interactions from output file for context")
        
        batch_results = []
        
        for idx, interaction in tqdm(enumerate(batch), desc="Making predictions", total=len(batch)):
            current_idx = start_index + batch_idx * batch_size + idx
            
            if debug:
                print(f"  Processing interaction {current_idx + 1} (int_id: {interaction.get('int_id', 'N/A')})")
            
            try:
                prediction_result = direct_predict(
                    current_interaction=interaction,
                    previous_interactions=previous_interactions,
                    llm_client=llm_client,
                    debug=debug
                )
                
                batch_results.append(prediction_result)
                
                if debug:
                    print(f"    ✓ Prediction completed - Rating: {prediction_result['predicted_rating']} - Review length: {len(prediction_result['predicted_review'])}")
                    
            except Exception as e:
                print(f"    ✗ Error in prediction: {e}")
                error_result = {
                    **interaction,
                    'predicted_rating': 2.5,
                    'predicted_review': f"Prediction failed due to error: {str(e)}",
                    'prediction_error': str(e),
                    'previous_interactions_count': len(previous_interactions)
                }
                batch_results.append(error_result)
                        
        # Write the complete batch to output file
        print(f"💾 Writing batch {batch_idx + 1} ({len(batch_results)} interactions) to output file...")
        append_jsonl(batch_results, output_jsonl_path)
        
        # Save progress log
        with open(progress_log_path, "w") as f:
            f.write(str(batch_idx + 1))
        
        # Calculate and display batch stats
        valid_predictions = [r for r in batch_results if 'predicted_rating' in r and r['predicted_rating'] is not None]
        if valid_predictions:
            avg_pred = sum(float(r['predicted_rating']) for r in valid_predictions) / len(valid_predictions)
            avg_true = sum(float(r['true_rating']) for r in valid_predictions) / len(valid_predictions)
            print(f"  📊 Batch {batch_idx + 1} Stats:")
            print(f"      Average Predicted Rating: {avg_pred:.2f}")
            print(f"      Average True Rating: {avg_true:.2f}")
            print(f"      Prediction Error (MAE): {abs(avg_pred - avg_true):.2f}")
    
    print("✓ Direct baseline processing completed successfully!")
    print(f"📊 Final Stats:")
    print(f"    Total interactions processed: {len(interactions_to_process)}")
    print(f"    Start index: {start_index}")
    
    # Calculate overall stats
    final_results = load_jsonl(output_jsonl_path)
    # Only count results after start_index
    prediction_results = final_results[start_index:]
    
    if prediction_results:
        valid_predictions = [r for r in prediction_results if 'predicted_rating' in r and r['predicted_rating'] is not None]
        if valid_predictions:
            avg_pred = sum(float(r['predicted_rating']) for r in valid_predictions) / len(valid_predictions)
            avg_true = sum(float(r['true_rating']) for r in valid_predictions) / len(valid_predictions)
            print(f"    Overall Average Predicted Rating: {avg_pred:.2f}")
            print(f"    Overall Average True Rating: {avg_true:.2f}")
            print(f"    Overall Prediction Error (MAE): {abs(avg_pred - avg_true):.2f}")


def run_direct_baseline_single_interaction(
    interaction: Dict[Any, Any],
    previous_interactions: List[Dict[Any, Any]],
    llm_client: ChatOpenAI = None
) -> Dict[Any, Any]:
    """
    Process a single interaction with direct baseline method (for compatibility).
    
    Args:
        interaction: Current interaction data
        previous_interactions: List of previous interactions
        llm_client: LLM client for predictions
    
    Returns:
        Dictionary with prediction results
    """
    
    # Initialize LLM client if not provided
    if llm_client is None:
        llm_client = initialize_llm_client()
    
    return direct_predict(
        current_interaction=interaction,
        previous_interactions=previous_interactions,
        llm_client=llm_client
    )
