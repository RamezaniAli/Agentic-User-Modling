import json
import os
from typing import List, Dict, Any
from tqdm import tqdm


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
    
    # Convert to output format (keeping true values as predicted for initial interactions)
    formatted_initial = []
    for interaction in initial_interactions:
        formatted_interaction = {
            'user_id': interaction["user_id"],
            'user_information': interaction["user_information"],
            'int_id': interaction["int_id"],
            'item_information': interaction["item_information"],
            'true_rating': interaction["true_rating"],
            'predicted_rating': interaction["true_rating"],  # Use true as predicted for initial
            'initial_interaction': True
        }
        formatted_initial.append(formatted_interaction)
    
    # Save to output file
    save_jsonl(formatted_initial, output_jsonl_path)
    
    if debug:
        print(f"Initialized output file with {len(formatted_initial)} interactions")


def calculate_average_rating(output_jsonl_path: str, debug: bool = False) -> float:
    """
    Calculate average rating from all interactions in output file.
    
    Args:
        output_jsonl_path: Path to output JSONL file
        debug: Whether to print debug information
    
    Returns:
        Average rating from all previous interactions
    """
    
    if not os.path.exists(output_jsonl_path):
        if debug:
            print("No output file exists yet")
        return None
    
    # Load all previous interactions
    previous_interactions = load_jsonl(output_jsonl_path)
    
    if not previous_interactions:
        if debug:
            print("No previous interactions found")
        return None
    
    # Calculate average of true ratings
    valid_ratings = []
    for interaction in previous_interactions:
        if 'true_rating' in interaction and interaction['true_rating'] is not None:
            try:
                rating = float(interaction['true_rating'])
                valid_ratings.append(rating)
            except (ValueError, TypeError):
                continue
    
    if not valid_ratings:
        if debug:
            print("No valid ratings found in previous interactions")
        return None
    
    average_rating = sum(valid_ratings) / len(valid_ratings)
    
    if debug:
        print(f"Calculated average rating: {average_rating:.2f} from {len(valid_ratings)} interactions")
    
    return average_rating


def rule_based_predict(
    current_interaction: Dict[Any, Any],
    average_rating: float,
    debug: bool = False
) -> Dict[Any, Any]:
    """
    Rule-based prediction function that uses average rating as prediction.
    
    Args:
        current_interaction: Current interaction data to predict
        average_rating: Average rating from previous interactions
        debug: Whether to print debug information
    
    Returns:
        Dictionary with prediction results
    """
    
    # If no average available, use true rating as prediction
    if average_rating is None:
        predicted_rating = current_interaction["true_rating"]
        if debug:
            print(f"  No average available, using true rating: {predicted_rating}")
    else:
        predicted_rating = average_rating
        if debug:
            print(f"  Using average rating as prediction: {predicted_rating}")
    
    # Create result dictionary
    result = {
        'user_id': current_interaction["user_id"],
        'user_information': current_interaction["user_information"],
        'int_id': current_interaction["int_id"],
        'item_information': current_interaction["item_information"],
        'true_rating': current_interaction["true_rating"],
        'predicted_rating': predicted_rating
    }
    
    return result


def run_rule_based_baseline(
    input_jsonl_path: str,
    output_jsonl_path: str,
    start_index: int = 20,
    batch_size: int = 5,
    debug: bool = False,
    progress_log_path: str = "rule_based_progress.txt"
) -> None:
    """
    Run rule-based baseline method on a JSONL file with batch processing.
    
    Args:
        input_jsonl_path: Path to input JSONL file
        output_jsonl_path: Path to output JSONL file
        start_index: Index to start predictions from (20 or 50)
        batch_size: Number of interactions per batch
        debug: Whether to print debug information
        progress_log_path: Path to progress log file
    """
    
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
        
        # Calculate average rating from all previous interactions in output file
        average_rating = calculate_average_rating(output_jsonl_path, debug)
        
        batch_results = []
        
        for idx, interaction in tqdm(enumerate(batch), desc="Making predictions", total=len(batch)):
            current_idx = start_index + batch_idx * batch_size + idx
            
            if debug:
                print(f"  Processing interaction {current_idx + 1} (int_id: {interaction.get('int_id', 'N/A')})")
            
            try:
                prediction_result = rule_based_predict(
                    current_interaction=interaction,
                    average_rating=average_rating,
                    debug=debug
                )
                
                batch_results.append(prediction_result)
                
                if debug:
                    print(f"    ✓ Prediction completed - Rating: {prediction_result['predicted_rating']}")
                    
            except Exception as e:
                print(f"    ✗ Error in prediction: {e}")
                error_result = {
                    **interaction,
                    'predicted_rating': interaction["true_rating"],  # Fallback to true rating
                    'prediction_error': str(e)
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
    
    print("✓ Rule-based baseline processing completed successfully!")
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


def run_rule_based_baseline_single_interaction(
    interaction: Dict[Any, Any],
    previous_interactions: List[Dict[Any, Any]]
) -> Dict[Any, Any]:
    """
    Process a single interaction with rule-based method (for compatibility).
    
    Args:
        interaction: Current interaction data
        previous_interactions: List of previous interactions
    
    Returns:
        Dictionary with prediction results
    """
    
    # Calculate average from previous interactions
    if not previous_interactions:
        average_rating = None
    else:
        valid_ratings = []
        for prev_interaction in previous_interactions:
            if 'true_rating' in prev_interaction and prev_interaction['true_rating'] is not None:
                try:
                    rating = float(prev_interaction['true_rating'])
                    valid_ratings.append(rating)
                except (ValueError, TypeError):
                    continue
        
        average_rating = sum(valid_ratings) / len(valid_ratings) if valid_ratings else None
    
    return rule_based_predict(
        current_interaction=interaction,
        average_rating=average_rating
    )