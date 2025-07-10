import json
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from evaluation.eval_config import BASE_URL, LLM_MODEL_NAME, MODEL_TEMP
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


def save_json(data: Dict[Any, Any], file_path: str) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def initialize_llm_client() -> ChatOpenAI:
    """Initialize the LLM client."""
    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=MODEL_TEMP,
        base_url=BASE_URL,
    )


def load_evaluation_prompt() -> str:
    """Load the evaluation prompt from the human_prompt.txt file."""
    current_dir = os.path.dirname(__file__)
    
    # Load human prompt (the evaluation prompt)
    human_prompt_path = os.path.join(current_dir, "prompts", "human_prompt.txt")
    with open(human_prompt_path, "r", encoding="utf-8") as f:
        human_prompt = f.read()
    
    return human_prompt


def calculate_mae_rating(true_ratings: List[float], predicted_ratings: List[float]) -> List[float]:
    """Calculate Mean Absolute Error for each rating pair."""
    mae_values = []
    for true_rating, predicted_rating in zip(true_ratings, predicted_ratings):
        try:
            mae = abs(float(true_rating) - float(predicted_rating))
            mae_values.append(mae)
        except (ValueError, TypeError):
            # Handle cases where ratings are not valid numbers
            mae_values.append(None)
    return mae_values


def evaluate_review_similarity(
    book_info: str,
    true_review: str,
    predicted_review: str,
    llm_client: ChatOpenAI,
    evaluation_prompt: str,
    debug: bool = False
) -> Optional[Dict[str, int]]:
    """
    Evaluate similarity between true and predicted reviews using LLM.
    
    Args:
        book_info: Book information
        true_review: True review text
        predicted_review: Predicted review text
        llm_client: LLM client for evaluation
        evaluation_prompt: Prompt template for evaluation
        debug: Whether to print debug information
    
    Returns:
        Dictionary with relevance, sentiment, and emotion scores or None if failed
    """
    
    try:
        # Fill the evaluation prompt template
        filled_prompt = evaluation_prompt.format(
            book_info=book_info,
            review_1=true_review,
            review_2=predicted_review
        )
        
        # Create message (no system message needed based on your prompt)
        human_msg = HumanMessage(content=filled_prompt)
        
        # Call LLM
        response = llm_client.invoke([human_msg])
        
        if debug:
            print(f"  LLM Response: {response.content[:200]}...")
        
        # Parse LLM response
        required_keys = ["relevance", "sentiment", "emotion"]
        result = safe_json_parse(response.content, required_keys)
        
        if result is None:
            if debug:
                print("ERROR: Could not parse valid JSON from LLM response.")
                print(f"Response preview: {response.content[:500]}...")
            return None
        
        # Validate that all values are integers between 1-5
        for key in required_keys:
            if key not in result:
                if debug:
                    print(f"ERROR: Missing key '{key}' in LLM response")
                return None
            
            try:
                value = int(result[key])
                if not (1 <= value <= 5):
                    if debug:
                        print(f"ERROR: Value for '{key}' is {value}, must be between 1-5")
                    return None
                result[key] = value
            except (ValueError, TypeError):
                if debug:
                    print(f"ERROR: Invalid value for '{key}': {result[key]}")
                return None
        
        if debug:
            print(f"  ✓ Successfully parsed LLM response - Relevance: {result['relevance']}, Sentiment: {result['sentiment']}, Emotion: {result['emotion']}")
        
        return result
    
    except Exception as e:
        if debug:
            print(f"ERROR: LLM call failed: {e}")
        return None


def create_batches(data: List[Dict[Any, Any]], batch_size: int) -> List[List[Dict[Any, Any]]]:
    """Split data into batches of specified size."""
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches


def save_progress(
    current_results: Dict[str, List],
    output_path: str,
    progress_log_path: str,
    batch_idx: int
) -> None:
    """Save current results and progress."""
    # Save current results
    save_json(current_results, output_path)
    
    # Save progress log
    with open(progress_log_path, "w") as f:
        f.write(str(batch_idx))


def run_evaluation(
    input_jsonl_path: str,
    output_json_path: str,
    start_index: int = 0,
    llm_client: ChatOpenAI = None,
    batch_size: int = 10,
    debug: bool = False,
    progress_log_path: str = "evaluation_progress.txt"
) -> None:
    """
    Run evaluation process on a JSONL file with batch processing.
    
    Args:
        input_jsonl_path: Path to input JSONL file
        output_json_path: Path to output JSON file
        start_index: Index to start real evaluation from (before this: MAE=0, scores=5)
        llm_client: LLM client for evaluation (will be initialized if None)
        batch_size: Number of interactions per batch
        debug: Whether to print debug information
        progress_log_path: Path to progress log file
    """
    
    
    print(f"Loading data from {input_jsonl_path}...")
    all_interactions = load_jsonl(input_jsonl_path)
    print(f"Loaded {len(all_interactions)} interactions")
    
    if len(all_interactions) == 0:
        print("Error: No interactions found in input file")
        return
    
        # Initialize LLM client if not provided (only if we need it)
    if start_index < len(all_interactions):
        if llm_client is None:
            print("Initializing LLM client...")
            llm_client = initialize_llm_client()
        
        # Load evaluation prompt
        print("Loading evaluation prompt...")
        evaluation_prompt = load_evaluation_prompt()
    else:
        evaluation_prompt = None
        
    # Initialize results structure
    results = {
        "MAE_Rating": [],
        "relevance": [],
        "sentiment": [],
        "emotion": []
    }
    
    # Handle interactions before start_index
    if start_index > 0:
        print(f"Setting default values for first {start_index} interactions...")
        for i in range(min(start_index, len(all_interactions))):
            results["MAE_Rating"].append(0)
            results["relevance"].append(5)
            results["sentiment"].append(5)
            results["emotion"].append(5)
        
        if debug:
            print(f"  Added {min(start_index, len(all_interactions))} default entries")
    
    # Get interactions to actually evaluate
    interactions_to_evaluate = all_interactions[start_index:] if start_index < len(all_interactions) else []
    
    if len(interactions_to_evaluate) == 0:
        print(f"All {len(all_interactions)} interactions are before start_index {start_index}. Using default values only.")
        save_json(results, output_json_path)
        return
    
    print(f"Will evaluate {len(interactions_to_evaluate)} interactions starting from index {start_index}")
    
    # Create batches from interactions to evaluate
    batches = create_batches(interactions_to_evaluate, batch_size)
    print(f"Created {len(batches)} batches of size {batch_size} for evaluation")
    
    # Determine starting batch index
    start_batch = 0
    if os.path.exists(progress_log_path):
        with open(progress_log_path, "r") as f:
            content = f.read().strip()
            if content:
                try:
                    start_batch = int(content)
                    print(f"Resuming from batch {start_batch + 1}...")
                    
                    # Load existing results if available
                    if os.path.exists(output_json_path):
                        with open(output_json_path, "r", encoding="utf-8") as f:
                            results = json.load(f)
                        print(f"Loaded existing results with {len(results['MAE_Rating'])} entries")
                    
                except ValueError:
                    print("⚠️ Invalid number in progress log. Starting from scratch.")
    
    # Process batches
    for batch_idx, batch in tqdm(enumerate(batches), desc="Processing batches", total=len(batches)):
        if batch_idx < start_batch:
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing Batch {batch_idx + 1}/{len(batches)} ({len(batch)} interactions)")
        print(f"{'='*50}")
        
        batch_mae_ratings = []
        batch_relevance_scores = []
        batch_sentiment_scores = []
        batch_emotion_scores = []
        
        for idx, interaction in tqdm(enumerate(batch), desc="Evaluating interactions", total=len(batch)):
            current_idx = start_index + batch_idx * batch_size + idx
            
            if debug:
                print(f"  Processing interaction {current_idx + 1} (int_id: {interaction.get('int_id', 'N/A')})")
            
            try:
                # Extract required fields
                true_rating = interaction.get('true_rating')
                predicted_rating = interaction.get('predicted_rating')
                true_review = interaction.get('true_review', '')
                predicted_review = interaction.get('predicted_review', '')
                item_information = interaction.get('item_information', '')
                
                # Calculate MAE for rating
                if true_rating is not None and predicted_rating is not None:
                    mae = abs(float(true_rating) - float(predicted_rating))
                    batch_mae_ratings.append(mae)
                    
                    if debug:
                        print(f"    Rating MAE: {mae:.3f} (True: {true_rating}, Predicted: {predicted_rating})")
                else:
                    batch_mae_ratings.append(None)
                    if debug:
                        print(f"    Rating MAE: None (Missing ratings)")
                
                # Evaluate review similarity using LLM only if reviews exist
                if true_review and predicted_review and item_information:
                    similarity_scores = evaluate_review_similarity(
                        book_info=item_information,
                        true_review=true_review,
                        predicted_review=predicted_review,
                        llm_client=llm_client,
                        evaluation_prompt=evaluation_prompt,
                        debug=debug
                    )
                    
                    if similarity_scores:
                        batch_relevance_scores.append(similarity_scores['relevance'])
                        batch_sentiment_scores.append(similarity_scores['sentiment'])
                        batch_emotion_scores.append(similarity_scores['emotion'])
                        
                        if debug:
                            print(f"    Review Scores - Relevance: {similarity_scores['relevance']}, Sentiment: {similarity_scores['sentiment']}, Emotion: {similarity_scores['emotion']}")
                    else:
                        batch_relevance_scores.append(None)
                        batch_sentiment_scores.append(None)
                        batch_emotion_scores.append(None)
                        if debug:
                            print(f"    Review Scores: Failed to evaluate")
                else:
                    # No reviews available - skip LLM evaluation, append None
                    batch_relevance_scores.append(None)
                    batch_sentiment_scores.append(None)
                    batch_emotion_scores.append(None)
                    if debug:
                        print(f"    Review Scores: None (No reviews to evaluate)")
                        
            except Exception as e:
                print(f"    ✗ Error in evaluation: {e}")
                batch_mae_ratings.append(None)
                batch_relevance_scores.append(None)
                batch_sentiment_scores.append(None)
                batch_emotion_scores.append(None)
        
        # Add batch results to overall results
        results["MAE_Rating"].extend(batch_mae_ratings)
        results["relevance"].extend(batch_relevance_scores)
        results["sentiment"].extend(batch_sentiment_scores)
        results["emotion"].extend(batch_emotion_scores)
        
        # Save progress
        save_progress(results, output_json_path, progress_log_path, batch_idx + 1)
        
        # Calculate and display batch stats
        valid_mae = [mae for mae in batch_mae_ratings if mae is not None]
        valid_relevance = [score for score in batch_relevance_scores if score is not None]
        valid_sentiment = [score for score in batch_sentiment_scores if score is not None]
        valid_emotion = [score for score in batch_emotion_scores if score is not None]
        
        print(f"  📊 Batch {batch_idx + 1} Stats:")
        if valid_mae:
            print(f"      Average MAE Rating: {sum(valid_mae) / len(valid_mae):.3f}")
        if valid_relevance:
            print(f"      Average Relevance Score: {sum(valid_relevance) / len(valid_relevance):.2f}")
        if valid_sentiment:
            print(f"      Average Sentiment Score: {sum(valid_sentiment) / len(valid_sentiment):.2f}")
        if valid_emotion:
            print(f"      Average Emotion Score: {sum(valid_emotion) / len(valid_emotion):.2f}")
        
        print(f"      Valid evaluations: {len(valid_mae)}/{len(batch)} MAE, {len(valid_relevance)}/{len(batch)} Reviews")
    
    # Final save
    save_json(results, output_json_path)
    
    print("✓ Evaluation completed successfully!")
    print(f"📊 Final Stats:")
    print(f"    Total interactions: {len(all_interactions)}")
    print(f"    Start index: {start_index}")
    print(f"    Interactions with default values: {min(start_index, len(all_interactions))}")
    print(f"    Interactions evaluated: {len(interactions_to_evaluate)}")
    
    # Calculate overall stats (excluding default values)
    actual_mae = results["MAE_Rating"][start_index:]
    actual_relevance = results["relevance"][start_index:]
    actual_sentiment = results["sentiment"][start_index:]
    actual_emotion = results["emotion"][start_index:]
    
    valid_mae = [mae for mae in actual_mae if mae is not None]
    valid_relevance = [score for score in actual_relevance if score is not None]
    valid_sentiment = [score for score in actual_sentiment if score is not None]
    valid_emotion = [score for score in actual_emotion if score is not None]
    
    if valid_mae:
        print(f"    Overall Average MAE Rating: {sum(valid_mae) / len(valid_mae):.3f}")
    if valid_relevance:
        print(f"    Overall Average Relevance Score: {sum(valid_relevance) / len(valid_relevance):.2f}")
    if valid_sentiment:
        print(f"    Overall Average Sentiment Score: {sum(valid_sentiment) / len(valid_sentiment):.2f}")
    if valid_emotion:
        print(f"    Overall Average Emotion Score: {sum(valid_emotion) / len(valid_emotion):.2f}")
    
    print(f"    Valid evaluations: {len(valid_mae)}/{len(interactions_to_evaluate)} MAE, {len(valid_relevance)}/{len(interactions_to_evaluate)} Reviews")
    print(f"    Results saved to: {output_json_path}")


def main():
    """Main function to run the evaluation."""
    # Example usage
    input_file = "path/to/your/input.jsonl"
    output_file = "path/to/your/output.json"
    
    run_evaluation(
        input_jsonl_path=input_file,
        output_json_path=output_file,
        start_index=20,  # Start real evaluation from index 20
        batch_size=10,
        debug=True
    )


if __name__ == "__main__":
    main()