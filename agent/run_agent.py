from langchain_core.messages import HumanMessage
import os
from agent.react_agent import build_react_graph, ReactState
from agent.reflect_agent import build_reflect_graph, ReflectState
import json
from typing import List, Dict, Any
from tqdm import tqdm

react_graph = build_react_graph()
reflect_graph = build_reflect_graph()


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


def process_single_interaction_react(interaction: Dict[Any, Any], debug) -> Dict[Any, Any]:
    """Process a single interaction through the ReAct agent."""
    
    # Load ReAct prompt template
    current_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(current_dir, "prompts", "react_human_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        react_template = file.read()

    # Fill the ReAct prompt
    filled_react_prompt = react_template.format(
        user_id=interaction["user_id"],
        user_information=interaction["user_information"],
        item_id=interaction["item_id"],
        item_information=interaction["item_information"]
    )
    
    first_react_msg = HumanMessage(content=filled_react_prompt)

    # Create ReAct state
    react_state: ReactState = {
        "user_id": interaction.get("user_id"),
        "user_information": interaction.get("user_information", {}),
        "react_messages": [first_react_msg],
    }

    # Run ReAct agent
    react_result = react_graph.invoke(react_state)

    if debug == "react" or debug == "both":
        for msg in react_result['react_messages']:
            msg.pretty_print()

    # Combine original interaction data with ReAct results
    processed_interaction = {
        'user_id': interaction["user_id"],
        'user_information': interaction["user_information"],
        'item_id': interaction["item_id"],
        'item_information': interaction["item_information"],
        'true_rating': interaction["true_rating"],
        'true_review': interaction["true_review"],
        'persona': react_result["persona"],
        'predicted_rating': react_result['predicted_rating'],
        'predicted_review': react_result['predicted_review'],
        'retrieved_interactions': react_result['retrieved_interactions'],
        # 'react_messages': react_result['react_messages']  # Keep for debugging if needed
    }
    
    return processed_interaction


def process_batch_reflect(batch_interactions: List[Dict[Any, Any]], debug) -> List[Dict[Any, Any]]:
    """Process a batch of interactions through the Reflect agent."""
    
    # Load Reflect prompt template
    current_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(current_dir, "prompts", "reflect_human_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        reflect_template = file.read()

    final_results = []
    
    for interaction in batch_interactions:
        # Fill the Reflect prompt for this interaction
        filled_reflect_prompt = reflect_template.format(
            user_id=interaction["user_id"],
            user_information=interaction["user_information"],
            item_id=interaction["item_id"],
            item_information=interaction["item_information"],
            true_rating=interaction["true_rating"],
            true_review=interaction["true_review"],
            persona=interaction["persona"],
            predicted_rating=interaction['predicted_rating'],
            predicted_review=interaction['predicted_review'],
            retrieved_interactions=interaction['retrieved_interactions']
        )
        
        first_reflect_message = HumanMessage(content=filled_reflect_prompt)

        # Create Reflect state
        reflect_state: ReflectState = {
            "reflect_messages": [first_reflect_message],
        }
        
        # Run Reflect agent
        reflect_result = reflect_graph.invoke(reflect_state)

        if debug == "reflect" or debug == "both":
            for msg in reflect_result['reflect_messages']:
                msg.pretty_print()

        # Combine all data for final result
        final_result = {
            'user_id': interaction["user_id"],
            'user_information': interaction["user_information"],
            'item_id': interaction["item_id"],
            'item_information': interaction["item_information"],
            'true_rating': interaction["true_rating"],
            'true_review': interaction["true_review"],
            'persona': interaction["persona"],
            'predicted_rating': interaction['predicted_rating'],
            'predicted_review': interaction['predicted_review'],
            'retrieved_interactions': interaction['retrieved_interactions'],
            "updated_interaction": reflect_result['updated_interaction'],
            "updated_retrieved_interactions": reflect_result['updated_retrieved_interactions'],
            "updated_persona": reflect_result['updated_persona'],
            # 'reflect_messages': reflect_result['reflect_messages']  # Keep for debugging if needed
        }
        
        final_results.append(final_result)
    
    return final_results


def run_batch_agent(input_jsonl_path: str, output_jsonl_path: str, batch_size: int = 5, debug=None) -> None:
    """
    Main function to process JSONL file in batches.
    
    Args:
        input_jsonl_path: Path to input JSONL file
        output_jsonl_path: Path to output JSONL file
        batch_size: Number of interactions per batch
    """
    
    print(f"Loading data from {input_jsonl_path}...")
    
    # Load all interactions from JSONL
    all_interactions = load_jsonl(input_jsonl_path)
    print(f"Loaded {len(all_interactions)} interactions")
    
    # Create batches
    batches = create_batches(all_interactions, batch_size)
    print(f"Created {len(batches)} batches of size {batch_size}")
    
    all_final_results = []
    
    # Process each batch
    for batch_idx, batch in tqdm(enumerate(batches), desc="Processing batches", total=len(batches)):
        print(f"\n{'='*50}")
        print(f"Processing Batch {batch_idx + 1}/{len(batches)} ({len(batch)} interactions)")
        print(f"{'='*50}")
        
        # Phase 1: Process each interaction in batch through ReAct agent
        print("Phase 1: ReAct Agent Processing...")
        batch_react_results = []
        
        for idx, interaction in tqdm(enumerate(batch), desc="Processing interactions", total=len(batch)):
            print(f"  Processing interaction {idx + 1}/{len(batch)} (User: {interaction['user_id']}, Item: {interaction['item_id']})")
            
            try:
                react_result = process_single_interaction_react(interaction, debug=debug)
                batch_react_results.append(react_result)
                print(f"    ✓ ReAct completed - Predicted rating: {react_result['predicted_rating']}")
                
            except Exception as e:
                print(f"    ✗ Error in ReAct processing: {e}")
                # Add error placeholder to maintain batch structure
                error_result = {**interaction, 'error': str(e)}
                batch_react_results.append(error_result)
        
        # Phase 2: Process entire batch through Reflect agent
        print("\nPhase 2: Reflect Agent Processing...")
        
        try:
            batch_final_results = process_batch_reflect(batch_react_results, debug=debug)
            all_final_results.extend(batch_final_results)
            print(f"  ✓ Batch {batch_idx + 1} reflection completed")
            
            # Calculate and display batch statistics
            valid_predictions = [r for r in batch_final_results if 'predicted_rating' in r and r['predicted_rating'] is not None]
            if valid_predictions:
                avg_predicted_rating = sum(float(r['predicted_rating']) for r in valid_predictions) / len(valid_predictions)
                avg_true_rating = sum(float(r['true_rating']) for r in valid_predictions) / len(valid_predictions)
                print(f"  📊 Batch {batch_idx + 1} Stats:")
                print(f"      Average Predicted Rating: {avg_predicted_rating:.2f}")
                print(f"      Average True Rating: {avg_true_rating:.2f}")
                print(f"      Prediction Error (MAE): {abs(avg_predicted_rating - avg_true_rating):.2f}")
        
        except Exception as e:
            print(f"  ✗ Error in Reflect processing for batch {batch_idx + 1}: {e}")
            # Add error results to maintain structure
            error_results = [{**interaction, 'reflect_error': str(e)} for interaction in batch_react_results]
            all_final_results.extend(error_results)
    
    # Save all results to output JSONL
    print(f"\n{'='*50}")
    print(f"Saving {len(all_final_results)} results to {output_jsonl_path}...")
    save_jsonl(all_final_results, output_jsonl_path)
    print("✓ Processing completed successfully!")
    
    # Final statistics
    valid_results = [r for r in all_final_results if 'predicted_rating' in r and r['predicted_rating'] is not None]
    if valid_results:
        overall_avg_predicted = sum(float(r['predicted_rating']) for r in valid_results) / len(valid_results)
        overall_avg_true = sum(float(r['true_rating']) for r in valid_results) / len(valid_results)
        print(f"\n📊 Overall Statistics:")
        print(f"    Total Interactions Processed: {len(valid_results)}")
        print(f"    Overall Average Predicted Rating: {overall_avg_predicted:.2f}")
        print(f"    Overall Average True Rating: {overall_avg_true:.2f}")
        print(f"    Overall Prediction Error (MAE): {abs(overall_avg_predicted - overall_avg_true):.2f}")


def run_single_interaction(input_json: dict) -> dict:
    """
    Legacy function to process a single interaction (for backward compatibility).
    """
    # Process through ReAct
    react_result = process_single_interaction_react(input_json)
    
    # Process through Reflect
    reflect_results = process_batch_reflect([react_result])
    
    return reflect_results[0]


# # Example usage
# if __name__ == "__main__":
#     # Example of how to use the batch processing
#     input_file = "input_interactions.jsonl"
#     output_file = "output_results.jsonl"
#     batch_size = 5
    
#     run_batch_agent(input_file, output_file, batch_size, debug="both")