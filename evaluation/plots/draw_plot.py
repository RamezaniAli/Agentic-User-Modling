import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
import os


def process_and_save_batch_averages(methods_dict, output_dir):
    """
    Process JSON files to create batch averages and save them to a new directory.
    
    Parameters:
    methods_dict (dict): Dictionary with method names as keys, each containing list of 3 JSON file paths
    output_dir (str): Directory to save the batch-averaged JSON files
    
    The function processes the data by:
    1. Transforming emotion, sentiment, relevance by subtracting from 5
    2. Averaging values in batches of 5
    3. Saving batch-averaged data to JSON files with same structure as input
    """
    
    # Validate input
    if not isinstance(methods_dict, dict):
        raise ValueError("methods_dict must be a dictionary")
    
    if len(methods_dict) != 4:
        raise ValueError("methods_dict must contain exactly 4 methods")
    
    # Check that each method has 3 files
    for method_name, json_files in methods_dict.items():
        if len(json_files) != 3:
            raise ValueError(f"Method '{method_name}' must have exactly 3 JSON files")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Keys that need transformation (subtract from 5)
    transform_keys = ['emotion', 'sentiment', 'relevance']
    
    # Process each method
    for method_name, json_files in methods_dict.items():
        # Process each person (file)
        for person_idx, json_file in enumerate(json_files):
            # Skip if file is None
            if json_file is None:
                continue
            
            try:
                # Load JSON data
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Process all keys to create batch-averaged data
                batch_averaged_data = {}
                
                # Process each key in the original data
                for key in ['MAE_Rating', 'emotion', 'sentiment', 'relevance']:
                    if key not in data:
                        print(f"Warning: Key '{key}' not found in {json_file}, skipping this key...")
                        continue
                    
                    values = data[key]
                    
                    # Check if all values are None - if so, preserve as None
                    if all(v is None for v in values):
                        # Calculate number of batches and fill with None
                        num_batches = (len(values) + 4) // 5  # Ceiling division
                        batch_averaged_data[key] = [None] * num_batches
                        continue
                    
                    # Transform if needed
                    if key in transform_keys:
                        transformed_values = [5 - v if v is not None else None for v in values]
                    else:
                        transformed_values = values
                    
                    # Average in batches of 5, ignoring None values
                    batch_averages = []
                    for j in range(0, len(transformed_values), 5):
                        batch = transformed_values[j:j+5]
                        # Filter out None values from the batch
                        valid_batch = [v for v in batch if v is not None]
                        if valid_batch:
                            batch_averages.append(np.mean(valid_batch))
                        else:
                            batch_averages.append(None)  # Changed from 0 to None
                    
                    batch_averaged_data[key] = batch_averages
                
                # Save batch-averaged data to JSON file
                if batch_averaged_data:
                    # Create the output filename with the same name as input file
                    input_filename = Path(json_file).name
                    output_filepath = os.path.join(output_dir, input_filename)
                    
                    try:
                        with open(output_filepath, 'w') as f:
                            json.dump(batch_averaged_data, f, indent=2)
                        # print(f"Saved batch-averaged data to: {output_filepath}")
                    except Exception as e:
                        print(f"Error saving batch-averaged data to {output_filepath}: {str(e)}")
                
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}, skipping...")
                continue

def plot_mae_analysis_from_averages(methods_dict, title, key_name=None):
    """
    Plot MAE analysis for multiple people and methods from batch-averaged JSON files.
    Includes both batch averages and cumulative averages for trend analysis.
    
    Parameters:
    methods_dict (dict): Dictionary with method names as keys, each containing list of 3 JSON file paths
                        (these should be the batch-averaged files)
    title (str): General title for the entire plot
    key_name (str, optional): Specific key to plot. If None, plots average of all keys.
    
    The function assumes the JSON files contain batch-averaged data with the same structure
    as the original files (4 keys with lists of batch averages).
    """
    
    # Validate input
    if not isinstance(methods_dict, dict):
        raise ValueError("methods_dict must be a dictionary")
    
    if len(methods_dict) != 4:
        raise ValueError("methods_dict must contain exactly 4 methods")
    
    # Check that each method has 3 files
    for method_name, json_files in methods_dict.items():
        if len(json_files) != 3:
            raise ValueError(f"Method '{method_name}' must have exactly 3 JSON files")
    
    # Create subplot figure - 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Colors for different methods
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    method_names = list(methods_dict.keys())
    
    # Process each person (subplot)
    for person_idx in range(3):
        subplot_title = f'Person {person_idx + 1}'
        
        # Store all method data for this person
        all_method_data = {}
        
        # Process each method
        for method_idx, method_name in enumerate(method_names):
            json_file = methods_dict[method_name][person_idx]
            
            # Skip if file is None
            if json_file is None:
                continue
            
            try:
                # Load JSON data (already batch-averaged)
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Update subplot title with actual filename if available
                subplot_title = Path(json_file).stem
                
                # Determine what to plot based on key_name parameter
                if key_name:
                    # Plot specific key only
                    if key_name not in data:
                        print(f"Warning: Key '{key_name}' not found in {json_file}, skipping...")
                        continue
                    
                    y_values = data[key_name]
                    
                else:
                    # Plot average of all 4 keys
                    available_keys = [key for key in ['MAE_Rating', 'emotion', 'sentiment', 'relevance'] if key in data]
                    
                    if not available_keys:
                        print(f"Warning: No valid keys found in {json_file}, skipping...")
                        continue
                    
                    # Find the maximum length to ensure we process all data points
                    max_length = max(len(data[key]) for key in available_keys)
                    
                    # Create averaged values by considering only non-None values at each position
                    averaged_values = []
                    for i in range(max_length):
                        position_values = []
                        for key in available_keys:
                            if i < len(data[key]) and data[key][i] is not None:
                                position_values.append(data[key][i])
                        
                        # Only add average if we have at least one non-None value
                        if position_values:
                            averaged_values.append(np.mean(position_values))
                        else:
                            averaged_values.append(None)
                    
                    y_values = averaged_values
                
                # Store data for this method
                all_method_data[method_name] = y_values
                
                # Create x-axis labels
                x_labels = [f'Batch {j+1}' for j in range(len(y_values))]
                x_positions = range(len(y_values))
                
                # Plot batch averages (top row)
                axes[0, person_idx].plot(x_positions, y_values, 
                                        marker='o', 
                                        linewidth=2, 
                                        markersize=6,
                                        color=colors[method_idx],
                                        label=method_name)
                
                # Calculate and plot moving averages (bottom row)
                # Use a window size that adapts to the data length but focuses on recent points
                window_size = min(max(3, len(y_values) // 3), 5)  # Between 3-5 points, adaptive
                
                moving_averages = []
                for i in range(len(y_values)):
                    # Calculate moving average using a window that focuses on recent data
                    start_idx = max(0, i - window_size + 1)
                    window_values = [v for v in y_values[start_idx:i+1] if v is not None]
                    
                    if window_values:
                        moving_avg = np.mean(window_values)
                        moving_averages.append(moving_avg)
                    else:
                        # If no valid values in window, use the last valid moving average or 0
                        if moving_averages:
                            moving_averages.append(moving_averages[-1])
                        else:
                            moving_averages.append(0)
                
                axes[1, person_idx].plot(x_positions, moving_averages,
                                        marker='s',  # square markers for distinction
                                        linewidth=2,
                                        markersize=6,
                                        color=colors[method_idx],
                                        label=method_name)
                
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}, skipping...")
                continue
        
        # Set subplot properties for batch averages (top row)
        axes[0, person_idx].set_title(f'{subplot_title} - Batch Averages', fontweight='bold')
        axes[0, person_idx].set_xlabel('Batches')
        axes[0, person_idx].set_ylabel('MAE Values')
        axes[0, person_idx].grid(True, alpha=0.3)
        axes[0, person_idx].legend()
        axes[0, person_idx].spines['top'].set_visible(False)
        axes[0, person_idx].spines['right'].set_visible(False)
        
        # Set subplot properties for moving averages (bottom row)
        axes[1, person_idx].set_title(f'{subplot_title} - Moving Average Trend', fontweight='bold')
        axes[1, person_idx].set_xlabel('Batches')
        axes[1, person_idx].set_ylabel('Moving Average MAE Values')
        axes[1, person_idx].grid(True, alpha=0.3)
        axes[1, person_idx].legend()
        axes[1, person_idx].spines['top'].set_visible(False)
        axes[1, person_idx].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def display_mae_summary(json_files, title):
    """
    Display MAE summary for multiple people from JSON files.
    
    Parameters:
    json_files (list): List of 3 JSON file paths
    title (str): Title to display above the summary
    
    The function processes the data by:
    1. Transforming emotion, sentiment, relevance by subtracting from 5
    2. Computing average of each key separately for each person
    3. Computing overall average of these averages
    """
    
    if len(json_files) != 3:
        raise ValueError("Exactly 3 JSON files are required")
    
    # Keys that need transformation (subtract from 5)
    transform_keys = ['emotion', 'sentiment', 'relevance']
    all_keys = ['MAE_Rating', 'emotion', 'sentiment', 'relevance']
    
    # Store results for each person
    person_results = []
    
    for json_file in json_files:
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Get person name from filename
        person_name = Path(json_file).stem
        
        # Calculate average for each key separately
        key_averages = {}
        
        for key in all_keys:
            if key not in data:
                raise ValueError(f"Key '{key}' not found in {json_file}")
            
            values = data[key]
            
            # Transform if needed
            if key in transform_keys:
                values = [5 - v for v in values]
            
            # Calculate average for this key
            key_averages[key] = np.mean(values)
        
        person_results.append((person_name, key_averages))
    
    # Print results in a clean format
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    print()
    
    # Display results for each person
    for person_name, key_averages in person_results:
        print(f"📊 {person_name}:")
        print(f"   • MAE_Rating: {key_averages['MAE_Rating']:.4f}")
        print(f"   • MAE_Emotion:    {key_averages['emotion']:.4f}")
        print(f"   • MAE_Sentiment:  {key_averages['sentiment']:.4f}")
        print(f"   • MAE_Relevance:  {key_averages['relevance']:.4f}")
        print()
    
    # Calculate overall averages across all people
    print("=" * 80)
    print("📈 Overall MAE Averages:")
    print("=" * 80)
    
    for key in all_keys:
        key_sum = sum(person_data[1][key] for person_data in person_results)
        overall_avg = key_sum / len(person_results)
        print(f"   • {key:<12}: {overall_avg:.4f}")
    
    print()
    print("=" * 80)

# Example usage:
# display_mae_summary(['u_01.json', 'u_02.json', 'u_03.json'], 'MAE Analysis Summary Report')