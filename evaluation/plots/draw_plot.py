import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_mae_analysis(methods_dict, title, key_name=None):
    """
    Plot MAE analysis for multiple people and methods from JSON files.
    Now includes both batch averages and cumulative averages for trend analysis.
    
    Parameters:
    methods_dict (dict): Dictionary with method names as keys, each containing list of 3 JSON file paths
    title (str): General title for the entire plot
    key_name (str, optional): Specific key to plot. If None, plots average of all keys.
    
    The function processes the data by:
    1. Transforming emotion, sentiment, relevance by subtracting from 5
    2. Averaging values in batches of 5
    3. Either plotting specific key or average of all 4 keys
    4. Handling None data gracefully
    5. Creating cumulative averages for trend analysis
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
    
    # Create subplot figure - now 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Keys that need transformation (subtract from 5)
    transform_keys = ['emotion', 'sentiment', 'relevance']
    
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
                # Load JSON data
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Update subplot title with actual filename if available
                subplot_title = Path(json_file).stem
                
                if key_name:
                    # Plot specific key only
                    if key_name not in data:
                        print(f"Warning: Key '{key_name}' not found in {json_file}, skipping...")
                        continue
                    
                    values = data[key_name]
                    
                    # Transform if needed
                    if key_name in transform_keys:
                        values = [5 - v if v is not None else None for v in values]
                    
                    # Average in batches of 5, ignoring None values
                    batch_averages = []
                    for j in range(0, len(values), 5):
                        batch = values[j:j+5]
                        # Filter out None values from the batch
                        valid_batch = [v for v in batch if v is not None]
                        if valid_batch:
                            batch_averages.append(np.mean(valid_batch))
                        else:
                            batch_averages.append(0)  # or np.nan if you prefer
                    
                    y_values = batch_averages
                    
                else:
                    # Plot average of all 4 keys
                    all_transformed_values = {}
                    
                    # Process each key and collect all values
                    for key in ['MAE_Rating', 'emotion', 'sentiment', 'relevance']:
                        if key not in data:
                            print(f"Warning: Key '{key}' not found in {json_file}, skipping this key...")
                            continue
                        
                        values = data[key]
                        
                        # Transform if needed
                        if key in transform_keys:
                            values = [5 - v if v is not None else None for v in values]
                        
                        all_transformed_values[key] = values
                    
                    # Skip if no valid keys found
                    if not all_transformed_values:
                        print(f"Warning: No valid keys found in {json_file}, skipping...")
                        continue
                    
                    # Find the maximum length to ensure we process all data points
                    max_length = max(len(values) for values in all_transformed_values.values())
                    
                    # Create averaged values by considering only non-None values at each position
                    averaged_values = []
                    for i in range(max_length):
                        position_values = []
                        for key, values in all_transformed_values.items():
                            if i < len(values) and values[i] is not None:
                                position_values.append(values[i])
                        
                        # Only add average if we have at least one non-None value
                        if position_values:
                            averaged_values.append(np.mean(position_values))
                        else:
                            averaged_values.append(None)
                    
                    # Average in batches of 5, ignoring None values
                    batch_averages = []
                    for j in range(0, len(averaged_values), 5):
                        batch = averaged_values[j:j+5]
                        # Filter out None values from the batch
                        valid_batch = [v for v in batch if v is not None]
                        if valid_batch:
                            batch_averages.append(np.mean(valid_batch))
                        else:
                            batch_averages.append(0)  # or np.nan if you prefer
                    
                    y_values = batch_averages
                
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
                
                # Calculate and plot cumulative averages (bottom row)
                cumulative_averages = []
                for i in range(len(y_values)):
                    # Calculate cumulative average up to current point
                    cumulative_avg = np.mean(y_values[:i+1])
                    cumulative_averages.append(cumulative_avg)
                
                axes[1, person_idx].plot(x_positions, cumulative_averages,
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
        
        # Set subplot properties for cumulative averages (bottom row)
        axes[1, person_idx].set_title(f'{subplot_title} - Cumulative Averages (Trend)', fontweight='bold')
        axes[1, person_idx].set_xlabel('Batches')
        axes[1, person_idx].set_ylabel('Cumulative MAE Values')
        axes[1, person_idx].grid(True, alpha=0.3)
        axes[1, person_idx].legend()
        axes[1, person_idx].spines['top'].set_visible(False)
        axes[1, person_idx].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def plot_mae_analysis2(methods_dict, title, key_name=None, trend_method='moving_average', window_size=3):
    """
    Plot MAE analysis for multiple people and methods from JSON files.
    Now includes multiple trend analysis methods with confidence intervals.
    
    Parameters:
    methods_dict (dict): Dictionary with method names as keys, each containing list of 3 JSON file paths
    title (str): General title for the entire plot
    key_name (str, optional): Specific key to plot. If None, plots average of all keys.
    trend_method (str): Method for trend analysis:
        - 'moving_average': Rolling/moving average with confidence intervals
        - 'linear_regression': Linear trend line with confidence intervals
        - 'lowess': LOWESS smoothing (locally weighted regression)
        - 'exponential': Exponential moving average
    window_size (int): Window size for moving average (default: 3)
    
    The function processes the data by:
    1. Transforming emotion, sentiment, relevance by subtracting from 5
    2. Averaging values in batches of 5
    3. Either plotting specific key or average of all 4 keys
    4. Handling None data gracefully
    5. Creating trend analysis with confidence intervals
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
    
    # Create subplot figure - now 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{title} - Trend Method: {trend_method.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    
    # Keys that need transformation (subtract from 5)
    transform_keys = ['emotion', 'sentiment', 'relevance']
    
    # Colors for different methods
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    method_names = list(methods_dict.keys())
    
    def calculate_moving_average_with_ci(data, window_size=3):
        """Calculate moving average with confidence intervals"""
        if len(data) < window_size:
            return data, [0] * len(data), [0] * len(data)
        
        moving_avg = []
        lower_ci = []
        upper_ci = []
        
        for i in range(len(data)):
            if i < window_size - 1:
                # For early points, use all available data
                window_data = data[:i+1]
            else:
                # Use sliding window
                window_data = data[i-window_size+1:i+1]
            
            mean_val = np.mean(window_data)
            std_val = np.std(window_data, ddof=1) if len(window_data) > 1 else 0
            se = std_val / np.sqrt(len(window_data))
            
            # 95% confidence interval
            ci = 1.96 * se
            
            moving_avg.append(mean_val)
            lower_ci.append(mean_val - ci)
            upper_ci.append(mean_val + ci)
        
        return moving_avg, lower_ci, upper_ci
    
    def calculate_linear_trend_with_ci(data):
        """Calculate linear regression trend with confidence intervals"""
        if len(data) < 2:
            return data, [0] * len(data), [0] * len(data)
        
        x = np.arange(len(data)).reshape(-1, 1)
        y = np.array(data)
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(x, y)
        
        # Predict
        y_pred = model.predict(x)
        
        # Calculate residuals and standard error
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        se = np.sqrt(mse)
        
        # 95% confidence interval
        ci = 1.96 * se
        
        lower_ci = y_pred - ci
        upper_ci = y_pred + ci
        
        return y_pred.tolist(), lower_ci.tolist(), upper_ci.tolist()
    
    def calculate_exponential_moving_average(data, alpha=0.3):
        """Calculate exponential moving average"""
        if len(data) == 0:
            return [], [], []
        
        ema = [data[0]]
        for i in range(1, len(data)):
            ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
        
        # Simple confidence interval based on recent variance
        window_size = min(5, len(data))
        recent_std = np.std(data[-window_size:]) if len(data) > 1 else 0
        ci = 1.96 * recent_std / np.sqrt(window_size)
        
        lower_ci = [val - ci for val in ema]
        upper_ci = [val + ci for val in ema]
        
        return ema, lower_ci, upper_ci
    
    def calculate_lowess_smooth(data, frac=0.3):
        """Calculate LOWESS smoothing (simplified version)"""
        if len(data) < 3:
            return data, [0] * len(data), [0] * len(data)
        
        # Simple local regression approximation
        x = np.arange(len(data))
        smoothed = []
        
        for i in range(len(data)):
            # Define local window
            window_size = max(3, int(frac * len(data)))
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            
            # Local linear regression
            x_local = x[start:end]
            y_local = np.array(data[start:end])
            
            if len(x_local) > 1:
                # Fit local linear regression
                A = np.vstack([x_local, np.ones(len(x_local))]).T
                coeffs = np.linalg.lstsq(A, y_local, rcond=None)[0]
                smoothed.append(coeffs[0] * x[i] + coeffs[1])
            else:
                smoothed.append(data[i])
        
        # Confidence interval based on local variance
        local_std = np.std(data) if len(data) > 1 else 0
        ci = 1.96 * local_std / np.sqrt(len(data))
        
        lower_ci = [val - ci for val in smoothed]
        upper_ci = [val + ci for val in smoothed]
        
        return smoothed, lower_ci, upper_ci
    
    # Process each person (subplot)
    for person_idx in range(3):
        subplot_title = f'Person {person_idx + 1}'
        
        # Process each method
        for method_idx, method_name in enumerate(method_names):
            json_file = methods_dict[method_name][person_idx]
            
            # Skip if file is None
            if json_file is None:
                continue
            
            try:
                # Load JSON data
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Update subplot title with actual filename if available
                subplot_title = Path(json_file).stem
                
                if key_name:
                    # Plot specific key only
                    if key_name not in data:
                        print(f"Warning: Key '{key_name}' not found in {json_file}, skipping...")
                        continue
                    
                    values = data[key_name]
                    
                    # Transform if needed
                    if key_name in transform_keys:
                        values = [5 - v if v is not None else None for v in values]
                    
                    # Average in batches of 5, ignoring None values
                    batch_averages = []
                    for j in range(0, len(values), 5):
                        batch = values[j:j+5]
                        # Filter out None values from the batch
                        valid_batch = [v for v in batch if v is not None]
                        if valid_batch:
                            batch_averages.append(np.mean(valid_batch))
                        else:
                            batch_averages.append(0)  # or np.nan if you prefer
                    
                    y_values = batch_averages
                    
                else:
                    # Plot average of all 4 keys
                    all_transformed_values = {}
                    
                    # Process each key and collect all values
                    for key in ['MAE_Rating', 'emotion', 'sentiment', 'relevance']:
                        if key not in data:
                            print(f"Warning: Key '{key}' not found in {json_file}, skipping this key...")
                            continue
                        
                        values = data[key]
                        
                        # Transform if needed
                        if key in transform_keys:
                            values = [5 - v if v is not None else None for v in values]
                        
                        all_transformed_values[key] = values
                    
                    # Skip if no valid keys found
                    if not all_transformed_values:
                        print(f"Warning: No valid keys found in {json_file}, skipping...")
                        continue
                    
                    # Find the maximum length to ensure we process all data points
                    max_length = max(len(values) for values in all_transformed_values.values())
                    
                    # Create averaged values by considering only non-None values at each position
                    averaged_values = []
                    for i in range(max_length):
                        position_values = []
                        for key, values in all_transformed_values.items():
                            if i < len(values) and values[i] is not None:
                                position_values.append(values[i])
                        
                        # Only add average if we have at least one non-None value
                        if position_values:
                            averaged_values.append(np.mean(position_values))
                        else:
                            averaged_values.append(None)
                    
                    # Average in batches of 5, ignoring None values
                    batch_averages = []
                    for j in range(0, len(averaged_values), 5):
                        batch = averaged_values[j:j+5]
                        # Filter out None values from the batch
                        valid_batch = [v for v in batch if v is not None]
                        if valid_batch:
                            batch_averages.append(np.mean(valid_batch))
                        else:
                            batch_averages.append(0)  # or np.nan if you prefer
                    
                    y_values = batch_averages
                
                # Skip if no valid data
                if not y_values or len(y_values) < 2:
                    continue
                
                # Create x-axis labels
                x_positions = range(len(y_values))
                
                # Plot original batch averages (top row)
                axes[0, person_idx].plot(x_positions, y_values, 
                                        marker='o', 
                                        linewidth=2, 
                                        markersize=6,
                                        color=colors[method_idx],
                                        label=method_name,
                                        alpha=0.7)
                
                # Calculate trend based on selected method
                if trend_method == 'moving_average':
                    trend_values, lower_ci, upper_ci = calculate_moving_average_with_ci(y_values, window_size)
                elif trend_method == 'linear_regression':
                    trend_values, lower_ci, upper_ci = calculate_linear_trend_with_ci(y_values)
                elif trend_method == 'exponential':
                    trend_values, lower_ci, upper_ci = calculate_exponential_moving_average(y_values)
                elif trend_method == 'lowess':
                    trend_values, lower_ci, upper_ci = calculate_lowess_smooth(y_values)
                else:
                    trend_values, lower_ci, upper_ci = calculate_moving_average_with_ci(y_values, window_size)
                
                # Plot trend with confidence intervals (bottom row)
                axes[1, person_idx].plot(x_positions, trend_values,
                                        linewidth=3,
                                        color=colors[method_idx],
                                        label=method_name)
                
                # Add confidence interval
                axes[1, person_idx].fill_between(x_positions, lower_ci, upper_ci,
                                                color=colors[method_idx],
                                                alpha=0.2)
                
                # Add original data points for reference
                axes[1, person_idx].scatter(x_positions, y_values,
                                          color=colors[method_idx],
                                          alpha=0.5,
                                          s=30)
                
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}, skipping...")
                continue
        
        # Set subplot properties for batch averages (top row)
        axes[0, person_idx].set_title(f'{subplot_title} - Original Batch Averages', fontweight='bold')
        axes[0, person_idx].set_xlabel('Batches')
        axes[0, person_idx].set_ylabel('MAE Values')
        axes[0, person_idx].grid(True, alpha=0.3)
        axes[0, person_idx].legend()
        axes[0, person_idx].spines['top'].set_visible(False)
        axes[0, person_idx].spines['right'].set_visible(False)
        
        # Set subplot properties for trend analysis (bottom row)
        axes[1, person_idx].set_title(f'{subplot_title} - Trend Analysis with 95% CI', fontweight='bold')
        axes[1, person_idx].set_xlabel('Batches')
        axes[1, person_idx].set_ylabel('Trend Values')
        axes[1, person_idx].grid(True, alpha=0.3)
        axes[1, person_idx].legend()
        axes[1, person_idx].spines['top'].set_visible(False)
        axes[1, person_idx].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()


# Example usage with different trend methods:
# plot_mae_analysis(methods_dict, "MAE Analysis", trend_method='moving_average', window_size=3)
# plot_mae_analysis(methods_dict, "MAE Analysis", trend_method='linear_regression')
# plot_mae_analysis(methods_dict, "MAE Analysis", trend_method='exponential')
# plot_mae_analysis(methods_dict, "MAE Analysis", trend_method='lowess')

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