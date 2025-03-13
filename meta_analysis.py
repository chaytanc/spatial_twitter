import os
import glob
import shutil
from matplotlib import pyplot as plt
import pandas as pd
import yaml
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image


def load_config(config_file="params.yaml"):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None


def create_spatial_animations(config, base_dir="analysis/spatial_results", output_dir="animations"):
    """
    Find existing PNG files in the spatial results directories and create animations.
    
    Args:
        base_dir: Base directory containing the spatial results folders (0-20)
        output_dir: Directory to save the animations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file patterns to search for
    file_patterns = {
        "accuracy_comparison": "model_accuracy_comparison.png",
        "comparison_maps": "model_comparison_maps.png",
        "comparison_metrics": "model_comparison_metric.png",
        "variance_maps": "variance_comparison_maps.png",
        "ci_comparison": "ci_coverage_comparison.png"
    }
    
    # Dictionary to collect image paths by type
    animation_files = {pattern: [] for pattern in file_patterns}
    
    # Find all relevant images for each sample (0-20)
    for sample_num in range(config.get("N_CLUSTERS")): 
        sample_dir = os.path.join(base_dir, str(sample_num))
        
        # Skip if directory doesn't exist
        if not os.path.exists(sample_dir):
            print(f"Warning: Sample directory {sample_num} not found. Skipping.")
            continue
        
        # Find each image type in this sample directory
        for img_type, file_pattern in file_patterns.items():
            img_path = os.path.join(sample_dir, file_pattern)
            if os.path.exists(img_path):
                animation_files[img_type].append((sample_num, img_path))
                print(f"Found {img_type} for sample {sample_num}: {img_path}")
            else:
                print(f"Missing {img_type} for sample {sample_num}")
    
    # Create animations
    for img_type, files in animation_files.items():
        if not files:
            print(f"No images found for {img_type} animation")
            continue
            
        # Sort by sample number
        files.sort(key=lambda x: x[0])
        image_paths = [path for _, path in files]
        
        # Create output paths
        gif_path = os.path.join(output_dir, f"{img_type}_animation.gif")
        mp4_path = os.path.join(output_dir, f"{img_type}_animation.mp4")
        
        # Load images
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                # Add sample number as text overlay
                sample_num = int(Path(img_path).parts[-2])  # Extract sample number from path
                
                # Convert to numpy array for processing
                img_array = np.array(img)
                images.append(img_array)
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
        
        if not images:
            print(f"No valid images loaded for {img_type}")
            continue
            
        print(f"Creating {img_type} animation with {len(images)} frames")
        
        # Create GIF
        try:
            imageio.mimsave(gif_path, images, duration=1, loop=0)
            print(f"✓ Created GIF: {gif_path}")
        except Exception as e:
            print(f"Error creating GIF: {e}")
        
        # Create MP4 if possible
        try:
            imageio.mimsave(mp4_path, images, fps=2)
            print(f"✓ Created MP4: {mp4_path}")
        except Exception as e:
            print(f"Could not create MP4 (requires imageio-ffmpeg): {e}")
            
    print(f"\nAll animations created in {output_dir}")
    return animation_files


def get_metrics_df(config):
    output_dir = Path(config.get("OUTPUT_DIR", "analysis/results"))
    all_metrics = []
    
    # Load metrics from each sample
    for i in range(config.get("N_CLUSTERS")):
        sample_output_dir = output_dir / str(i)
        metrics_file = sample_output_dir / "model_comparison_metrics.yaml"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = yaml.safe_load(f)
                metrics['sample'] = i
                all_metrics.append(metrics)
    
    if not all_metrics:
        print("No metrics found. Run resampling first.")
        return
    
    # Convert to DataFrame for easier analysis
    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df


def compare_resamples(config, metrics_df):
    """Compare metrics across multiple resamples
      - number of cases where bym is better
      - average coverage % of each model across samples
    """
    output_dir = Path(config.get("OUTPUT_DIR", "analysis/results"))
    
    # Create comparison plots
    metrics_to_plot = ['bym_coverage_probability', 'iid_coverage_probability', 
                       'bym_mean_bias', 'iid_mean_bias',
                       'bym_mean_absolute_error', 'iid_mean_absolute_error',
                       'bym_mean_variance', 'iid_mean_variance']
    
    fig, axs = plt.subplots(4, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axs[i]
        ax.plot(metrics_df['sample'], metrics_df[metric], 'o-')
        ax.set_xlabel('Sample Number')
        ax.set_ylabel(metric)
        ax.set_title(f'Variation in {metric} Across Samples')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_comparison.png")
    return metrics_to_plot


def compute_combined_posterior(df, config):
    """
    Aggregates posterior samples across all resamples and computes summary statistics.
    Generates visualizations for both raw distributions and summary statistics.
    
    Args:
        df: DataFrame containing metrics across resamples
        config: Configuration dictionary with output directory
    
    Returns:
        summary_df: DataFrame with summary statistics for each metric
        metrics_to_plot: List of metrics that were analyzed
    """
    output_dir = Path(config.get("OUTPUT_DIR", "analysis/results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_to_plot = ['bym_coverage_probability', 'iid_coverage_probability', 
                    'bym_mean_bias', 'iid_mean_bias',
                    'bym_mean_absolute_error', 'iid_mean_absolute_error',
                    'bym_mean_variance', 'iid_mean_variance']
    
    # Filter metrics that exist in the dataframe
    metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]
    
    # Compute summary statistics
    summary_stats = {}
    for metric in metrics_to_plot:
        # Check that the metric has valid data
        if df[metric].notnull().sum() > 0:
            # Compute statistics
            mean_value = np.mean(df[metric])
            median_value = np.median(df[metric])
            ci_lower, ci_upper = np.percentile(df[metric], [2.5, 97.5])
            
            summary_stats[metric] = {
                "posterior_mean": mean_value,
                "posterior_median": median_value,
                "credible_interval_lower": ci_lower,
                "credible_interval_upper": ci_upper
            }
    
    # Create DataFrame from dictionary of dictionaries
    summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')
    
    # Plot 1: Raw distributions with histograms - fix histogram plotting
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()
    
    # Plot each metric in its own subplot to avoid issues with different scales
    for i, metric in enumerate(metrics_to_plot):
        if i < len(axs):
            ax = axs[i]
            # Convert to numpy array to ensure it's numeric and make a copy
            values = np.array(df[metric].dropna().values)
            
            if len(values) > 0:
                # Try to create a histogram with fewer bins if there are values
                try:
                    ax.hist(values, bins=min(10, len(values)), alpha=0.7, 
                          density=True, color='skyblue', edgecolor='black')
                    ax.axvline(np.mean(values), color='red', linestyle='--', 
                              linewidth=2, label='Mean')
                    ax.set_title(f"{metric.replace('_', ' ').title()}")
                    ax.set_xlabel("Value")
                    ax.set_ylabel("Density")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
    
    # If there are unused subplots, hide them
    for i in range(len(metrics_to_plot), len(axs)):
        axs[i].set_visible(False)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig(output_dir / "raw_posterior_distributions.png", bbox_inches='tight')
    plt.close(fig)
    
    # Create x positions for bars
    metrics_with_prefix = {
        'coverage_probability': ['bym_coverage_probability', 'iid_coverage_probability'],
        'mean_bias': ['bym_mean_bias', 'iid_mean_bias'],
        'mean_absolute_error': ['bym_mean_absolute_error', 'iid_mean_absolute_error'],
        'mean_variance': ['bym_mean_variance', 'iid_mean_variance']
    }
    
    # Plot 2: Summary statistics visualization with error bars
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # Increased height to avoid tight layout issues
    axs = axs.flatten()
    
    # Plot each metric type in a separate subplot
    for i, (metric_type, metrics) in enumerate(metrics_with_prefix.items()):
        ax = axs[i]
        valid_metrics = [m for m in metrics if m in summary_df.index]
        
        if not valid_metrics:
            ax.text(0.5, 0.5, f"No metrics found for {metric_type}", 
                   ha='center', va='center')
            continue
            
        x_pos = np.arange(len(valid_metrics))
        means = [summary_df.loc[m, 'posterior_mean'] for m in valid_metrics]
        errors_lower = [summary_df.loc[m, 'posterior_mean'] - summary_df.loc[m, 'credible_interval_lower'] 
                        for m in valid_metrics]
        errors_upper = [summary_df.loc[m, 'credible_interval_upper'] - summary_df.loc[m, 'posterior_mean'] 
                        for m in valid_metrics]
        
        # Create bar chart with error bars
        bars = ax.bar(x_pos, means, yerr=[errors_lower, errors_upper], 
                     capsize=10, alpha=0.7, 
                     color=['blue', 'orange'])
        
        # Add labels and formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.split('_')[0].upper() for m in valid_metrics])
        ax.set_title(f"{metric_type.replace('_', ' ').title()}")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{means[j]:.3f}', ha='center', va='bottom')
    
    # Use subplots_adjust instead of tight_layout to have more control
    plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.9, bottom=0.1, left=0.1, right=0.9)
    plt.suptitle("Summary Statistics with 95% Credible Intervals", fontsize=16)
    plt.savefig(output_dir / "summary_statistics.png", bbox_inches='tight')
    plt.close(fig)
    
    # Plot 3: BYM vs IID comparison
    # Get metrics without prefix
    metric_bases = []
    for m in metrics_to_plot:
        if m.startswith('bym_') or m.startswith('iid_'):
            base = m.replace('bym_', '').replace('iid_', '')
            if base not in metric_bases:
                metric_bases.append(base)
    
    if metric_bases:
        fig, ax = plt.subplots(figsize=(12, 8))  # Increased height
        
        x = np.arange(len(metric_bases))
        width = 0.35
        
        bym_means = []
        iid_means = []
        bym_error_lower = []
        bym_error_upper = []
        iid_error_lower = []
        iid_error_upper = []
        
        for base in metric_bases:
            bym_key = f'bym_{base}'
            iid_key = f'iid_{base}'
            
            if bym_key in summary_df.index:
                bym_means.append(summary_df.loc[bym_key, 'posterior_mean'])
                bym_error_lower.append(summary_df.loc[bym_key, 'posterior_mean'] - 
                                      summary_df.loc[bym_key, 'credible_interval_lower'])
                bym_error_upper.append(summary_df.loc[bym_key, 'credible_interval_upper'] - 
                                      summary_df.loc[bym_key, 'posterior_mean'])
            else:
                bym_means.append(0)
                bym_error_lower.append(0)
                bym_error_upper.append(0)
                
            if iid_key in summary_df.index:
                iid_means.append(summary_df.loc[iid_key, 'posterior_mean'])
                iid_error_lower.append(summary_df.loc[iid_key, 'posterior_mean'] - 
                                      summary_df.loc[iid_key, 'credible_interval_lower'])
                iid_error_upper.append(summary_df.loc[iid_key, 'credible_interval_upper'] - 
                                      summary_df.loc[iid_key, 'posterior_mean'])
            else:
                iid_means.append(0)
                iid_error_lower.append(0)
                iid_error_upper.append(0)
        
        # Create grouped bar chart
        rects1 = ax.bar(x - width/2, bym_means, width, label='BYM', 
                       yerr=[bym_error_lower, bym_error_upper], capsize=5, color='blue', alpha=0.7)
        rects2 = ax.bar(x + width/2, iid_means, width, label='IID', 
                       yerr=[iid_error_lower, iid_error_upper], capsize=5, color='orange', alpha=0.7)
        
        # Add labels and formatting
        ax.set_ylabel('Value')
        ax.set_title('BYM vs IID Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([base.replace('_', ' ').title() for base in metric_bases])
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for rect in rects1:
            height = rect.get_height()
            if height != 0:  # Only add label if there's a value
                ax.annotate(f'{height:.3f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', rotation=45)
        
        for rect in rects2:
            height = rect.get_height()
            if height != 0:  # Only add label if there's a value
                ax.annotate(f'{height:.3f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', rotation=45)
        
        # Use explicit spacing instead of tight_layout
        plt.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95)
        plt.savefig(output_dir / "bym_vs_iid_comparison.png")
        plt.close(fig)
    
    # Save summary statistics to CSV
    summary_df.to_csv(output_dir / "summary_statistics.csv")
    
    return summary_df, metrics_to_plot


if __name__ == "__main__":
    config = load_config()
    # results = create_spatial_animations(config)
    metrics_df = get_metrics_df(config)
    # compare_resamples(config, metrics_df)
    combined_posterior_summary = compute_combined_posterior(metrics_df, config)


