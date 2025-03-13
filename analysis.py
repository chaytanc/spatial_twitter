from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from calc_prevalences import calc_prevs
from preprocess import load_pickled_dataset
from sample_data import generate_resamples, load_sample
import yaml
import rpy2.robjects as ro
import os
import rpy2.robjects.pandas2ri as pd2ri
from EmbeddingTextDataset import EmbeddingTextDataset
from pathlib import Path
import pickle

# Set R environment variables
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
os.environ['R_LIBS'] = "/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/library"
os.environ['R_LIBS_USER'] = "/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/library"

def load_config(config_file="params.yaml"):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None

def run_spatial_analysis(config, sample_number):
    """Run one round of spatial analysis with the current sample"""
    # Create output directories
    output_dir = Path(config.get("OUTPUT_DIR", "analysis/results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_output_dir = output_dir / str(sample_number)
    sample_output_dir.mkdir(exist_ok=True)

    sample_input_dir = Path(config.get("SAMPLE_DIR")) / str(sample_number)
 
    sample_df, pca, kmeans, gdf = load_sample(sample_input_dir)
    
    # TODO recalc true prevs for every different shape file
    # true_prevs = load_pickled_dataset(config["TRUE_PREVS_FILE"])
    true_prevs = calc_prevs(sample_number)

    pd2ri.activate()
    r_sample_df = pd2ri.py2rpy(sample_df)
    # shapefile_path = config.get("SAMPLE_SHAPE_FILE", "~/Desktop/chaytan/spatial_twitter/sample/-1/sample.shp")
    shapefile_path = str(sample_input_dir / config.get("SAMPLE_SHAPE_FILE"))
    
    # Define the R code for INLA analysis
    r_code = f"""
    library(INLA)
    library(spdep) 
    library(sf) 
    library(prioritizr)
    library(dplyr)

    gdf <- st_read("{shapefile_path}")
    data <- as.data.frame(r_sample_df)  # Convert the DataFrame to a data frame in R
    spatial_data <- data.frame(
      Cluster_ID = as.numeric(as.factor(data$Cluster)),  # Assuming the 'Cluster' column is used to identify clusters
      Stance = as.numeric(data$Stance),       # The stance variable we want to model
      X = data$x,                 # The 2D spatial coordinates (x-coordinate)
      Y = data$y                  # The 2D spatial coordinates (y-coordinate)
    )

    polygons <- st_as_sf(gdf)
    # Create adjacency matrix from spatial structure (polygons)
    # Use Queen contiguity (polygons sharing an edge or a vertex are neighbors)
    nb <- poly2nb(gdf)
    adj <- nb2mat(nb, style="B", zero.policy=TRUE)
    mat <- adjacency_matrix(gdf)
    colnames(mat) <- rownames(mat) <- levels(as.factor(spatial_data$Cluster_ID))
    mat <- as.matrix(mat[1:dim(mat)[1], 1:dim(mat)[1]])

    cluster_data <- spatial_data %>%
      group_by(Cluster_ID) %>%
      summarise(
        Successes = sum(Stance), 
        n = n()                   # Total number of tweets in each cluster
      )

    # BYM with spatial dependence
    formula <- Successes ~ f(Cluster_ID, model = "bym", graph = adj)
    result <- inla(formula, family = "binomial", data = cluster_data, 
      control.family = list(link = "logit"), Ntrials = cluster_data$n)
    print(summary(result))
    saveRDS(result, file = "{sample_output_dir}/inla_result.rds")
    """
    
    # Run R-INLA
    ro.globalenv['r_sample_df'] = r_sample_df
    ro.r(r_code)
    
    # Load and process INLA results
    result = ro.r['readRDS'](f"{sample_output_dir}/inla_result.rds")
    summary_fitted_values = result.rx2("summary.fitted.values") 
    
    # Convert to Pandas DataFrame
    df = pd2ri.rpy2py(summary_fitted_values)
    df['Cluster_ID'] = range(0, len(df))
    
    # Merge with geometry and true prevalence data
    gdf = gdf.merge(df, left_on="Cluster_ID", right_on="Cluster_ID", how="left")
    gdf = gdf.merge(true_prevs, on="Cluster_ID")
    
    # Calculate error metrics
    gdf['absolute_error'] = np.abs(gdf['mean'] - gdf['True Prevalence'])
    gdf['squared_error'] = (gdf['mean'] - gdf['True Prevalence'])**2
    gdf['bias'] = gdf['mean'] - gdf['True Prevalence']  # Positive means overestimation
    gdf['within_CI'] = ((gdf['True Prevalence'] >= gdf['0.025quant']) & 
                        (gdf['True Prevalence'] <= gdf['0.975quant']))
    
    # Calculate variance from the INLA results
    if 'sd' in gdf.columns:
        gdf['variance'] = gdf['sd']**2
    else:
        # Estimate variance from quantiles (approximate)
        gdf['variance'] = ((gdf['0.975quant'] - gdf['0.025quant']) / (2 * 1.96))**2
    
    # Save the merged results
    gdf.to_file(f"{sample_output_dir}/results.shp")
    
    # Create visualizations
    
    # Basic probability map
    fig, ax = plt.subplots(figsize=(10, 6))
    gdf.plot(column="0.5quant", cmap="viridis", legend=True, ax=ax)
    ax.set_title(f"INLA Spatial Effects - Sample {sample_number}")
    plt.tight_layout()
    plt.savefig(f"{sample_output_dir}/inla_spatial_effects.png", dpi=300)
    plt.close()
    
    # Comparison maps
    fig, axs = plt.subplots(1, 4, figsize=(18, 6))
    
    gdf.plot(column="mean", cmap="viridis", legend=True, ax=axs[0])
    axs[0].set_title("Mean Probability")
    
    gdf.plot(column="0.025quant", cmap="viridis", legend=True, ax=axs[1])
    axs[1].set_title("Lower 95% CI")
    
    gdf.plot(column="0.975quant", cmap="viridis", legend=True, ax=axs[2])
    axs[2].set_title("Upper 95% CI")
    
    gdf.plot(column="True Prevalence", cmap="viridis", legend=True, ax=axs[3])
    axs[3].set_title("True Prevalences")
    
    plt.tight_layout()
    plt.savefig(f"{sample_output_dir}/inla_spatial_results.png", dpi=300)
    plt.close()
    
    # Error analysis maps
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original estimates
    gdf.plot(column="mean", cmap="viridis", legend=True, ax=axs[0, 0])
    axs[0, 0].set_title("INLA Estimated Probability")
    
    # True values
    gdf.plot(column="True Prevalence", cmap="viridis", legend=True, ax=axs[0, 1])
    axs[0, 1].set_title("True Prevalence")
    
    # Absolute error
    error_max = gdf['absolute_error'].max()
    gdf.plot(column="absolute_error", cmap="Reds", legend=True, 
             vmin=0, vmax=error_max, ax=axs[0, 2])
    axs[0, 2].set_title("Absolute Error")
    
    # Bias (over/under estimation)
    bias_max = max(np.abs(gdf['bias'].min()), gdf['bias'].max())
    gdf.plot(column="bias", cmap="RdBu_r", legend=True, 
             vmin=-bias_max, vmax=bias_max, ax=axs[1, 0])
    axs[1, 0].set_title("Bias (Red = Overestimation, Blue = Underestimation)")
    
    # Variance
    gdf.plot(column="variance", cmap="Purples", legend=True, ax=axs[1, 1])
    axs[1, 1].set_title("Variance (Uncertainty)")
    
    # Within CI indicator
    gdf.plot(column="within_CI", cmap="RdYlGn", legend=True, 
             categorical=True, ax=axs[1, 2])
    axs[1, 2].set_title("True Value Within 95% CI (Green = Yes)")
    
    plt.tight_layout()
    plt.savefig(f"{sample_output_dir}/inla_spatial_validation.png", dpi=300)
    plt.close()
    
    # Detailed error analysis
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot the relationship between estimated and true values
    axs[0].scatter(gdf['True Prevalence'], gdf['mean'])
    axs[0].plot([0, 1], [0, 1], 'k--')  # Add diagonal reference line
    axs[0].set_xlabel('True Prevalence')
    axs[0].set_ylabel('Estimated Value')
    axs[0].set_title('Estimated vs True Values')
    axs[0].grid(True, alpha=0.3)
    
    # Add error bars to show uncertainty
    for i, row in gdf.iterrows():
        axs[0].plot([row['True Prevalence'], row['True Prevalence']], 
                   [row['0.025quant'], row['0.975quant']], 'r-', alpha=0.5)
    
    # Plot error distribution
    axs[1].hist(gdf['bias'], bins=10, alpha=0.7)
    axs[1].axvline(x=0, color='k', linestyle='--')
    axs[1].set_xlabel('Bias (Estimated - True)')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Distribution of Bias')
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{sample_output_dir}/inla_error_analysis.png", dpi=300)
    plt.close()
    
    # Calculate summary statistics
    summary_stats = {
        'mean_absolute_error': float(gdf['absolute_error'].mean()),
        'root_mean_squared_error': float(np.sqrt(gdf['squared_error'].mean())),
        'mean_bias': float(gdf['bias'].mean()),
        'coverage_probability': float(gdf['within_CI'].mean() * 100),
        'max_absolute_error': float(gdf['absolute_error'].max()),
        'min_absolute_error': float(gdf['absolute_error'].min()),
        'mean_variance': float(gdf['variance'].mean())
    }
    
    # Save metrics to YAML
    with open(f"{sample_output_dir}/metrics.yaml", 'w') as f:
        yaml.dump(summary_stats, f)
    
    # Create a summary visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(summary_stats.keys(), summary_stats.values())
    ax.set_xticklabels(summary_stats.keys(), rotation=45, ha='right')
    ax.set_title(f'Error Metrics Summary - Sample {sample_number}')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{sample_output_dir}/inla_summary_metrics.png", dpi=300)
    plt.close()
    
    return summary_stats

def analyze_with_resampling(config):
    # generate_resamples()
    resample_n = config.get("RESAMPLE_N")
    for i in range(resample_n):
        summary_stats = run_spatial_analysis(config, i)

# TODO later
def compare_resamples(config, num_samples):
    """Compare metrics across multiple resamples"""
    output_dir = Path(config.get("OUTPUT_DIR", "analysis/results"))
    all_metrics = []
    
    # Load metrics from each sample
    for i in range(num_samples):
        sample_output_dir = output_dir / str(i)
        metrics_file = sample_output_dir / "metrics.yaml"
        
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
    
    # Create comparison plots
    metrics_to_plot = ['mean_absolute_error', 'root_mean_squared_error', 
                       'coverage_probability', 'mean_variance']
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
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

if __name__ == "__main__":
  config = load_config()
  analyze_with_resampling(config)