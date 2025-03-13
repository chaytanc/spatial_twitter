from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from tqdm import tqdm
from calc_prevalences import calc_prevs
from preprocess import clear_memory, load_pickled_dataset
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

    # TODO add n_clusters to sample number and run with 10 clusters?
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

    # For an IID (non-spatial) model
    formula_iid <- Successes ~ 1 + f(Cluster_ID, model = "iid")
    result_iid <- inla(
      formula_iid, 
      family = "binomial", 
      data = cluster_data,
      Ntrials = cluster_data$n,
      control.predictor = list(compute = TRUE),
      control.compute = list(return.marginals = TRUE)
    )
    saveRDS(result_iid, file = "{sample_output_dir}/iid_inla_result.rds")

    """
    
    # Run R-INLA
    ro.globalenv['r_sample_df'] = r_sample_df
    ro.r(r_code)
    
    # Load and process both INLA results
    result = ro.r['readRDS'](f"{sample_output_dir}/inla_result.rds")
    iid_result = ro.r['readRDS'](f"{sample_output_dir}/iid_inla_result.rds")  # Assuming different file for IID model

    # Process BYM model results
    summary_fitted_values = result.rx2("summary.fitted.values") 
    df = pd2ri.rpy2py(summary_fitted_values)
    df['Cluster_ID'] = range(0, len(df))

    # Process IID model results
    iid_summary_fitted_values = iid_result.rx2("summary.fitted.values")
    iid_df = pd2ri.rpy2py(iid_summary_fitted_values)
    iid_df['Cluster_ID'] = range(0, len(iid_df))
    iid_df = iid_df.add_prefix('iid_')  # Add prefix to avoid column name conflicts
    iid_df = iid_df.rename(columns={'iid_Cluster_ID': 'Cluster_ID'})  # Keep Cluster_ID as is

    # Merge with geometry and true prevalence data
    gdf = gdf.merge(df, left_on="Cluster_ID", right_on="Cluster_ID", how="left")
    gdf = gdf.merge(iid_df, on="Cluster_ID", how="left")
    gdf = gdf.merge(true_prevs, on="Cluster_ID")

    # Calculate error metrics for both models
    # BYM model
    gdf['absolute_error'] = np.abs(gdf['mean'] - gdf['True Prevalence'])
    gdf['squared_error'] = (gdf['mean'] - gdf['True Prevalence'])**2
    gdf['bias'] = gdf['mean'] - gdf['True Prevalence']
    gdf['within_CI'] = ((gdf['True Prevalence'] >= gdf['0.025quant']) & 
                        (gdf['True Prevalence'] <= gdf['0.975quant']))

    # IID model
    gdf['iid_absolute_error'] = np.abs(gdf['iid_mean'] - gdf['True Prevalence'])
    gdf['iid_squared_error'] = (gdf['iid_mean'] - gdf['True Prevalence'])**2
    gdf['iid_bias'] = gdf['iid_mean'] - gdf['True Prevalence']
    gdf['iid_within_CI'] = ((gdf['True Prevalence'] >= gdf['iid_0.025quant']) & 
                            (gdf['True Prevalence'] <= gdf['iid_0.975quant']))

    # Calculate variance for both models
    if 'sd' in gdf.columns and 'iid_sd' in gdf.columns:
        gdf['variance'] = gdf['sd']**2
        gdf['iid_variance'] = gdf['iid_sd']**2
    else:
        # Estimate variance from quantiles (approximate)
        gdf['variance'] = ((gdf['0.975quant'] - gdf['0.025quant']) / (2 * 1.96))**2
        gdf['iid_variance'] = ((gdf['iid_0.975quant'] - gdf['iid_0.025quant']) / (2 * 1.96))**2

    # Save the merged results
    gdf.to_file(f"{sample_output_dir}/results_comparison.shp")

    # Create comparison visualizations
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))  # Expanded to 4 columns

    # BYM vs IID vs True Values (first 3 columns remain the same)
    gdf.plot(column="mean", cmap="viridis", legend=True, ax=axs[0, 0])
    axs[0, 0].set_title("BYM Model: Mean Probability")

    gdf.plot(column="iid_mean", cmap="viridis", legend=True, ax=axs[0, 1])
    axs[0, 1].set_title("IID Model: Mean Probability")

    gdf.plot(column="True Prevalence", cmap="viridis", legend=True, ax=axs[0, 2])
    axs[0, 2].set_title("True Prevalences")

    # Add new column for CI coverage comparison (both models)
    # Create a combined CI coverage comparison
    gdf['ci_comparison'] = 0  # Neither model covers
    gdf.loc[gdf['within_CI'] & ~gdf['iid_within_CI'], 'ci_comparison'] = 1  # Only BYM covers
    gdf.loc[~gdf['within_CI'] & gdf['iid_within_CI'], 'ci_comparison'] = 2  # Only IID covers
    gdf.loc[gdf['within_CI'] & gdf['iid_within_CI'], 'ci_comparison'] = 3  # Both models cover

    # Create a categorical colormap
    ci_cmap = plt.cm.get_cmap('RdYlGn', 4)
    gdf.plot(column="ci_comparison", cmap=ci_cmap, legend=True, ax=axs[0, 3],
            categorical=True, legend_kwds={'labels': ['Neither', 'Only BYM', 'Only IID', 'Both']})
    axs[0, 3].set_title("95% CI Coverage Comparison")

    # Absolute error comparison (bottom row - first 3 columns remain the same)
    error_max = max(gdf['absolute_error'].max(), gdf['iid_absolute_error'].max())

    gdf.plot(column="absolute_error", cmap="Reds", legend=True, 
            vmin=0, vmax=error_max, ax=axs[1, 0])
    axs[1, 0].set_title("BYM Model: Absolute Error")

    gdf.plot(column="iid_absolute_error", cmap="Reds", legend=True, 
            vmin=0, vmax=error_max, ax=axs[1, 1])
    axs[1, 1].set_title("IID Model: Absolute Error")

    # Model difference
    gdf['model_difference'] = gdf['mean'] - gdf['iid_mean']
    diff_max = max(abs(gdf['model_difference'].min()), abs(gdf['model_difference'].max()))
    gdf.plot(column="model_difference", cmap="RdBu_r", legend=True, 
            vmin=-diff_max, vmax=diff_max, ax=axs[1, 2])
    axs[1, 2].set_title("Model Difference (BYM - IID)")


    # Define the categories and corresponding colors
    category_colors = {0: 'gray', 1: 'blue', 2: 'red'}
    category_labels = {0: 'Similar', 1: 'BYM Better', 2: 'IID Better'}
    cmap = ListedColormap([category_colors[k] for k in sorted(category_colors.keys())])

    # Calculate which model performs better
    gdf['better_model'] = 0  # Tie (very close performance)
    threshold = 0.008  # Threshold for determining significantly better performance
    gdf.loc[gdf['absolute_error'] < gdf['iid_absolute_error'] - threshold, 'better_model'] = 1  # BYM better
    gdf.loc[gdf['iid_absolute_error'] < gdf['absolute_error'] - threshold, 'better_model'] = 2  # IID better


    # Ensure all categories exist in the data (temporarily)
    for key in category_colors:
        if key not in gdf["better_model"].values:
            gdf.loc[len(gdf)] = {**gdf.iloc[0], "better_model": key}  # Add dummy row

    # Plot
    gdf.plot(column="better_model", cmap=cmap, legend=False, ax=axs[1, 3],
            categorical=True)

    # Manually create the legend with all categories
    legend_patches = [mpatches.Patch(color=category_colors[k], label=category_labels[k]) for k in category_labels]
    axs[1, 3].legend(handles=legend_patches, title="Better Model", loc="upper right")

    axs[1, 3].set_title("Better Performing Model")


    plt.tight_layout()
    plt.savefig(f"{sample_output_dir}/model_comparison_maps.png", dpi=300)
    plt.close()

    # 2. Error analysis comparison
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Bias comparison
    bias_max = max(abs(gdf['bias'].min()), abs(gdf['bias'].max()), 
                  abs(gdf['iid_bias'].min()), abs(gdf['iid_bias'].max()))

    gdf.plot(column="bias", cmap="RdBu_r", legend=True, 
            vmin=-bias_max, vmax=bias_max, ax=axs[0, 0])
    axs[0, 0].set_title("BYM Model: Bias")

    gdf.plot(column="iid_bias", cmap="RdBu_r", legend=True, 
            vmin=-bias_max, vmax=bias_max, ax=axs[0, 1])
    axs[0, 1].set_title("IID Model: Bias")

    # Variance comparison
    var_max = max(gdf['variance'].max(), gdf['iid_variance'].max())

    gdf.plot(column="variance", cmap="Purples", legend=True, 
            vmin=0, vmax=var_max, ax=axs[1, 0])
    axs[1, 0].set_title("BYM Model: Variance")

    gdf.plot(column="iid_variance", cmap="Purples", legend=True, 
            vmin=0, vmax=var_max, ax=axs[1, 1])
    axs[1, 1].set_title("IID Model: Variance")

    # Within CI comparison
    fig_ci, axs_ci = plt.subplots(1, 2, figsize=(12, 6))

    gdf.plot(column="within_CI", cmap="RdYlGn", legend=True, 
            categorical=True, ax=axs_ci[0])
    axs_ci[0].set_title("BYM Model: True Value Within 95% CI")

    gdf.plot(column="iid_within_CI", cmap="RdYlGn", legend=True, 
            categorical=True, ax=axs_ci[1])
    axs_ci[1].set_title("IID Model: True Value Within 95% CI")

    plt.tight_layout()
    plt.savefig(f"{sample_output_dir}/ci_coverage_comparison.png", dpi=300)
    plt.close()

    # Add variance comparison map
    # Create a separate figure for variance comparison
    fig_var, axs_var = plt.subplots(1, 3, figsize=(18, 6))

    # Variance maps for both models
    var_max = max(gdf['variance'].max(), gdf['iid_variance'].max())

    gdf.plot(column="variance", cmap="Purples", legend=True, 
            vmin=0, vmax=var_max, ax=axs_var[0])
    axs_var[0].set_title("BYM Model: Variance")

    gdf.plot(column="iid_variance", cmap="Purples", legend=True, 
            vmin=0, vmax=var_max, ax=axs_var[1])
    axs_var[1].set_title("IID Model: Variance")

    # Variance difference map
    gdf['variance_difference'] = gdf['variance'] - gdf['iid_variance']
    var_diff_max = max(abs(gdf['variance_difference'].min()), abs(gdf['variance_difference'].max()))
    gdf.plot(column="variance_difference", cmap="RdBu_r", legend=True, 
            vmin=-var_diff_max, vmax=var_diff_max, ax=axs_var[2])
    axs_var[2].set_title("Variance Difference (BYM - IID)")

    plt.tight_layout()
    plt.savefig(f"{sample_output_dir}/variance_comparison_maps.png", dpi=300)
    plt.close()

    # Optionally, add a scatter plot comparing variances
    fig_var_scatter, ax_var_scatter = plt.subplots(figsize=(8, 8))
    ax_var_scatter.scatter(gdf['variance'], gdf['iid_variance'])
    ax_var_scatter.plot([0, var_max], [0, var_max], 'k--')  # Diagonal line
    ax_var_scatter.set_xlabel('BYM Model Variance')
    ax_var_scatter.set_ylabel('IID Model Variance')
    ax_var_scatter.set_title('BYM vs IID Model Variance Comparison')
    ax_var_scatter.grid(True, alpha=0.3)

    # Add annotations for regions with biggest differences
    n_to_annotate = 3  # Number of regions to annotate
    top_diff_indices = gdf['variance_difference'].abs().nlargest(n_to_annotate).index
    for idx in top_diff_indices:
        row = gdf.loc[idx]
        ax_var_scatter.annotate(
            f"Region {idx}", 
            (row['variance'], row['iid_variance']),
            xytext=(5, 5), textcoords='offset points'
        )

    plt.tight_layout()
    plt.savefig(f"{sample_output_dir}/variance_scatter_comparison.png", dpi=300)
    plt.close()
    # 3. Scatter plot comparison
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # BYM model scatter
    axs[0].scatter(gdf['True Prevalence'], gdf['mean'], label='BYM', alpha=0.7)
    axs[0].plot([0, 1], [0, 1], 'k--')  # Add diagonal reference line
    axs[0].set_xlabel('True Prevalence')
    axs[0].set_ylabel('Estimated Value')
    axs[0].set_title('BYM vs IID Estimates')
    axs[0].grid(True, alpha=0.3)

    # Add IID model to the same plot
    axs[0].scatter(gdf['True Prevalence'], gdf['iid_mean'], label='IID', alpha=0.7, marker='x')
    axs[0].legend()

    # Plot error distribution for both models
    axs[1].hist(gdf['bias'], bins=10, alpha=0.5, label='BYM Bias')
    axs[1].hist(gdf['iid_bias'], bins=10, alpha=0.5, label='IID Bias')
    axs[1].axvline(x=0, color='k', linestyle='--')
    axs[1].set_xlabel('Bias (Estimated - True)')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Distribution of Bias')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"{sample_output_dir}/model_accuracy_comparison.png", dpi=300)
    plt.close()

    # Calculate summary statistics for both models
    summary_stats = {
        'bym_mean_absolute_error': float(gdf['absolute_error'].mean()),
        'iid_mean_absolute_error': float(gdf['iid_absolute_error'].mean()),
        'bym_root_mean_squared_error': float(np.sqrt(gdf['squared_error'].mean())),
        'iid_root_mean_squared_error': float(np.sqrt(gdf['iid_squared_error'].mean())),
        'bym_mean_bias': float(gdf['bias'].mean()),
        'iid_mean_bias': float(gdf['iid_bias'].mean()),
        'bym_coverage_probability': float(gdf['within_CI'].mean() * 100),
        'iid_coverage_probability': float(gdf['iid_within_CI'].mean() * 100),
        'bym_mean_variance': float(gdf['variance'].mean()),
        'iid_mean_variance': float(gdf['iid_variance'].mean()),
    }

    # Save comparison metrics to YAML
    with open(f"{sample_output_dir}/model_comparison_metrics.yaml", 'w') as f:
        yaml.dump(summary_stats, f)

    # Create a summary comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Restructure data for grouped bar chart
    metrics = ['mean_absolute_error', 'root_mean_squared_error', 'mean_bias', 
              'coverage_probability', 'mean_variance']
    bym_values = [summary_stats[f'bym_{m}'] for m in metrics]
    iid_values = [summary_stats[f'iid_{m}'] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, bym_values, width, label='BYM Model')
    ax.bar(x + width/2, iid_values, width, label='IID Model')

    ax.set_xticks(x)
    ax.set_xticklabels([' '.join(m.split('_')).title() for m in metrics], rotation=45, ha='right')
    ax.set_title('Model Comparison Metrics')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Add text labels
    for i, v in enumerate(bym_values):
        ax.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center', va='bottom', rotation=90)
        
    for i, v in enumerate(iid_values):
        ax.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center', va='bottom', rotation=90)

    plt.tight_layout()
    plt.savefig(f"{sample_output_dir}/model_comparison_metrics.png", dpi=300)
    plt.close()

    return summary_stats

def analyze_with_resampling(config):
    generate_resamples()
    resample_n = config.get("RESAMPLE_N")
    for i in tqdm(range(resample_n), "Running spatial analysis with new samples and regions"):
        summary_stats = run_spatial_analysis(config, i)



if __name__ == "__main__":
  config = load_config()
  clear_memory()
  # analyze_with_resampling(config)
  compare_resamples(config)