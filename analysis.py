from matplotlib import pyplot as plt
import numpy as np
from preprocess import load_pickled_dataset
from sample_data import load_sample
import yaml
import rpy2.robjects as ro
import os
import rpy2.robjects.pandas2ri as pd2ri

os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
os.environ['R_LIBS'] = "/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/library"
os.environ['R_LIBS_USER'] = "/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/library" 
# print(ro.r('version'))

with open("params.yaml", 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        config = None

sample_df, pca, kmeans, gdf = load_sample()
true_prevs = load_pickled_dataset(config["TRUE_PREVS_FILE"])
# Enable conversion between pandas and R dataframes
pd2ri.activate()
# Convert `sample_df` from pandas to R dataframe
r_sample_df = pd2ri.py2rpy(sample_df)
ro.globalenv['r_sample_df'] = r_sample_df
shapefile_path = "~/Desktop/chaytan/spatial_twitter/sample/sample.shp" #TODO yaml

r_code = """
library(INLA)
library(spdep) 
library(sf) 
library(prioritizr)
library(dplyr)

gdf <- st_read("~/Desktop/chaytan/spatial_twitter/sample/sample.shp")
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
saveRDS(result, file = "inla_result.rds")
"""

# Run R-INLA
# ro.r(r_code)
result = ro.r['readRDS']("analysis/inla_result.rds")
summary_fitted_values = result.rx2("summary.fitted.values") 

# Convert to Pandas DataFrame
df = pd2ri.rpy2py(summary_fitted_values)
df['Cluster_ID'] = range(0, len(df))
gdf = gdf.merge(df, left_on="Cluster_ID", right_on="Cluster_ID", how="left")
gdf = gdf.merge(true_prevs, on="Cluster_ID")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
gdf.plot(column="0.5quant", cmap="viridis", legend=True, ax=ax)
ax.set_title("INLA Spatial Effects")
plt.show()

fig, axs = plt.subplots(1, 4, figsize=(18, 6))

gdf.plot(column="mean", cmap="viridis", legend=True, ax=axs[0])
axs[0].set_title("Mean Probability")

# Plot lower credible interval
gdf.plot(column="0.025quant", cmap="viridis", legend=True, ax=axs[1])
axs[1].set_title("Lower 95% CI")

# Plot upper credible interval
gdf.plot(column="0.975quant", cmap="viridis", legend=True, ax=axs[2])
axs[2].set_title("Upper 95% CI")

gdf.plot(column="True Prevalence", cmap="viridis", legend=True, ax=axs[3])
axs[3].set_title("True Prevalences")

plt.tight_layout()
plt.savefig("analysis/inla_spatial_results.png", dpi=300)
plt.show()

### Further analysis ###

# Calculate error metrics
gdf['absolute_error'] = np.abs(gdf['mean'] - gdf['True Prevalence'])
gdf['squared_error'] = (gdf['mean'] - gdf['True Prevalence'])**2
gdf['bias'] = gdf['mean'] - gdf['True Prevalence']  # Positive means overestimation
gdf['within_CI'] = ((gdf['True Prevalence'] >= gdf['0.025quant']) & 
                    (gdf['True Prevalence'] <= gdf['0.975quant']))

# Calculate variance from the INLA results
# Variance can be computed from the SD column if available
if 'sd' in gdf.columns:
    gdf['variance'] = gdf['sd']**2
else:
    # Estimate variance from quantiles (approximate)
    gdf['variance'] = ((gdf['0.975quant'] - gdf['0.025quant']) / (2 * 1.96))**2

# Create figure for comparing estimates vs true values
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
plt.savefig("analysis/inla_spatial_validation.png", dpi=300)
plt.show()

# Create detailed comparative analysis plots
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
plt.savefig("analysis/inla_error_analysis.png", dpi=300)
plt.show()

# Calculate summary statistics
summary_stats = {
    'mean_absolute_error': gdf['absolute_error'].mean(),
    'root_mean_squared_error': np.sqrt(gdf['squared_error'].mean()),
    'mean_bias': gdf['bias'].mean(),
    'coverage_probability': gdf['within_CI'].mean() * 100,  # % of true values within CI
    'max_absolute_error': gdf['absolute_error'].max(),
    'min_absolute_error': gdf['absolute_error'].min(),
    'mean_variance': gdf['variance'].mean()
}

# Print summary statistics
print("\nSummary Statistics:")
for stat, value in summary_stats.items():
    print(f"{stat}: {value:.4f}")

# Create a summary visualization
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(summary_stats.keys(), summary_stats.values())
ax.set_xticklabels(summary_stats.keys(), rotation=45, ha='right')
ax.set_title('Error Metrics Summary')
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom', rotation=0)

plt.tight_layout()
plt.savefig("analysis/inla_summary_metrics.png", dpi=300)
plt.show()