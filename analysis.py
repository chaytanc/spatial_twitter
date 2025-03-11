from matplotlib import pyplot as plt
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
ro.r(r_code)
result = ro.r['readRDS']("inla_result.rds")
summary_fitted_values = result.rx2("summary.fitted.values") 

# Convert to Pandas DataFrame
df = pd2ri.rpy2py(summary_fitted_values)
df['Cluster_ID'] = range(0, len(df))
gdf = gdf.merge(df, left_on="Cluster_ID", right_on="Cluster_ID", how="left")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
gdf.plot(column="0.5quant", cmap="coolwarm", legend=True, ax=ax)
ax.set_title("INLA Spatial Effects")
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

gdf.plot(column="mean", cmap="coolwarm", legend=True, ax=axs[0])
axs[0].set_title("Mean Probability")

# Plot lower credible interval
gdf.plot(column="0.025quant", cmap="coolwarm", legend=True, ax=axs[1])
axs[1].set_title("Lower 95% CI")

# Plot upper credible interval
gdf.plot(column="0.975quant", cmap="coolwarm", legend=True, ax=axs[2])
axs[2].set_title("Upper 95% CI")

plt.tight_layout()
plt.savefig("inla_spatial_results.png", dpi=300)
plt.show()

