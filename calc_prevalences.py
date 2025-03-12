from finetune import load_text_data
from preprocess import clear_memory, pickle_data, load_pickled_dataset, map_to_2d
import yaml
from EmbeddingTextDataset import EmbeddingTextDataset
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
from sample_data import code_anti_pro, load_sample
# Calculate actual number of pro brexit tweets in each region within the polygons stored in the shp file in sample

with open("params.yaml", 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        config = None

# load embeddings from both pickled files (lowkey you fucked up the .dat files but thank god do have both pkl files...)
dataset = load_pickled_dataset(filename=config["EMBEDDING_PKL_FILE"]).embeddings
dataset2 = load_pickled_dataset(filename=config["EMBEDDING_PKL_FILE2"]).embeddings
# count the pro and anti in each file and keep track of which tweets are which in a dataframe
first_half_labels, tweets = code_anti_pro(half=1)
second_half_labels, other_tweets = code_anti_pro(half=2)
labels = np.concat([first_half_labels, second_half_labels])
data = np.concat([dataset, dataset2], axis=0)
print(data.shape)
assert(len(data) == len(labels))
# load the kmeans and pca from the sample/ dir
sample_df, pca, kmeans, gdf = load_sample()

# put them in 2 dim and cluster every single point
two_dim, pca = map_to_2d(data)
pred_clusters = kmeans.predict(two_dim)
# get labels and count how many in each cluster
n = np.bincount(pred_clusters)
# Combine coding of anti-pro with num clusters
# elementwise multiply anti-pro w pred_clusters to get only pro in clusters
# first have to turn zeroes in labels to -1 because 0 is a valid cluster label
labels = np.where(labels == 0, -1, labels)
only_pro = np.multiply(labels, pred_clusters).astype(np.int8)
# zero is a real problem child
only_pro = np.where((labels == -1) & (only_pro == 0), -1, only_pro)
# bin count to get num pro in each cluster
pro_count_per_cluster = np.bincount(only_pro[only_pro >= 0])
# divide pro by n in clusters to get percentage true / true prevs per cluster
true_prevs = pro_count_per_cluster / n 

# add cluster ids to dataframe
df = pd.DataFrame({"Cluster_ID": range(30), "n" : n, "True Prevalence": true_prevs, })
pickle_data(df, config["TRUE_PREVS_FILE"])
