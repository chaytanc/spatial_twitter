import preprocess as p
from finetune import load_text_data
import yaml
from EmbeddingTextDataset import EmbeddingTextDataset
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi
import pickle

with open("params.yaml", 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        config = None
N_CLUSTERS = config.get("N_CLUSTERS")
SAMPLES_PER_CLUSTER = 500
n = N_CLUSTERS * SAMPLES_PER_CLUSTER

def code_anti_pro():
    ''' Returns a vector one hot encoded to indicate a pro tweet. Makes many assumptions about specific training circumstances'''
    # current embeddings.dat was cut by 2
    _, anti, pro = load_text_data(cut_data_by=2)
    num_anti = len(anti)
    num_pro = len(pro)
    # Assuming embeddings.shape[0] == num_anti + num_pro
    labels = np.concatenate([np.zeros(num_anti), np.ones(num_pro)])
    return labels, anti + pro

def make_sample():
    # load embeddings
    dataset = p.load_pickled_dataset(filename=config["EMBEDDING_PKL_FILE"])
    # generate n random sample indicies
    indices = np.random.choice(dataset.embeddings.shape[0], size=n, replace=False)
    # index embeddings
    sample = dataset.embeddings[indices,:,:]
    return sample, indices

def voronoi_polygons(vor):
    polygons = []
    for region in vor.regions:
        if not region or -1 in region:  # Skip infinite regions
            continue
        polygons.append(Polygon([vor.vertices[i] for i in region]))
    return polygons

def map(samples):
    two_dim, pca = p.map_to_2d(samples)
    kmeans = p.cluster(two_dim)
    centers = kmeans.cluster_centers_
    # boundaries and shape file
    vor = Voronoi(centers)
    polygons = voronoi_polygons(vor)
    # TODO am assumign that voronoi maintains the order that kmeans.cluster_centers gives
    gdf = gpd.GeoDataFrame(pd.DataFrame({'Cluster_ID': range(len(polygons))}),
                        geometry=polygons)
    return two_dim, pca, kmeans, gdf

def create_sampled_dataframe():
    # Load embeddings and randomly sample
    sample_embeddings, indices = make_sample()
    anti_pro_labels, all_texts = code_anti_pro()
    
    # Extract metadata for sampled indices
    sampled_texts = np.array(all_texts)[indices]
    sampled_labels = anti_pro_labels[indices]

    two_dim, pca, kmeans, gdf = map(sample_embeddings)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Tweet": sampled_texts,
        "Embedding_Index": indices, # We're going to store the embedding index, not the embedding itself, and index into embeddings.dat
        # "Embedding": list(sample_embeddings),  # Store as list to avoid issues with high-dimensional arrays
        "Stance": sampled_labels,  # 1 = Pro, 0 = Anti
        "Cluster": kmeans.labels_,
        "x": two_dim[:, 0],
        "y": two_dim[:, 1]
    })

    pickle_it_up(df, pca, kmeans, gdf)
    
def pickle_it_up(df, pca, kmeans, gdf):
    with open(config["SAMPLE_FILE"], "wb") as f:
        pickle.dump(df, f)
        f.close()
    with open(config["SAMPLE_PCA_FILE"], "wb") as f:
        pickle.dump(pca, f)
        f.close()
    with open(config["SAMPLE_KMEANS_FILE"], "wb") as f:
        pickle.dump(kmeans, f)
        f.close()
    gdf.to_file(config["SAMPLE_SHAPE_FILE"])



create_sampled_dataframe()
