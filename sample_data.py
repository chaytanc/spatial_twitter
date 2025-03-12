import preprocess as p
from finetune import load_other_text_data, load_text_data
import yaml
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
import pickle

with open("params.yaml", 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        config = None
N_CLUSTERS = config.get("N_CLUSTERS")
SAMPLES_PER_CLUSTER = config.get("SAMPLES_PER_CLUSTER")
n = N_CLUSTERS * SAMPLES_PER_CLUSTER

def code_anti_pro(half=1):
    ''' Returns a vector one hot encoded to indicate a pro tweet. Makes many assumptions about specific training circumstances'''
    # current embeddings.dat was cut by 2
    if half == 1:
        _, anti, pro = load_text_data(cut_data_by=2)
    else:
        _, anti, pro = load_other_text_data(cut_data_by=2)
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

def get_dynamic_bbox(points, margin=1.0):
    """
    Calculate a dynamic bounding box based on the minimum and maximum coordinates of the points.
    margin: Add a margin around the bounding box to ensure the polygons are fully enclosed.
    """
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    # Add margin to the bounding box
    return (min_x - margin, max_x + margin, min_y - margin, max_y + margin)

def close_voronoi_region(vor, region, bbox):
    vertices = [vor.vertices[i] for i in region if i != -1]  # Keep finite vertices
    if len(vertices) >= 3:  # Check if there are enough points to form a polygon
        return Polygon(vertices)
    
    # If there are infinite edges (region contains -1), close the polygon
    # Extend edges of the infinite regions to the bounding box
    region_points = []
    for i in region:
        if i != -1:
            region_points.append(vor.vertices[i])
        else:
            # TODO have to intersect bounding box with lines in voronoi in order to close the polygons which is a huge pain in the ass
            # Handle infinite edges: extend to the bounding box
            # Find the nearest bounding box points to create a closed edge
            region_points.append(bbox)  # Add the bounding box vertices

    # Return the closed polygon for infinite regions
    return Polygon(region_points)

def voronoi_polygons(vor, margin=1.0):
    """
    Convert a Voronoi diagram into a list of closed polygons.
    Automatically generates the bounding box based on the Voronoi points.
    margin: Optional margin to extend beyond the points for closing polygons.
    """
    polygons = []
    
    # Get the bounding box based on the Voronoi points with a margin
    bbox = get_dynamic_bbox(vor.points, margin)
    bbox_poly = Polygon([(bbox[0], bbox[2]), (bbox[1], bbox[2]),
                         (bbox[1], bbox[3]), (bbox[0], bbox[3])])

    for i, center in enumerate(vor.points):
        region_index = vor.point_region[i]  # Get corresponding region index
        region = vor.regions[region_index]  # Get region vertices

        if -1 in region or not region:  # Handle infinite regions
            polygon = close_voronoi_region(vor, region, bbox)
        else:
            polygon = Polygon([vor.vertices[j] for j in region])  # Create the polygon

        polygons.append(polygon)

    return polygons


def map(samples):
    two_dim, pca = p.map_to_2d(samples)
    kmeans = p.cluster(two_dim)
    centers = kmeans.cluster_centers_
    # boundaries and shape file
    vor = Voronoi(centers)
    # polygons, center_map = voronoi_polygons(vor)
    # ordered_polygons = [center_map[tuple(center)] for center in centers]
    polygons = voronoi_polygons(vor)

    gdf = gpd.GeoDataFrame(
        pd.DataFrame({'Cluster_ID': range(len(polygons))}),
        geometry=polygons
    )
    gdf = gdf.dropna(subset=['geometry'])
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

def load_sample():
    with open(config["SAMPLE_FILE"], "rb") as f:
        df = pickle.load(f)
        f.close()
    with open(config["SAMPLE_PCA_FILE"], "rb") as f:
        pca = pickle.load(f)
        f.close()
    with open(config["SAMPLE_KMEANS_FILE"], "rb") as f:
        kmeans = pickle.load(f)
        f.close()
    gdf = gpd.read_file(config["SAMPLE_SHAPE_FILE"])
    return df, pca, kmeans, gdf

if __name__ == "__main__":
    create_sampled_dataframe()
