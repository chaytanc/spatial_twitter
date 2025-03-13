from pathlib import Path
import preprocess as p
from finetune import load_other_text_data, load_text_data
from EmbeddingTextDataset import EmbeddingTextDataset
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
    # TODO weighted sampling?? load true.pkl from sample/prev/true.pkl and use n col from df
    # index embeddings
    sample = dataset.embeddings[indices,:,:]
    return sample, indices


def bounded_voronoi(points):
    """
    Compute Voronoi polygons and intersect them with a boundary.
    
    A simpler alternative approach that avoids handling infinite regions directly:
    1. Add distant points around the boundary to ensure all regions are closed
    2. Compute Voronoi diagram with these extra points
    3. Extract only the regions for the original points
    4. Intersect with the boundary
    """
    # Create boundary as Shapely polygon 
    min_x, min_y = np.min(points, axis=0) - 1
    max_x, max_y = np.max(points, axis=0) + 1
    boundary = Polygon([
        (min_x, min_y), (max_x, min_y),
        (max_x, max_y), (min_x, max_y)
    ])
    
    # Get boundary coordinates
    boundary_coords = np.array(boundary.exterior.coords)
    
    # Calculate the range of the input points to scale distant points
    x_range = np.ptp(points[:, 0]) * 10
    y_range = np.ptp(points[:, 1]) * 10
    
    # Add distant points around boundary to close regions
    # (much farther than the boundary, but not too far to cause numerical issues)
    min_x, min_y = np.min(boundary_coords, axis=0) - [x_range, y_range]
    max_x, max_y = np.max(boundary_coords, axis=0) + [x_range, y_range]
    
    distant_points = np.array([
        [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y],
        [min_x, (min_y+max_y)/2], [max_x, (min_y+max_y)/2],
        [(min_x+max_x)/2, min_y], [(min_x+max_x)/2, max_y]
    ])
    
    # Combine original and distant points
    all_points = np.vstack([points, distant_points])
    
    # Compute Voronoi diagram with all points
    vor = Voronoi(all_points)
    
    # Extract polygons for original points only
    polygons = []
    for i in range(len(points)):  # Only consider original points
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        
        # Skip empty or unbounded regions (though there should be none with distant points)
        if -1 in region or len(region) == 0:
            continue
            
        # Create polygon
        vertices = [vor.vertices[v] for v in region]
        cell = Polygon(vertices)
        
        # Clip to boundary
        cell = cell.intersection(boundary)
        
        if not cell.is_empty and cell.area > 0:
            polygons.append(cell)
    
    return polygons


def map(samples):
    two_dim, pca = p.map_to_2d(samples)
    kmeans = p.cluster(two_dim)
    centers = kmeans.cluster_centers_
    # boundaries and shape file
    vor = Voronoi(centers)
    # polygons, center_map = voronoi_polygons(vor)
    # ordered_polygons = [center_map[tuple(center)] for center in centers]
    # polygons = voronoi_polygons(vor)
    polygons = bounded_voronoi(vor.points)

    gdf = gpd.GeoDataFrame(
        pd.DataFrame({'Cluster_ID': range(len(polygons))}),
        geometry=polygons
    )
    gdf = gdf.dropna(subset=['geometry'])
    return two_dim, pca, kmeans, gdf

def create_sampled_dataframe(output_dir):
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

    pickle_it_up(df, pca, kmeans, gdf, output_dir)
    
def pickle_it_up(df, pca, kmeans, gdf, output_dir):
    with open(output_dir / "sample_df.pkl", "wb") as f:
        pickle.dump(df, f)
        f.close()
    with open(output_dir / "sample_pca.pkl", "wb") as f:
        pickle.dump(pca, f)
        f.close()
    with open(output_dir / "sample_kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)
        f.close()
    gdf.to_file(output_dir / "sample.shp")

def load_sample(output_dir):
    with open(output_dir / "sample_df.pkl", "rb") as f:
        df = pickle.load(f)
        f.close()
    with open(output_dir / "sample_pca.pkl", "rb") as f:
        pca = pickle.load(f)
        f.close()
    with open(output_dir / "sample_kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
        f.close()
    gdf = gpd.read_file(output_dir / "sample.shp")
    return df, pca, kmeans, gdf

def generate_resamples():
    output_dir = Path(config.get("SAMPLE_DIR", "sample/"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(config.get("RESAMPLE_N")):
        sample_dir = output_dir / str(i) 
        sample_dir.mkdir(exist_ok=True)
        create_sampled_dataframe(sample_dir)

if __name__ == "__main__":
    # create_sampled_dataframe("sample/-1/")
    generate_resamples()
