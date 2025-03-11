import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from EmbeddingTextDataset import EmbeddingTextDataset
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import gc
import os
import shutil
import pickle
import yaml

with open("params.yaml", 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        config = None

# Models
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Do some setup
# Headers:
#,Date,Headline,URL,Opening Text,Hit Sentence,Source,Influencer,Country,Subregion,Language,Reach,Desktop Reach,Mobile Reach,Twitter Social Echo,Facebook Social Echo,Reddit Social Echo,National Viewership,Engagement,AVE,Sentiment,Key Phrases,Input Name,Keywords,Twitter Authority,Tweet Id,Twitter Id,Twitter Client,Twitter Screen Name,User Profile Url,Twitter Bio,Twitter Followers,Twitter Following,Alternate Date Format,Time,State,City,Document Tags 
anti_f = config["ANTI_FILE"]
pro_f = config["PRO_FILE"]
anti_df = pd.read_csv(anti_f)
pro_df = pd.read_csv(pro_f)
scaler = StandardScaler().set_output(transform="pandas")
MAX_LENGTH = config.get("MAX_LENGTH")
EMBEDDING_DIM = config.get("EMBEDDING_DIM") 
EMBEDDING_FILE = "test_embeddings.dat"
EMBEDDING_PKL_FILE = "test_embeds.pkl"

N_CLUSTERS = config.get("N_CLUSTERS")

from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

#TODO use the same models across files kind of sort of maybe
# TODO distilgpt2????
model = GPT2Model.from_pretrained('gpt2', output_hidden_states=True)
recon_model = GPT2LMHeadModel.from_pretrained('gpt2')
# recon_model = GPT2LMHeadModel.from_pretrained("./finetuned_gpt2_embeddings")
# recon_tokenizer = GPT2Tokenizer.from_pretrained("./finetuned_gpt2_embeddings")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if torch.backends.mps.is_available():
    try:
        model.to(device)  # Try to use MPS GPU
        recon_model.to(device)
    except RuntimeError as e:
        print("MPS GPU out of memory, switching to CPU...")
        model.to("cpu")  # Fall back to CPU


def clear_memory(embeddings_mmap=None):
    """Frees up memory by clearing caches and deleting large objects."""
    
    # If memory-mapped file exists, flush & close it
    if embeddings_mmap is not None:
        embeddings_mmap.flush()  # Ensure data is written to disk
        del embeddings_mmap

    # Clear PyTorch GPU cache (for MPS or CUDA)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("Memory cleanup completed.")


# Pickle dataset
def pickle_data(dataset, filename):
    with open(filename, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved dataset to {filename}")


def load_pickled_dataset(filename):
    """Loads a pickled dataset from a .pkl file."""
    with open(filename, "rb") as f:
        dataset = pickle.load(f)
        f.close()
    print(f"Loaded dataset from {filename}")
    return dataset


def copy_embeddings_file(source_file, backup_dir="embeddings_backup"):
    """Copies the embeddings file to a backup location to avoid overwriting."""
    # Ensure the backup directory exists
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Define the backup file path
    filename = os.path.basename(source_file)
    backup_path = os.path.join(backup_dir, filename)
    
    # Copy the file, preserving metadata
    shutil.copy2(source_file, backup_path) 
    
    print(f"Embeddings file copied to {backup_path}")


def embed(text):
    input_ids = tokenizer(text, 
                          return_tensors="pt", 
                          padding="max_length",
                          truncation=True,
                          max_length=MAX_LENGTH)['input_ids']
    input_ids = input_ids.to(device)
    with torch.no_grad():
        embeddings = model(input_ids).last_hidden_state

    del input_ids
    gc.collect() 
    return embeddings


# Generate embeddings and store in memory-mapped file
def generate_embeddings(texts, mmap_file=EMBEDDING_FILE):
    num_texts = len(texts)
    embeddings_mmap = np.memmap(mmap_file, dtype="float32", mode="w+", shape=(num_texts, MAX_LENGTH, EMBEDDING_DIM))

    # Find the last written index
    # TODO does this work?
    last_written_idx = 0
    for i in range(num_texts):
        if np.count_nonzero(embeddings_mmap[i]) == 0:
            break
        else:
            last_written_idx += 1
    # last_written_idx = min(last_written_idx, num_texts - 1)

    print(f"Resuming from index {last_written_idx}...")

    for i in tqdm(range(last_written_idx, num_texts), desc="Generating Embeddings"):
        embedding = embed(texts[i]).detach().cpu().numpy().squeeze(0)
        embeddings_mmap[i, :embedding.shape[0], :] = embedding
        embeddings_mmap.flush()

    print("Embedding generation complete!")
    copy_embeddings_file("embeddings.dat", "embeddings_backup")
    clear_memory(embeddings_mmap=embeddings_mmap)
    return mmap_file


def reconstruct_embedding(embeddings):
    embeddings = embeddings.to(torch.float32)  # Force embeddings to float32
    recon_model.eval()
    with torch.no_grad():
        outputs = recon_model.forward(inputs_embeds=embeddings)
        decode_ids = torch.argmax(outputs.logits, -1)
    decoded_text = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)
    # decoded_text = tokenizer.batch_decode()
    print(decoded_text)
    return decoded_text 

# test_em = embed("test with multiple tokens and embeddings i hope").to(torch.float32)  # Ensure float32
# reconstruct_embedding(test_em)


# Do we want to cluster before or after mapping to 2d? clustering before might retain the rich high dimensional data before we pca, but then if we do PCA on individual clusters we can't put them on the same map... or can't we?
# Clustering afterward is going to be necessary for sure in order to make our visualization readable (not having 100000 centroids)
def cluster(embeddings):
    # TODO n clusters???
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init="auto").fit(embeddings)
    print("kmeans centers", kmeans.cluster_centers_)
    return kmeans

def neighborhood(embeddings):
    clf = KNeighborsClassifier(n_neighbors=20)
    #TODO

def standardize(two_dim):
    std_two_dim = scaler.fit_transform(two_dim)
    #TODO nearest neighbors + built in voronoi

def map_to_2d(embeddings):
    # Note: can get the axes of our new space with pca.components_
    pca = PCA(n_components=2)
    embeddings = np.mean(embeddings, axis=1)
    pca.fit(embeddings)
    two_dim = pca.transform(embeddings)
    return two_dim, pca

# def reconstruct_pca_meanings(pca, reduced):
#     reconstructions = pca.inverse_transform(reduced)
#     reconstructed_meanings = model.decode(reconstructions)
#     print(reconstructed_meanings)
#     return reconstructed_meanings

def show_map(kmeans):
    # https://stackoverflow.com/questions/49347738/drawing-boundary-lines-based-on-kmeans-cluster-centres
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:,0],centers[:,1], marker='s', s=100)
    vor = Voronoi(centers)
    fig = voronoi_plot_2d(vor,plt.gca())
    plt.show()

def process(load=False):
    # anti = generate_embeddings(texts=anti_df["Hit Sentence"][:1000])
    texts = pro_df["Hit Sentence"][:1000]
    if load:
        dataset = load_pickled_dataset(EMBEDDING_PKL_FILE)
    else:
        mmap_file = generate_embeddings(texts)
        dataset = EmbeddingTextDataset(mmap_file, texts, tokenizer)
        pickle_data(dataset, EMBEDDING_PKL_FILE)
    # TODO concat pro and anti?
    two_dim, pca = map_to_2d(dataset.embeddings)
    kmeans = cluster(two_dim)
    show_map(kmeans)
    # reconstruct_pca_meanings(pca, kmeans.cluster_centers_)

# process()

#TODO decode the cluster centroids and get a representative tweet?
# TODO shading and prevalence estimates
# TODO -- need to mix pro and anti tweets in clustering data / not do separately, but still keep track of which is which / have labels

if __name__ == "__main__":
    process(load=True)
    # reconstruct_embedding(test_em)
