import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

# Models
# Llama
from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
tokenizer.encode("Hello this is a test")

# Do some setup
# Headers:
#,Date,Headline,URL,Opening Text,Hit Sentence,Source,Influencer,Country,Subregion,Language,Reach,Desktop Reach,Mobile Reach,Twitter Social Echo,Facebook Social Echo,Reddit Social Echo,National Viewership,Engagement,AVE,Sentiment,Key Phrases,Input Name,Keywords,Twitter Authority,Tweet Id,Twitter Id,Twitter Client,Twitter Screen Name,User Profile Url,Twitter Bio,Twitter Followers,Twitter Following,Alternate Date Format,Time,State,City,Document Tags 
anti_f = "dataverse_files/TweetDataset_AntiBrexit_Jan-Mar2022.csv"
pro_f = "dataverse_files/TweetDataset_ProBrexit_Jan-Mar2022.csv"
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
anti_df = pd.read_csv(anti_f)
pro_df = pd.read_csv(pro_f)
scaler = StandardScaler().set_output(transform="pandas")


#TODO get rid of silly functions
def embed(tweets):
    embeddings = model.encode(tweets, show_progress_bar=True)
    return embeddings

# Do we want to cluster before or after mapping to 2d? clustering before might retain the rich high dimensional data before we pca, but then if we do PCA on individual clusters we can't put them on the same map... or can't we?
# Clustering afterward is going to be necessary for sure in order to make our visualization readable (not having 100000 centroids)
def cluster(embeddings):
    # TODO n clusters???
    kmeans = KMeans(n_clusters=30, random_state=0, n_init="auto").fit(embeddings)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)
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
    pca.fit(embeddings)
    two_dim = pca.transform(embeddings)
    return two_dim, pca

def reconstruct_pca_meanings(pca, reduced):
    reconstructions = pca.inverse_transform(reduced)
    reconstructed_meanings = model.decode(reconstructions)
    print(reconstructed_meanings)
    return reconstructed_meanings

def show_map(kmeans):
    # https://stackoverflow.com/questions/49347738/drawing-boundary-lines-based-on-kmeans-cluster-centres
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:,0],centers[:,1], marker='s', s=100)
    vor = Voronoi(centers)
    fig = voronoi_plot_2d(vor,plt.gca())
    plt.show()

def process():
    anti = embed(anti_df["Hit Sentence"][:1000])
    pro = embed(pro_df["Hit Sentence"][:1000])
    # TODO concat pro and anti?
    two_dim, pca = map_to_2d(pro)
    kmeans = cluster(two_dim)
    show_map(kmeans)
    reconstruct_pca_meanings(pca, two_dim)

process()

#TODO decode the cluster centroids and get a representative tweet?
# TODO shading and prevalence estimates
# TODO -- need to mix pro and anti tweets in clustering data / not do separately, but still keep track of which is which / have labels



