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
import torch
from tqdm import tqdm

# Models
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Do some setup
# Headers:
#,Date,Headline,URL,Opening Text,Hit Sentence,Source,Influencer,Country,Subregion,Language,Reach,Desktop Reach,Mobile Reach,Twitter Social Echo,Facebook Social Echo,Reddit Social Echo,National Viewership,Engagement,AVE,Sentiment,Key Phrases,Input Name,Keywords,Twitter Authority,Tweet Id,Twitter Id,Twitter Client,Twitter Screen Name,User Profile Url,Twitter Bio,Twitter Followers,Twitter Following,Alternate Date Format,Time,State,City,Document Tags 
anti_f = "dataverse_files/TweetDataset_AntiBrexit_Jan-Mar2022.csv"
pro_f = "dataverse_files/TweetDataset_ProBrexit_Jan-Mar2022.csv"
anti_df = pd.read_csv(anti_f)
pro_df = pd.read_csv(pro_f)
scaler = StandardScaler().set_output(transform="pandas")


# TODO try to get perfect reconstructions from BART, and if that doesn't work, try prompt engineering
# to ask for "repeat the meaning of these embeddings without adding any additional output tokens" from 
# any generative model by appending embeddings for the system prompt to the embedding from the clusters
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load BART tokenizer and model
model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# def get_bart_embedding(text):
#     """Extract token and sentence embeddings from BART"""
#     inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
#     input_ids = inputs["input_ids"]

#     with torch.no_grad():
#         encoder_outputs = model.model.encoder(input_ids)

#     token_embeddings = encoder_outputs[0]  # Token-wise embeddings
#     sentence_embedding = token_embeddings.mean(dim=1)  # Mean pooling for sentence embedding

#     return token_embeddings, sentence_embedding, input_ids
# 
# def embedding_to_text(embedding, input_ids):
#     """Decode embeddings back into text using BART's decoder"""
#     with torch.no_grad():
#         generated_ids = model.generate(
#             input_ids=input_ids,  # Providing original input_ids helps guide reconstruction
#             max_length=20
#         )
    
#     return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# Example usage
# word_embeddings, sentence_embedding, input_ids = get_bart_embedding("Hello world!")
# decoded_text = embedding_to_text(word_embeddings, input_ids)
# print("Decoded Text:", decoded_text)  # Expected: "Hello world!"

import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

text = "My friends are cool but they eat too many carbs."
inputs = tokenizer(text, return_tensors="jax")
encoder_outputs = model.encode(**inputs)

decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

generated_ids = model.generate(
    decoder_input_ids,
    encoder_outputs=encoder_outputs,
    num_return_sequences=1,
    do_sample=False  # Turn off sampling for deterministic output
)

# Decode generated token IDs into text
decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("Decoded Text:", decoded_text)

# Do we want to cluster before or after mapping to 2d? clustering before might retain the rich high dimensional data before we pca, but then if we do PCA on individual clusters we can't put them on the same map... or can't we?
# Clustering afterward is going to be necessary for sure in order to make our visualization readable (not having 100000 centroids)
def cluster(embeddings):
    # TODO n clusters???
    kmeans = KMeans(n_clusters=30, random_state=0, n_init="auto").fit(embeddings)
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
    anti = get_bart_embedding(anti_df["Hit Sentence"][:1000])
    pro = get_bart_embedding(pro_df["Hit Sentence"][:1000])
    print("pro0", pro[0])
    # TODO concat pro and anti?
    two_dim, pca = map_to_2d(pro)
    kmeans = cluster(two_dim)
    show_map(kmeans)
    reconstruct_pca_meanings(pca, kmeans.cluster_centers_)

# process()

#TODO decode the cluster centroids and get a representative tweet?
# TODO shading and prevalence estimates
# TODO -- need to mix pro and anti tweets in clustering data / not do separately, but still keep track of which is which / have labels




'''

'''