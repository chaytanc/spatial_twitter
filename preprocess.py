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
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Do some setup
# Headers:
#,Date,Headline,URL,Opening Text,Hit Sentence,Source,Influencer,Country,Subregion,Language,Reach,Desktop Reach,Mobile Reach,Twitter Social Echo,Facebook Social Echo,Reddit Social Echo,National Viewership,Engagement,AVE,Sentiment,Key Phrases,Input Name,Keywords,Twitter Authority,Tweet Id,Twitter Id,Twitter Client,Twitter Screen Name,User Profile Url,Twitter Bio,Twitter Followers,Twitter Following,Alternate Date Format,Time,State,City,Document Tags 
anti_f = "dataverse_files/TweetDataset_AntiBrexit_Jan-Mar2022.csv"
pro_f = "dataverse_files/TweetDataset_ProBrexit_Jan-Mar2022.csv"
anti_df = pd.read_csv(anti_f)
pro_df = pd.read_csv(pro_f)
scaler = StandardScaler().set_output(transform="pandas")

from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

# Instantiate the model and tokenizer
# model = AutoModelCausalLM.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', output_hidden_states=True)
recon_model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def embed(text):
    input_ids = tokenizer(text, return_tensors="pt")['input_ids']
    # Note: no positional embeddings... also these are embeddings before they've gone through the transformer, not after
    # embeddings = model.transformer.wte.weight[input_ids] AutoModel
    embeddings = model(input_ids).last_hidden_state
    # TODO embeddings = model(**encoded_input).last_hidden_state ??
    # embeddings = model(**encoded_input).last_hidden_state
    # https://stackoverflow.com/questions/75547772/recovering-input-ids-from-input-embeddings-using-gpt-2
    return embeddings

test_em = embed("test with multiple tokens and embeddings i hope")
def reconstruct_embedding(embeddings):
    sys_prompt = "Repeat the precise meaning of the following without adding any additional characters whatsoever: "
    em_sys_prompt = embed(sys_prompt)
    reconstruction_embedding = torch.cat((em_sys_prompt, embeddings), 1)
    recon_model.eval()
    with torch.no_grad():
        # Pass embedding through GPT-2 transformer layers
        outputs = recon_model(inputs_embeds=reconstruction_embedding)  # Generate logits over vocabulary
    token_ids = torch.argmax(outputs.logits, dim=-1)
    # decoded_text = tokenizer.decode(pred_ids)
    decoded_text = tokenizer.decode(token_ids[0])
    print(decoded_text)
    return reconstruction_embedding
reconstruct_embedding(test_em)


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
    anti = embed(anti_df["Hit Sentence"][:1000])
    pro = embed(pro_df["Hit Sentence"][:1000])
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