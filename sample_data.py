import preprocess as p

N_CLUSTERS = 30
SAMPLES_PER_CLUSTER = 500
# calc number of samples to take
n = N_CLUSTERS * SAMPLES_PER_CLUSTER
# load embeddings
dataset = p.load_pickled_dataset(filename=EMBEDDING_PKL_FILE)
# generate n random sample indicies
# index embeddings
# get corresponding tweet and pro/anti stance
    # code indices of embeddings as pro/anti (first half pro)
# create dataframe and pickle the sampled data


