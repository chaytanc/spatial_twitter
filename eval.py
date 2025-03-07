import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from preprocess import embed



# Custom dataset class (same as training dataset)
class EmbeddingTextDataset(Dataset):
    def __init__(self, embeddings, texts, tokenizer):
        self.embeddings = embeddings
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        input_embedding = self.embeddings[idx]
        text = self.texts[idx]
        tokenized_output = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=30)

        return {
            "inputs_embeds": input_embedding.squeeze(0),
            "labels": tokenized_output["input_ids"].squeeze(0)
        }

def get_eval_dataset(tokenizer):
    # Load the evaluation data
    eval_df = pd.read_csv("trumptweets1205-127.csv", encoding='utf-8', encoding_errors='ignore')
    eval_texts = eval_df["Tweet"].dropna().tolist()
    eval_embeddings = list(map(embed, tqdm(eval_texts, desc="Generating Evaluation Embeddings", leave=False)))
    eval_dataset = EmbeddingTextDataset(eval_embeddings, eval_texts, tokenizer)
    return eval_dataset
