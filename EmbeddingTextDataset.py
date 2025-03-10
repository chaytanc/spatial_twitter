
from torch.utils.data import Dataset
import torch
import numpy as np
MAX_LENGTH = 30
EMBEDDING_DIM = 768  # Adjust based on your model

# Dataset class
class EmbeddingTextDataset(Dataset):
    def __init__(self, mmap_path, texts, tokenizer, max_length=MAX_LENGTH):
        self.embeddings = np.memmap(mmap_path, dtype="float32", mode="r", shape=(len(texts), max_length, EMBEDDING_DIM))
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        tokenized_output = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "inputs_embeds": embedding,
            "labels": tokenized_output["input_ids"].squeeze(0),
        }

