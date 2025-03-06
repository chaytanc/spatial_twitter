import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from preprocess import embed
import pandas as pd
import pickle
from tqdm import tqdm
import gc
import os
import numpy as np

# Load GPT-2 model & tokenizer
MAX_LENGTH = 30
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Ensure model is in train mode
model.train()

class EmbeddingTextDataset(torch.utils.data.Dataset):
    def __init__(self, mmap_path, texts, tokenizer):
        self.embeddings = np.memmap(mmap_path, dtype="float32", mode="r", shape=(len(texts), 768))
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]  # Reads directly from disk without loading all data
        text = self.texts[idx]
        tokenized_text = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")

        return torch.tensor(embedding), tokenized_text


torch.mps.empty_cache()
gc.collect()

# Load both datasets
anti_brexit_df = pd.read_csv("dataverse_files/TweetDataset_AntiBrexit_Jan-Mar2022.csv")
pro_brexit_df = pd.read_csv("dataverse_files/TweetDataset_ProBrexit_Jan-Mar2022.csv")

anti_brexit_texts = anti_brexit_df["Hit Sentence"].dropna().tolist()
pro_brexit_texts = pro_brexit_df["Hit Sentence"].dropna().tolist()
texts = anti_brexit_texts + pro_brexit_texts

print(f"Loaded {len(texts)} tweets.")

# Generate embeddings
# embeddings = list(map(embed, tqdm(texts, desc="Generating Embeddings", leave=False)))

# Define dimensions
num_texts = len(texts)
embedding_dim = 768  # Adjust based on your model

# Create a memory-mapped file
embeddings_mmap = np.memmap("embeddings.dat", dtype="float32", mode="w+", shape=(num_texts, MAX_LENGTH, embedding_dim))

# Step 1: Identify the last written index
last_written_idx = 0
for i in range(num_texts):
    if not np.all(embeddings_mmap[i] == 0):  # Check if this row is already written
        last_written_idx = i + 1
    else:
        break

# Step 2: Resume writing from last index
print(f"Resuming from index {last_written_idx}...")

for i in tqdm(range(last_written_idx, num_texts)):  # Start from last saved index
    embedding = embed(texts[i]).detach().cpu().numpy()  # Generate embedding
    embedding = embedding.squeeze(0)
    embeddings_mmap[i, :embedding.shape[0], :] = embedding  # Save to memory-mapped file
    embeddings_mmap.flush()  # Ensure data is saved to disk

print("Embedding generation complete!")

# Create dataset
# dataset = EmbeddingTextDataset(embeddings, texts, tokenizer)
# Use it:
dataset = EmbeddingTextDataset("embeddings.dat", texts, tokenizer)

# Function to pickle dataset
def pickle_data(dataset, filename="synthetic_tweet_embeddings.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved dataset to {filename}")

pickle_data(dataset)

# Training arguments with checkpointing
training_args = TrainingArguments(
    output_dir="./gpt2_embedding_finetune",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="steps",  # Save checkpoints at regular steps
    save_steps=500,  # Save every 500 steps
    save_total_limit=3,  # Keep last 3 checkpoints
    evaluation_strategy="steps",  # Evaluate periodically
    eval_steps=500,  # Evaluate every 500 steps
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",
    load_best_model_at_end=True,  # Load best model based on loss
    metric_for_best_model="loss",
    greater_is_better=False
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train the model
trainer.train()

# Save final fine-tuned model
final_model_path = "./finetuned_gpt2_embeddings"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Fine-tuned model saved to {final_model_path}")
