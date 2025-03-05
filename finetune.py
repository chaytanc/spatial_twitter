import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from preprocess import embed
import pandas as pd
import pickle
from tqdm import tqdm
import gc

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

# Custom dataset class
class EmbeddingTextDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, texts, tokenizer):
        self.embeddings = embeddings  # Precomputed input embeddings (tensor shape: [num_samples, seq_len, hidden_dim])
        self.texts = texts  # Corresponding text from tweets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        input_embedding = self.embeddings[idx]  # Extract precomputed embedding
        text = self.texts[idx]
        # TODO max length 
        tokenized_output = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)

        return {
            "inputs_embeds": input_embedding.squeeze(0),
            "labels": tokenized_output["input_ids"].squeeze(0)
        }
torch.mps.empty_cache()  # Try clearing MPS-specific memory
gc.collect()

# Load both datasets
anti_brexit_df = pd.read_csv("dataverse_files/TweetDataset_AntiBrexit_Jan-Mar2022.csv")
pro_brexit_df = pd.read_csv("dataverse_files/TweetDataset_ProBrexit_Jan-Mar2022.csv")

# Extract the relevant text column
anti_brexit_texts = anti_brexit_df["Hit Sentence"].dropna().tolist()
pro_brexit_texts = pro_brexit_df["Hit Sentence"].dropna().tolist()

# Combine both datasets
texts = anti_brexit_texts + pro_brexit_texts

texts = texts[:80000]
print(f"Loaded {len(texts)} tweets.")

# Generate embeddings
# embeddings = [embed(text) for text in tqdm(texts, desc="Generating Embeddings", leave=False)]
embeddings = list(map(embed, tqdm(texts, desc="Generating Embeddings", leave=False)))
# embeddings = list(map(embed, texts))


# Create dataset
dataset = EmbeddingTextDataset(embeddings, texts, tokenizer)
# def pickle_data(embeddings, texts):
def pickle_data(dataset):
    # dataset = {
    #     "embeddings": embeddings,
    #     "texts": texts
    # }

    # Save as a pickle file
    with open("synthetic_tweet_embeddings.pkl", "wb") as f:
        pickle.dump(dataset, f)

    print("Saved dataset to synthetic_tweet_embeddings.pkl")

pickle_data(dataset)


# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_embedding_finetune",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    report_to="none"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train the model
trainer.train()

# Save fine-tuned model
model.save_pretrained("./finetuned_gpt2_embeddings")
tokenizer.save_pretrained("./finetuned_gpt2_embeddings")


