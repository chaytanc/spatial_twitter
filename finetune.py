import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from preprocess import clear_memory, generate_embeddings, pickle_data, load_pickled_dataset
from EmbeddingTextDataset import EmbeddingTextDataset
from eval import get_eval_dataset
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from PredictionLogger import PredictionLogger

with open("params.yaml", 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        config = None

# Constants
MAX_LENGTH = config["MAX_LENGTH"] 
EMBEDDING_DIM = config["EMBEDDING_DIM"]
EMBEDDING_FILE = "embeddings2.dat" # TODO do the other half of the embeddings
EMBEDDING_PKL_FILE = "synthetic_tweet_embeddings2.pkl"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Load model and tokenizer
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.train()
    return model, tokenizer


# Load dataset from CSV files
def load_text_data(cut_data_by):
    # TODO does this need to fuck with the encoding and utf8 shit?
    anti_brexit_df = pd.read_csv(config["ANTI_FILE"])
    pro_brexit_df = pd.read_csv(config["PRO_FILE"])

    anti_brexit_texts = anti_brexit_df["Hit Sentence"].dropna().tolist()
    pro_brexit_texts = pro_brexit_df["Hit Sentence"].dropna().tolist()

    # Trim datasets to balance classes
    # TODO reverse back to normal first half later?
    anti_brexit_texts = anti_brexit_texts[len(anti_brexit_texts) // cut_data_by :]
    pro_brexit_texts = pro_brexit_texts[len(pro_brexit_texts) // cut_data_by :]

    texts = anti_brexit_texts + pro_brexit_texts
    print(f"Loaded {len(texts)} tweets.")
    return texts, anti_brexit_texts, pro_brexit_texts


# Setup training arguments
def get_training_args():
    return TrainingArguments(
        output_dir="./gpt2_embedding_finetune2",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        learning_rate=5e-6,
        weight_decay=0.05,
        warmup_steps=1000,
    )

# TODO yaml this path
def save_fine_tuned_model(model, tokenizer, model_path="./finetuned_gpt2_embeddings2"):
    """Saves the fine-tuned model and tokenizer to the specified path."""
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Fine-tuned model saved to {model_path}")


def plot_training_progress(trainer, save_path="training_progress2.png"):
    """Plots training and evaluation loss over steps."""
    
    logs = trainer.state.log_history  # Extract log history from Trainer
    steps, train_losses, eval_steps, eval_losses = [], [], [], []

    for entry in logs:
        if "loss" in entry:  # Training loss
            steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:  # Evaluation loss
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    plt.figure(figsize=(10, 5))

    # Plot training loss
    plt.plot(steps, train_losses, label="Training Loss", color="blue", marker="o")

    # Plot evaluation loss (if available)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="Evaluation Loss", color="red", marker="x")

    # Graph settings
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training & Evaluation Loss Progress")
    plt.legend()
    plt.grid()

    # Save the plot
    plt.savefig(save_path)
    print(f"Loss graph saved to {save_path}")
    plt.show()

# Main function
def main():
    clear_memory()
    
    # Load model & tokenizer
    model, tokenizer = load_model()

    # Load raw data
    # texts, _, _ = load_text_data(cut_data_by=2)

    # Generate embeddings & save
    # mmap_file = generate_embeddings(texts)
    # dataset = EmbeddingTextDataset(mmap_file, texts, tokenizer)
    # pickle_data(dataset, filename=EMBEDDING_PKL_FILE)
    dataset = load_pickled_dataset(filename=EMBEDDING_PKL_FILE)
    eval_dataset = get_eval_dataset(tokenizer)
    prediction_logger = PredictionLogger(tokenizer, eval_dataset, eval_steps=100) 

    # Train model
    training_args = get_training_args()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        callbacks=[prediction_logger],
    )
    trainer.train()
    plot_training_progress(trainer)

    save_fine_tuned_model(model, tokenizer)
    


if __name__ == "__main__":
    main()
