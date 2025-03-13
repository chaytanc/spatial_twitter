from transformers import TrainerCallback
import torch
import torch.nn.functional as F
import yaml

def load_config(config_file="params.yaml"):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None
config = load_config()

class PredictionLogger(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, eval_steps):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Check if it's time to log predictions
        if state.global_step % self.eval_steps == 0:
            print(f"Logging predictions at step {state.global_step}...")
            
            # Evaluate the model on a batch from eval_dataset
            model = kwargs["model"]
            model.eval()
            
            i = state.global_step % len(self.eval_dataset)
            # Get predictions from a small batch
            if i + 2 > len(self.eval_dataset):
                i = 0
            batch = self.eval_dataset[i:i+2]  # You can modify this to use more samples
            # inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            labels = batch["labels"]
            batch["inputs_embeds"] = batch["inputs_embeds"].to(dtype=torch.float32, device=model.device).clone()

            # Generate predictions
            with torch.no_grad():
                outputs = model(inputs_embeds=batch["inputs_embeds"])
                logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

                # Apply temperature before sampling
                temperature = 1.0  # Adjust this (higher = more diverse, lower = more deterministic)
                probs = F.softmax(logits / temperature, dim=-1)

                # Sample from the distribution
                predictions = torch.multinomial(probs[:, -1, :], num_samples=config.get("MAX_LENGTH"))  # Last token

            # Decode predictions and labels
            decoded_predictions = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
            decoded_labels = self.tokenizer.decode(labels[0], skip_special_tokens=True)

            print(f"Step: {state.global_step}")
            print(f"Predictions: {decoded_predictions}")
            print(f"Labels: {decoded_labels}")