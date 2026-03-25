import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set


COMPUTE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Compute Device:", COMPUTE_DEVICE)
torch.backends.cudnn.benchmark = True


class TrainingConfiguration:
    """Centralized configuration for sequence generation models."""
    
    def __init__(self):
        self.rnn_hidden = 256
        self.lstm_hidden = 128
        self.batch_sz = 128
        self.rnn_epochs = 40
        self.lstm_epochs = 20
        self.attn_epochs = 40
        self.steps_per_ep = 200
        self.learning_rate = 0.002
        self.sampling_temp = 0.9
        self.output_path = "problem2_outputs"
        os.makedirs(self.output_path, exist_ok=True)


class NameDataset:
    """Manages vocabulary, encoding, and batch sampling for name sequences."""
    
    SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>"]
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_names = self._load_names()
        self.vocab = self._build_vocabulary()
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(self.vocab)
        self.pad_token_idx = self.char_to_idx["<PAD>"]
        self.encoded_sequences = self._preprocess_names()
    
    def _load_names(self) -> List[str]:
        """Load and normalize names from file."""
        with open(self.filepath, "r", encoding="utf-8") as f:
            return [line.strip().lower() for line in f if line.strip()]
    
    def _build_vocabulary(self) -> List[str]:
        """Construct character vocabulary with special tokens."""
        unique_chars = sorted(list(set("".join(self.raw_names))))
        return self.SPECIAL_TOKENS + unique_chars
    
    def _preprocess_names(self) -> List[List[int]]:
        """Convert names to token sequences."""
        tokenized = []
        for name in self.raw_names:
            tokens = [self.char_to_idx["<SOS>"]]
            tokens += [self.char_to_idx[c] for c in name]
            tokens += [self.char_to_idx["<EOS>"]]
            tokenized.append(tokens)
        return tokenized
    
    def sample_batch(self, batch_size: int) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate random batch of input-target pairs."""
        sampled = random.sample(self.encoded_sequences, batch_size)
        inputs = [seq[:-1] for seq in sampled]
        targets = [seq[1:] for seq in sampled]
        return inputs, targets
    
    def tensorize_sequences(self, sequences: List[List[int]]) -> torch.Tensor:
        """Pad sequences and convert to tensor."""
        max_length = max(len(s) for s in sequences)
        padded = torch.tensor(
            [s + [self.pad_token_idx] * (max_length - len(s)) for s in sequences],
            device=COMPUTE_DEVICE
        )
        return padded


class SimpleRNNSequence(nn.Module):
    """Single-layer RNN for sequence generation."""
    
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn_core = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        out, h = self.rnn_core(x, h)
        return self.output_proj(out), h


class BiLSTMSequence(nn.Module):
    """Bidirectional LSTM with dropout for sequence generation."""
    
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.drop = nn.Dropout(0.3)
        self.lstm_core = nn.LSTM(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.output_proj = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.drop(self.embedding(x))
        out, h = self.lstm_core(x, h)
        return self.output_proj(out), h


class RNNWithAttention(nn.Module):
    """RNN with self-attention mechanism for sequence generation."""
    
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn_core = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.attention_proj = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        out, h = self.rnn_core(x, h)
        
        attn_weights = torch.softmax(self.attention_proj(out), dim=1)
        attended = torch.sum(attn_weights * out, dim=1, keepdim=True)
        
        out = out + attended
        return self.output_proj(out), h


class SequenceGenerator:
    """Trains and generates sequences from neural models."""
    
    def __init__(self, model: nn.Module, dataset: NameDataset, config: TrainingConfiguration):
        self.model = model.to(COMPUTE_DEVICE)
        self.dataset = dataset
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_idx)
    
    def train_model(self, num_epochs: int, use_early_stop: bool = False) -> None:
        """Train model on generated batches."""
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for _ in range(self.config.steps_per_ep):
                # Sample batch and convert to tensors
                batch_inputs, batch_targets = self.dataset.sample_batch(self.config.batch_sz)
                X_tensor = self.dataset.tensorize_sequences(batch_inputs)
                Y_tensor = self.dataset.tensorize_sequences(batch_targets)
                
                predictions, _ = self.model(X_tensor)
                
                batch_loss = self.loss_fn(
                    predictions.view(-1, self.dataset.vocab_size),
                    Y_tensor.view(-1)
                )
                
                # Backpropagation with gradient clipping
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()
            
            avg_epoch_loss = epoch_loss / self.config.steps_per_ep
            print(f"Epoch {epoch+1}/{num_epochs} | Loss {avg_epoch_loss:.4f}")
            
            # Early stopping when loss plateaus
            if use_early_stop and avg_epoch_loss < 0.5:
                print("Early stopping activated")
                break
    
    def generate_sequence(self, max_length: int = 20) -> str:
        """Generate single name sequence."""
        self.model.eval()
        
        current_token = "<SOS>"
        generated_name = ""
        seen_chars: Set[str] = set()
        hidden_state = None
        
        for _ in range(max_length):
            token_idx = self.dataset.char_to_idx[current_token]
            x = torch.tensor([[token_idx]], device=COMPUTE_DEVICE)
            
            # Forward pass without gradient computation
            with torch.no_grad():
                logits, hidden_state = self.model(x, hidden_state)
            
            # Temperature scaling for controlled sampling
            token_distribution = torch.softmax(logits[0, -1] / self.config.sampling_temp, dim=0)
            
            # Apply repetition penalty
            for char in seen_chars:
                token_distribution[self.dataset.char_to_idx[char]] *= 0.6
            
            token_distribution = token_distribution / token_distribution.sum()
            
            sampled_idx = torch.multinomial(token_distribution, 1).item()
            current_token = self.dataset.idx_to_char[sampled_idx]
            
            if current_token == "<EOS>":
                break
            
            if current_token not in ["<PAD>", "<SOS>"]:
                generated_name += current_token
                seen_chars.add(current_token)
        
        return generated_name
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class GenerationEvaluator:
    """Evaluates quality of generated sequences."""
    
    @staticmethod
    def compute_metrics(generated_samples: List[str], training_data: Set[str]) -> Dict[str, float]:
        """Calculate novelty and diversity scores."""
        unique_generated = set(generated_samples)
        novel_count = len([n for n in generated_samples if n not in training_data])
        
        novelty_score = novel_count / len(generated_samples) if generated_samples else 0.0
        diversity_score = len(unique_generated) / len(generated_samples) if generated_samples else 0.0
        
        return {
            "novelty": novelty_score,
            "diversity": diversity_score
        }


def execute_generation_pipeline():
    """Main pipeline: load data, train models, evaluate, save results."""
    config = TrainingConfiguration()
    dataset = NameDataset("TrainingNames.txt")
    
    print(f"Loaded {len(dataset.raw_names)} names, vocab size: {dataset.vocab_size}\n")
    
    # Define models with their hyperparameters
    model_configs = {
        "RNN": (
            SimpleRNNSequence(dataset.vocab_size, config.rnn_hidden),
            config.rnn_epochs,
            False
        ),
        "BiLSTM": (
            BiLSTMSequence(dataset.vocab_size, config.lstm_hidden),
            config.lstm_epochs,
            True
        ),
        "Attention": (
            RNNWithAttention(dataset.vocab_size, config.rnn_hidden),
            config.attn_epochs,
            False
        )
    }
    
    evaluation_results = {}
    training_names = set(dataset.raw_names)
    
    # Train and evaluate each model
    for model_name, (model_arch, num_epochs, early_stop_flag) in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        generator = SequenceGenerator(model_arch, dataset, config)
        generator.train_model(num_epochs, use_early_stop=early_stop_flag)
        
        # Generate 1000 samples and compute evaluation metrics
        generated_samples = [generator.generate_sequence() for _ in range(1000)]
        metrics = GenerationEvaluator.compute_metrics(generated_samples, training_names)
        param_count = generator.count_parameters()
        
        evaluation_results[model_name] = {
            "novelty": metrics["novelty"],
            "diversity": metrics["diversity"],
            "params": param_count,
            "samples": generated_samples
        }
        
        print(f"{model_name} → Novelty {metrics['novelty']:.3f}, Diversity {metrics['diversity']:.3f}")
        
        # Save generated samples
        with open(f"{config.output_path}/{model_name}_samples.txt", "w") as f:
            for sample in generated_samples:
                f.write(sample + "\n")
    
    # Save results and visualizations
    metrics_file = f"{config.output_path}/metrics.txt"
    with open(metrics_file, "w") as f:
        for model_label, data in evaluation_results.items():
            f.write(f"{model_label}\n")
            f.write(f"Parameters: {data['params']}\n")
            f.write(f"Novelty: {data['novelty']:.4f}\n")
            f.write(f"Diversity: {data['diversity']:.4f}\n\n")
    
    # Generate comparison plots
    model_names_list = list(evaluation_results.keys())
    novelty_scores = [evaluation_results[m]["novelty"] for m in model_names_list]
    diversity_scores = [evaluation_results[m]["diversity"] for m in model_names_list]
    
    # Novelty comparison
    plt.figure(figsize=(8, 5))
    plt.bar(model_names_list, novelty_scores, color='steelblue')
    plt.title("Novelty Score Comparison", fontsize=14)
    plt.ylabel("Novelty")
    plt.savefig(f"{config.output_path}/novelty_comparison.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # Diversity comparison
    plt.figure(figsize=(8, 5))
    plt.bar(model_names_list, diversity_scores, color='darkorange')
    plt.title("Diversity Score Comparison", fontsize=14)
    plt.ylabel("Diversity")
    plt.savefig(f"{config.output_path}/diversity_comparison.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Pipeline completed. Results saved to {config.output_path}/")


if __name__ == "__main__":
    execute_generation_pipeline()
