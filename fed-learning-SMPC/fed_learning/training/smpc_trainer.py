import torch
import evaluate
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from typing import List, Tuple, Dict
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import crypten
import os
import pickle
from pathlib import Path

# --- Constants ---
MODEL_NAME = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4  # AG News has 4 classes

# Load tokenizer (global to avoid reloading)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# AG News label mappings
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
label2id = {v: k for k, v in id2label.items()}

# Cache for tokenized datasets
DATASET_CACHE = {}

# --- 1. LoRA Model Configuration ---
def load_lora_model() -> AutoModelForSequenceClassification:
    """Loads the base model and applies LoRA configuration."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id
    )

    # Configure LoRA for sequence classification
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=2,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_lin", "v_lin"],  # DistilBERT-specific
    )

    # Make the model parameter-efficient
    peft_model = get_peft_model(model, lora_config)
    print("\n--- LoRA Model Configuration ---")
    peft_model.print_trainable_parameters()
    print("---------------------------------\n")
    return peft_model.to(DEVICE)

# Define a global cache path
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "ag_news_tokenized.pkl"

# --- 2. Data Loading ---
def load_data(node_id: int, num_partitions: int):
    """Load, partition, and tokenize AG News data efficiently with disk caching."""
    # Use client-specific cache to avoid conflicts
    client_cache = CACHE_DIR / f"ag_news_tokenized_{node_id}.pkl"
    
    # Try to load from disk cache first
    if client_cache.exists():
        print(f"[Client {node_id}] Loading tokenized dataset from cache")
        with open(client_cache, "rb") as f:
            tokenized_dataset = pickle.load(f)
    else:
        print(f"[Client {node_id}] Tokenizing dataset and creating cache")
        dataset = load_dataset("ag_news")
        
        # Tokenization function with optimization
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
        
        # Tokenize with progress bar disabled for efficiency
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            batch_size=256,  # Increased batch size
            load_from_cache_file=False,
            desc=f"Tokenizing (Client {node_id})"
        )
        
        # Rename label column
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )
        
        # Save to disk cache
        with open(client_cache, "wb") as f:
            pickle.dump(tokenized_dataset, f)
    
    # Partition dataset
    partition_index = node_id % num_partitions
    train_data = tokenized_dataset["train"].shard(
        num_shards=num_partitions, 
        index=partition_index
    )
    test_data = tokenized_dataset["test"]
    
    # Use larger batches where possible
    trainloader = DataLoader(
        train_data, 
        batch_size=32, 
        shuffle=True,
        num_workers=2,  # Parallel loading
        pin_memory=True  # Faster GPU transfer
    )
    testloader = DataLoader(
        test_data, 
        batch_size=128,  # Larger batch for evaluation
        num_workers=2,
        pin_memory=True
    )
    
    return trainloader, testloader

# --- 3. Training and Evaluation ---
def train_and_metrics(net, trainloader, epochs) -> Tuple[float, float]:
    """Train model and return loss/accuracy metrics"""
    optimizer = torch.optim.AdamW(
        net.parameters(), 
        lr=2e-5,
        weight_decay=0.01  # Regularization
    )
    net.train()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for epoch in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Calculate accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == batch["labels"]).sum().item()
            total_samples += batch["labels"].size(0)
            total_loss += loss.item()
    
    avg_loss = total_loss / (len(trainloader) * epochs)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def test(net, testloader) -> Tuple[float, Dict]:
    """Evaluate the model on the test set."""
    net.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in testloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            total_loss += outputs.loss.item()
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    # Calculate classification metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    
    avg_loss = total_loss / len(testloader)
    return avg_loss, {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

# --- 4. LoRA Weight and SMPC Helper Functions ---
def get_lora_weights(net: torch.nn.Module) -> List[np.ndarray]:
    """Get only trainable LoRA weights in float32"""
    return [
        p.cpu().detach().numpy().astype(np.float32)
        for name, p in net.named_parameters() 
        if "lora" in name and p.requires_grad
    ]

def set_lora_weights(net: torch.nn.Module, parameters: List[np.ndarray]):
    """Set LoRA adapter weights from a list of NumPy arrays."""
    lora_params = [
        name for name, p in net.named_parameters() 
        if "lora" in name and p.requires_grad
    ]
    state_dict = OrderedDict({
        name: torch.tensor(param).to(DEVICE)
        for name, param in zip(lora_params, parameters)
    })
    net.load_state_dict(state_dict, strict=False)

# --- SMPC-SPECIFIC FUNCTIONS ---
def secret_share_weights(weights: List[np.ndarray]) -> List:
    """Convert weights to CrypTensors without quantization"""
    return [crypten.cryptensor(torch.tensor(w).float().to(DEVICE)) for w in weights]

def reconstruct_weights(encrypted_weights: List) -> List[np.ndarray]:
    """Convert encrypted weights to plaintext with validation"""
    plaintext = []
    for tensor in encrypted_weights:
        try:
            pt = tensor.get_plain_text().cpu().numpy()  # Move to CPU first
            plaintext.append(pt)
        except RuntimeError as e:
            print(f"Reconstruction error: {e}")
            # Add fallback to zeros
            plaintext.append(np.zeros(tensor.size(), dtype=np.float32))
    return plaintext

# --- 5. Weight Transformation Utilities ---
def numpy_to_tensors(weights: List[np.ndarray]) -> List[torch.Tensor]:
    """Convert numpy arrays to PyTorch tensors (for SMPC operations)"""
    return [torch.tensor(w).float() for w in weights]

def tensors_to_numpy(tensors: List[torch.Tensor]) -> List[np.ndarray]:
    """Convert PyTorch tensors to numpy arrays (for serialization)"""
    return [t.numpy() for t in tensors]

if __name__ == "__main__":
    model = load_lora_model()
    trainloader, testloader = load_data(0, 3)
    
    # Test training
    loss, acc = train_and_metrics(model, trainloader, epochs=1)
    print(f"Training completed with loss={loss:.4f}, accuracy={acc:.4f}")
    
    # Test evaluation
    loss, metrics = test(model, testloader)
    print(f"Evaluation completed with loss={loss:.4f}, accuracy={metrics['accuracy']:.4f}")
    
    # Test SMPC functions
    weights = get_lora_weights(model)
    encrypted = secret_share_weights(weights)
    decrypted = reconstruct_weights(encrypted)
    
    # Verify reconstruction
    for orig, dec in zip(weights, decrypted):
        if not np.allclose(orig, dec, atol=1e-4):
            print("Warning: Reconstruction mismatch!")
            break
    else:
        print("SMPC functions verified successfully")
