import torch
import crypten
import evaluate
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from flwr.common import Parameters
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from typing import List, Tuple, Dict
from collections import OrderedDict

# --- Constants and Initialization ---
MODEL_NAME = "distilbert-base-cased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize CrypTen
crypten.init()

# Load tokenizer, metric, and labels
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
metric = evaluate.load("seqeval")
label_list = load_dataset("conll2003", split="train").features["ner_tags"].feature.names
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {v: k for k, v in id2label.items()}


# --- 1. LoRA Model Configuration ---
def load_lora_model() -> AutoModelForTokenClassification:
    """Loads the base model and applies LoRA configuration."""
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    # Configure LoRA to adapt the query and value matrices in the attention layers
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )

    # Make the model parameter-efficient
    peft_model = get_peft_model(model, lora_config)
    print("\n--- LoRA Model Configuration ---")
    peft_model.print_trainable_parameters()
    print("---------------------------------\n")
    return peft_model.to(DEVICE)


# --- 2. Data Loading (same as original trainer) ---
def load_data(node_id: int, num_partitions: int):
    """Load, partition, and tokenize data."""
    dataset = load_dataset("conll2003")

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None or word_idx == previous_word_idx:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    tokenized_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Use the modulo operator to ensure a valid partition index
    partition_index = node_id % num_partitions
    
    # Partition the dataset for the specific client
    train_dataset = tokenized_dataset["train"].shard(
        num_shards=num_partitions, 
        index=partition_index
    )
    test_dataset = tokenized_dataset["test"]

    trainloader = DataLoader(
        train_dataset, batch_size=8, collate_fn=data_collator, shuffle=True
    )
    testloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

    return trainloader, testloader


# --- 3. Training and Evaluation ---
def train_and_metrics(net, trainloader, epochs) -> Tuple[float, float]:
    """Train model and return loss/accuracy metrics"""
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-4)
    net.train()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.logits, 2)
            
            # Mask out padding tokens (label = -100)
            mask = (batch['labels'] != -100)
            active_preds = predicted[mask]
            active_labels = batch['labels'][mask]
            
            if active_labels.numel() > 0:  # Only calculate if there are valid labels
                total_correct += (active_preds == active_labels).sum().item()
                total_samples += active_labels.numel()
            
            total_loss += loss.item()
    
    avg_loss = total_loss / (len(trainloader) * epochs)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def test(net, testloader) -> Tuple[float, Dict]:
    """Evaluate the model with LoRA adapters on the test set."""
    net.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    try:
        with torch.no_grad():
            for batch in testloader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = net(**batch)
                total_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=2)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
    except ValueError as e:
        print("\n--- DEBUGGING: CAUGHT VALUE ERROR ---")
        print(f"ERROR: {e}")
        print("Inspecting the first batch that caused the error:")
        try:
            for i, batch in enumerate(testloader):
                if i > 0: break
                print(f"Batch {i} keys: {batch.keys()}")
                for key, value in batch.items():
                    print(f"  - Key: '{key}', Type: {type(value)}")
                    if hasattr(value, 'shape'):
                        print(f"    Shape: {value.shape}")
                    # Print a small sample of the data
                    print(f"    Sample Value: {value[0]}")
        except Exception as inspect_e:
            print(f"Could not inspect batch. Reason: {inspect_e}")
        print("---------------------------------------\n")
        raise e # Re-raise the original error

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(all_predictions, all_labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(all_predictions, all_labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return total_loss / len(testloader), {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# --- 4. LoRA Weight and SMPC Helper Functions ---
def get_lora_weights(net: torch.nn.Module) -> List[np.ndarray]:
    return [p.cpu().detach().numpy().astype(np.float32)  # CrypTen requires float32
            for name, p in net.named_parameters() if "lora" in name]


def set_lora_weights(net: torch.nn.Module, parameters: List[np.ndarray]):
    """Set LoRA adapter weights from a list of NumPy arrays."""
    lora_params = [
        (name, p) for name, p in net.named_parameters() if "lora" in name
    ]
    params_dict = zip([name for name, _ in lora_params], parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=False)


def secret_share_weights(weights: List[np.ndarray]) -> List[torch.Tensor]:
    """Convert weights to CrypTensors with automatic secret sharing"""
    return [crypten.cryptensor(torch.tensor(w)) for w in weights]

if __name__ == "__main__":
    model = load_lora_model()
    trainloader, _ = load_data(0, 3)
    loss, acc = train_and_metrics(model, trainloader, epochs=1)
    print(f"Training completed with loss={loss:.4f}, accuracy={acc:.4f}")
