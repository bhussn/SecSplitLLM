import os
import pickle
from datasets import load_dataset
from transformers import BertTokenizerFast
import torch
from sklearn.utils import shuffle

# Configuration
model_name = "bert-base-uncased"
num_clients = 5
output_dir = "client_data"
os.makedirs(output_dir, exist_ok=True)

# Load SST-2 dataset
dataset = load_dataset("glue", "sst2")
train_data = dataset["train"]
val_data = dataset["validation"]

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# Tokenize function
def tokenize_batch(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128)

# Tokenize training and validation data
train_data = train_data.map(tokenize_batch, batched=True)
val_data = val_data.map(tokenize_batch, batched=True)

# Convert to PyTorch tensors
def convert_to_tensor_dict(dataset):
    return [
        {
            "input_ids": torch.tensor(example["input_ids"]),
            "attention_mask": torch.tensor(example["attention_mask"]),
            "labels": torch.tensor(example["label"]),
        }
        for example in dataset
    ]

# Shuffle and split training data into non-IID partitions
label_groups = {0: [], 1: []}
for example in train_data:
    label_groups[example["label"]].append(example)

# Distribute examples to clients in a label-skewed way
client_data = [[] for _ in range(num_clients)]
for label, examples in label_groups.items():
    examples = shuffle(examples, random_state=42)
    for i, example in enumerate(examples):
        client_id = i % num_clients
        if (label == 0 and client_id < 3) or (label == 1 and client_id >= 2):
            client_data[client_id].append(example)

# Save each client's data
for client_id, examples in enumerate(client_data):
    tensor_data = convert_to_tensor_dict(examples)
    with open(os.path.join(output_dir, f"client_{client_id}.pkl"), "wb") as f:
        pickle.dump(tensor_data, f)

# Save global validation data
val_tensor_data = convert_to_tensor_dict(val_data)
with open(os.path.join(output_dir, "global_val_data.pkl"), "wb") as f:
    pickle.dump(val_tensor_data, f)

print(f"Preprocessing complete. Data saved to '{output_dir}'")
