# import os
# import pickle
# from datasets import load_dataset
# from transformers import BertTokenizerFast
# import torch
# from sklearn.utils import shuffle
# from collections import Counter

# # Configuration
# model_name = "bert-base-uncased"
# num_clients = 2
# output_dir = "client_data"
# os.makedirs(output_dir, exist_ok=True)

# # Load MNLI dataset
# dataset = load_dataset("glue", "mnli")
# train_data = dataset["train"]
# val_data = dataset["validation_matched"]  # Use matched validation set as global val

# # Load tokenizer
# tokenizer = BertTokenizerFast.from_pretrained(model_name)

# # Tokenize function
# def tokenize_batch(batch):
#     return tokenizer(batch["premise"], batch["hypothesis"], padding="max_length", truncation=True, max_length=128)

# # Tokenize training and validation data
# train_data = train_data.map(tokenize_batch, batched=True)
# val_data = val_data.map(tokenize_batch, batched=True)

# # Filter out samples with invalid labels (-1)
# train_data = train_data.filter(lambda x: x["label"] != -1)
# val_data = val_data.filter(lambda x: x["label"] != -1)

# # Convert to PyTorch tensors
# def convert_to_tensor_dict(dataset):
#     return [
#         {
#             "input_ids": torch.tensor(example["input_ids"]),
#             "attention_mask": torch.tensor(example["attention_mask"]),
#             "labels": torch.tensor(example["label"]),
#         }
#         for example in dataset
#     ]

# # Shuffle and split training data into non-IID partitions by label
# label_groups = {0: [], 1: [], 2: []}  # MNLI labels: 0 = entailment, 1 = neutral, 2 = contradiction
# for example in train_data:
#     label_groups[example["label"]].append(example)

# # Distribute examples to clients in a label-skewed way
# client_data = [[] for _ in range(num_clients)]
# for label, examples in label_groups.items():
#     examples = shuffle(examples, random_state=42)
#     for i, example in enumerate(examples):
#         client_id = i % num_clients
#         # Example skew: first 3 clients get mostly entailment and neutral, last 2 get mostly contradiction
#         if (label in [0, 1] and client_id < 3) or (label == 2 and client_id >= 3):
#             client_data[client_id].append(example)

# # Save each client's data with new filename pattern and debug label counts
# for client_id, examples in enumerate(client_data):
#     label_list = [example["label"] for example in examples]
#     label_counts = Counter(label_list)
#     print(f"[DEBUG] Client {client_id} label distribution: {label_counts}")

#     tensor_data = convert_to_tensor_dict(examples)
#     with open(os.path.join(output_dir, f"client_mnli_{client_id}.pkl"), "wb") as f:
#         pickle.dump(tensor_data, f)

# # Debug print label distribution in validation set
# val_labels = [example["label"] for example in val_data]
# print(f"[DEBUG] Global validation label distribution: {Counter(val_labels)}")

# # Save global validation data
# val_tensor_data = convert_to_tensor_dict(val_data)
# with open(os.path.join(output_dir, "global_val_data_mnli.pkl"), "wb") as f:
#     pickle.dump(val_tensor_data, f)

# print(f"MNLI preprocessing complete. Data saved to '{output_dir}'")

# on iid
import os
import pickle
from datasets import load_dataset
from transformers import BertTokenizerFast
import torch
from sklearn.utils import shuffle
from collections import Counter

# Configuration
model_name = "bert-base-uncased"
num_clients = 5
output_dir = "client_data"
os.makedirs(output_dir, exist_ok=True)

# Load MNLI dataset
dataset = load_dataset("glue", "mnli")
train_data = dataset["train"]
val_data = dataset["validation_matched"]  # Use matched validation set as global val

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# Tokenize function
def tokenize_batch(batch):
    return tokenizer(batch["premise"], batch["hypothesis"], padding="max_length", truncation=True, max_length=128)

# Tokenize training and validation data
train_data = train_data.map(tokenize_batch, batched=True)
val_data = val_data.map(tokenize_batch, batched=True)

# Filter out samples with invalid labels (-1)
train_data = train_data.filter(lambda x: x["label"] != -1)
val_data = val_data.filter(lambda x: x["label"] != -1)

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

# Group examples by label
label_groups = {0: [], 1: [], 2: []}  # MNLI labels: 0 = entailment, 1 = neutral, 2 = contradiction
for example in train_data:
    label_groups[example["label"]].append(example)

# Initialize empty list for each client
client_data = [[] for _ in range(num_clients)]

# Balanced split: distribute examples equally across clients for each label
for label, examples in label_groups.items():
    examples = shuffle(examples, random_state=42)
    for i, example in enumerate(examples):
        client_id = i % num_clients
        client_data[client_id].append(example)

# Save each client's data and print label distribution for debugging
for client_id, examples in enumerate(client_data):
    label_list = [example["label"] for example in examples]
    label_counts = Counter(label_list)
    print(f"[DEBUG] Client {client_id} label distribution: {label_counts}")

    tensor_data = convert_to_tensor_dict(examples)
    with open(os.path.join(output_dir, f"client_mnli_{client_id}.pkl"), "wb") as f:
        pickle.dump(tensor_data, f)

# Print global validation label distribution
val_labels = [example["label"] for example in val_data]
print(f"[DEBUG] Global validation label distribution: {Counter(val_labels)}")

# Save global validation data
val_tensor_data = convert_to_tensor_dict(val_data)
with open(os.path.join(output_dir, "global_val_data_mnli.pkl"), "wb") as f:
    pickle.dump(val_tensor_data, f)

print(f"MNLI preprocessing complete. Data saved to '{output_dir}'")
