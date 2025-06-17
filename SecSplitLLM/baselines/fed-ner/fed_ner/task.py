"""fed-ner: A Flower / PyTorch app."""

from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast, DataCollatorForTokenClassification 
from datasets import load_dataset
import evaluate
from collections import OrderedDict
from sklearn.metrics import f1_score


# Constants
MODEL_NAME = "distilbert-base-cased"
NUM_LABELS = 9  # CoNLL-2003 has 9 labels including O and special tokens
MAX_LENGTH = 128
BATCH_SIZE = 16

# Load tokenizer once globally
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

# Map CoNLL labels to IDs
LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

# Load evaluation metric
metric = evaluate.load("seqeval")


class NERDataset(Dataset):
    """PyTorch dataset for NER tasks using tokenized encodings"""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels  # list of lists (token-level labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx])
        }
        return item


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        # Ensure exact length
        if len(label_ids) < MAX_LENGTH:
            label_ids += [-100] * (MAX_LENGTH - len(label_ids))
        else:
            label_ids = label_ids[:MAX_LENGTH]

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def load_data(partition_id: int, num_partitions: int) -> Tuple[DataLoader, DataLoader]:
    dataset = load_dataset("conll2003")
    full_train = dataset["train"].shuffle(seed=42)

    total_size = len(full_train)
    partition_size = total_size // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else total_size

    partition_data = full_train.select(range(start_idx, end_idx))
    tokenized_partition = partition_data.map(tokenize_and_align_labels, batched=True)

    train_size = int(0.8 * len(tokenized_partition))
    train_split = tokenized_partition.select(range(train_size))
    val_split = tokenized_partition.select(range(train_size, len(tokenized_partition)))

    train_encodings = {k: train_split[k] for k in train_split.features.keys() if k != 'labels'}
    train_labels = train_split['labels']

    val_encodings = {k: val_split[k] for k in val_split.features.keys() if k != 'labels'}
    val_labels = val_split['labels']

    train_dataset = NERDataset(train_encodings, train_labels)
    val_dataset = NERDataset(val_encodings, val_labels)

    # Use dynamic padding to avoid tensor size mismatch
    data_collator = DataCollatorForTokenClassification(tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

    return train_loader, val_loader


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DistilBertForTokenClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)


def train(net: torch.nn.Module, dataloader: DataLoader, epochs: int, device: torch.device) -> float:
    net.train()
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-5)
    total_loss = 0

    for _ in range(epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = net(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    avg_loss = total_loss / (epochs * len(dataloader))
    return avg_loss


def test(net, dataloader, device):
    net.eval()
    net.to(device)
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = net(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100
            filtered_preds = preds[mask].cpu().numpy()
            filtered_labels = labels[mask].cpu().numpy()

            all_preds.extend(filtered_preds)
            all_labels.extend(filtered_labels)

    avg_loss = total_loss / len(dataloader)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, accuracy, f1


def get_weights(net: torch.nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
