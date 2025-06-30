import csv
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from datasets import load_dataset
from time import time
from sklearn.metrics import f1_score
from splitfed.training.utils import append_timing

# Constants
MAX_LEN = 128
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 4
TIMING_CSV_PATH = "timing_log.csv"

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


class SST2Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_data(partition_id: int, num_partitions: int, split: str = "train") -> DataLoader:
    dataset = load_dataset("glue", "sst2")
    subset = dataset[split].shard(num_shards=num_partitions, index=partition_id)
    encodings = tokenizer(
        subset["sentence"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    labels = torch.tensor(subset["label"])

    return DataLoader(SST2Dataset(encodings, labels), batch_size=BATCH_SIZE, shuffle=True)


def train(client_model, server_model, dataloader, device, round_num=None):
    client_model.train()
    server_model.train()

    optimizer = torch.optim.AdamW(
        list(client_model.parameters()) + list(server_model.parameters()),
        lr=2e-5
    )

    total_fwd_time = 0.0
    total_bwd_time = 0.0
    last_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with Timer() as fwd_timer:
            hidden = client_model(input_ids, attention_mask)
        total_fwd_time += fwd_timer.interval

        logits, loss = server_model(hidden, attention_mask, labels)
        last_loss = loss.item()

        with Timer() as bwd_timer:
            loss.backward()
        total_bwd_time += bwd_timer.interval

        optimizer.step()

    avg_fwd = total_fwd_time / len(dataloader)
    avg_bwd = total_bwd_time / len(dataloader)

    if round_num is not None:
        append_timing(round_num, avg_fwd, avg_bwd, TIMING_CSV_PATH)

    return last_loss, avg_fwd, avg_bwd


def test(client_model, server_model, dataloader, device):
    client_model.eval()
    server_model.eval()

    correct = 0
    total = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            hidden = client_model(input_ids, attention_mask)
            logits, loss = server_model(hidden, attention_mask, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)

    f1 = f1_score(all_labels, all_preds, average="binary")

    return avg_loss, accuracy, f1


def get_weights(model):
    return {name: param.cpu().numpy() for name, param in model.state_dict().items()}


# Timer utility
class Timer:
    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.interval = (self.end - self.start) * 1000  # milliseconds



