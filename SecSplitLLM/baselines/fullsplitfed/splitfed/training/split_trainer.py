import csv
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from datasets import load_dataset
from time import time
from sklearn.metrics import f1_score
from splitfed.training.utils import append_timing
from splitfed.grpc import split_pb2

# Constants
MAX_LEN = 128
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
TIMING_CSV_PATH = "timing_log.csv"

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, cache_dir="splitfed/hf_cache")


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
    dataset = load_dataset("glue", "sst2", cache_dir="./hf_cache")
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


def train(client_model, dataloader, device, grpc_stub, round_num=None):
    client_model.train()
    optimizer = torch.optim.AdamW(client_model.parameters(), lr=2e-5)

    total_fwd_time = 0.0
    total_bwd_time = 0.0
    last_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with Timer() as fwd_timer:
            activation = client_model(input_ids, attention_mask)
        total_fwd_time += fwd_timer.interval

        # Prepare gRPC request to server
        request = split_pb2.ForwardRequest(
            activation=activation.detach().cpu().view(-1).tolist(),
            activation_shape=list(activation.shape),
            attention_mask=attention_mask.cpu().view(-1).tolist(),
            attention_mask_shape=list(attention_mask.shape),
            client_id=0,
            round=round_num or 0,
            batch_id=batch_idx
        )

        # gRPC call to run forward + backward server-side
        response = grpc_stub.ForwardPass(request)

        last_loss = response.loss

        # Receive logits for potential local metrics, and backprop through cut layer
        logits = torch.tensor(response.logits, dtype=torch.float32).reshape(response.logits_shape).to(device)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        with Timer() as bwd_timer:
            loss.backward()
        total_bwd_time += bwd_timer.interval

        optimizer.step()

    avg_fwd = total_fwd_time / len(dataloader)
    avg_bwd = total_bwd_time / len(dataloader)

    if round_num is not None:
        append_timing(round_num, avg_fwd, avg_bwd, "timing_log.csv")

    return last_loss, avg_fwd, avg_bwd


def test(client_model, dataloader, device, grpc_stub):
    client_model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            activation = client_model(input_ids, attention_mask)

            request = split_pb2.InferenceRequest(
                activation=activation.cpu().view(-1).tolist(),
                activation_shape=list(activation.shape),
                attention_mask=attention_mask.cpu().view(-1).tolist(),
                attention_mask_shape=list(attention_mask.shape),
                client_id=0,
                batch_id=batch_idx
            )

            response = grpc_stub.Inference(request)

            logits = torch.tensor(response.logits, dtype=torch.float32).reshape(response.logits_shape).to(device)
            loss = criterion(logits, labels)

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



