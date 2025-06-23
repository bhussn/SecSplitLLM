import torch
from transformers import BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from splitfed.utils import Timer
from splitfed.metrics import append_timing, log_timing_wandb

MAX_LEN = 128
MODEL_NAME = "bert-base-uncased"

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

def load_data(partition_id, num_partitions, split="train"):
    dataset = load_dataset("glue", "sst2")[split]
    total = len(dataset)
    shard_size = total // num_partitions
    start = partition_id * shard_size
    end = start + shard_size if partition_id < num_partitions - 1 else total

    subset = dataset.select(range(start, end))
    encodings = tokenizer(subset["sentence"], truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    labels = torch.tensor(subset["label"])

    data = SST2Dataset(encodings, labels)
    return DataLoader(data, batch_size=16, shuffle=True)

def train(client_model, server_model, dataloader, device, round_num=None):
    client_model.train()
    server_model.train()

    optimizer = torch.optim.AdamW(
        list(client_model.parameters()) + list(server_model.parameters()), lr=2e-5
    )

    total_forward_time = 0.0
    total_backward_time = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with Timer() as fwd_timer:
            hidden = client_model(input_ids, attention_mask)
        total_forward_time += fwd_timer.interval

        logits, loss = server_model(hidden, attention_mask, labels)

        optimizer.zero_grad()
        with Timer() as bwd_timer:
            loss.backward()
        total_backward_time += bwd_timer.interval

        optimizer.step()

    avg_fwd = total_forward_time / len(dataloader)
    avg_bwd = total_backward_time / len(dataloader)

    print(f"[Timing] Client forward avg: {avg_fwd:.2f} ms | Server backward avg: {avg_bwd:.2f} ms")

    if round_num is not None:
        append_timing(round_num, avg_fwd, avg_bwd)
        log_timing_wandb(round_num, avg_fwd, avg_bwd)

    return loss.item(), avg_fwd, avg_bwd


def test(client_model, server_model, dataloader, device):
    client_model.eval()
    server_model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

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

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, 0.0

def get_weights(model):
    state_dict = model.state_dict()
    weights = {}
    for name, param in state_dict.items():
        weights[name] = param.cpu().numpy()
    return weights
