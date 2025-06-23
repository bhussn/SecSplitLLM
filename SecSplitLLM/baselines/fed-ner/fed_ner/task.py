from typing import List, Tuple
from collections import OrderedDict
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification
from seqeval.metrics import f1_score, precision_score, recall_score
from fed_ner.ner_model import NERDataset, MAX_LENGTH

BATCH_SIZE = 16

# Label mapping (should match dataset labels)
id2label = {
    0: "O", 1: "B-MISC", 2: "I-MISC",
    3: "B-PER", 4: "I-PER",
    5: "B-ORG", 6: "I-ORG",
    7: "B-LOC", 8: "I-LOC"
}

tokenizer = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        from fed_ner.ner_model import get_tokenizer as ner_get_tokenizer
        tokenizer = ner_get_tokenizer()
    return tokenizer

def tokenize_and_align_labels(examples):
    tokenizer = get_tokenizer()

    tokenized_inputs = tokenizer(
        examples["tokens"], is_split_into_words=True,
        truncation=True, padding="max_length", max_length=MAX_LENGTH
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
        # Pad/truncate labels to MAX_LENGTH
        label_ids = label_ids[:MAX_LENGTH] + [-100] * (MAX_LENGTH - len(label_ids))
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def load_data(partition_id: int, num_partitions: int) -> Tuple[DataLoader, DataLoader]:
    tokenizer = get_tokenizer()

    dataset = load_dataset("conll2003")
    full_train = dataset["train"].shuffle(seed=42)

    total = len(full_train)
    part_size = total // num_partitions
    start = partition_id * part_size
    end = total if partition_id == num_partitions - 1 else start + part_size
    part_data = full_train.select(range(start, end))
    tokenized = part_data.map(tokenize_and_align_labels, batched=True)

    train_len = int(0.8 * len(tokenized))
    train_data = tokenized.select(range(train_len))
    val_data = tokenized.select(range(train_len, len(tokenized)))

    collator = DataCollatorForTokenClassification(tokenizer)
    train_dataset = NERDataset(train_data, train_data["labels"])
    val_dataset = NERDataset(val_data, val_data["labels"])

    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator),
        DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collator),
    )

def train(net: torch.nn.Module, dataloader: DataLoader, epochs: int, device: torch.device) -> float:
    net.train().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-5)
    total_loss = 0.0

    for _ in range(epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = net(input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

    return total_loss / (epochs * len(dataloader))

def test(net, dataloader, device):
    """
    Centralized evaluation using seqeval metrics for NER: Precision, Recall, F1.
    """
    net.eval().to(device)
    all_preds, all_labels = [], []
    loss_total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = net(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            for pred, label in zip(predictions, labels):
                true_labels = []
                pred_labels = []

                for p, l in zip(pred.cpu().numpy(), label.cpu().numpy()):
                    if l != -100:
                        true_labels.append(id2label[l])
                        pred_labels.append(id2label[p])

                all_labels.append(true_labels)
                all_preds.append(pred_labels)

            loss_total += outputs.loss.item()

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return loss_total / len(dataloader), precision, recall, f1

def get_weights(net: torch.nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for val in net.state_dict().values()]

def set_weights(net: torch.nn.Module, weights: List[np.ndarray]) -> None:
    keys = net.state_dict().keys()
    net.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in zip(keys, weights)}))
