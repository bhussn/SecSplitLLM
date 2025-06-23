import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional

MODEL_NAME = "distilbert-base-cased"
NUM_LABELS = 9
MAX_LENGTH = 128
LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

def get_tokenizer():
    from transformers import DistilBertTokenizerFast
    return DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


class Net(torch.nn.Module):
    """NER model based on DistilBERT for token classification."""

    def __init__(self) -> None:
        """Initialize the DistilBERT token classification model."""
        super().__init__()
        from transformers import DistilBertForTokenClassification
        self.model = DistilBertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        """Forward pass through the model."""
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def to_device(self, device: torch.device) -> None:
        """Move the model to the specified device."""
        self.model.to(device)


class NERDataset(Dataset):
    """Custom dataset for NER tasks with tokenized inputs and aligned labels."""

    def __init__(self, encodings: Dict[str, Any], labels: list) -> None:
        """Initialize with tokenized encodings and corresponding labels."""
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single sample at index `idx`."""
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx]),
        }
