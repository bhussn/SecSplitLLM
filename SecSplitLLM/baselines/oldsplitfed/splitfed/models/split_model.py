import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BertClient(nn.Module):
    def __init__(self, cut_layer: int = 4):
        # Client-side model: Embedding + encoder layers up to cut_layer.
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)

        self.cut_layer = cut_layer
        self.embeddings = self.bert.embeddings
        self.encoder_layers = nn.ModuleList(self.bert.encoder.layer[:cut_layer])

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, input_ids.size(), input_ids.device
        )

        for layer_module in self.encoder_layers:
            x = layer_module(x, extended_attention_mask)[0]
        return x


class BertServer(nn.Module):
    def __init__(self, cut_layer: int = 4):
        # Server-side model: Remaining encoder layers + pooler + classifier.
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)

        self.encoder_layers = nn.ModuleList(self.bert.encoder.layer[cut_layer:])
        self.pooler = self.bert.pooler
        self.classifier = nn.Linear(config.hidden_size, 2)  # Binary classification (SST-2)

    def forward(self, hidden_states, attention_mask, labels=None):
        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, attention_mask.size(), attention_mask.device
        )

        x = hidden_states
        for layer_module in self.encoder_layers:
            x = layer_module(x, extended_attention_mask)[0]

        pooled_output = self.pooler(x)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return logits, loss
