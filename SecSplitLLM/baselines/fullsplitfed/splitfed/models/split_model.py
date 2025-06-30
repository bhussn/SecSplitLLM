import torch
import torch.nn as nn
import warnings
from transformers import BertModel, BertConfig

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")


def _get_extended_attention_mask(attention_mask: torch.Tensor, input_shape, device) -> torch.Tensor:
    # Create extended attention mask for BERT from a 2D or 3D attention_mask tensor.
    if attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    elif attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    else:
        raise ValueError(f"Invalid attention_mask dimension {attention_mask.dim()}, expected 2 or 3.")

    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp32 for compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class BertClient(nn.Module):
    def __init__(self, cut_layer: int = 4):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-uncased")
        base_model = BertModel.from_pretrained("bert-base-uncased", config=config)

        self.embeddings = base_model.embeddings
        self.encoder_layers = nn.ModuleList(base_model.encoder.layer[:cut_layer])
        self.config = config

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        extended_attention_mask = _get_extended_attention_mask(
            attention_mask, input_ids.size(), input_ids.device
        )
        for layer in self.encoder_layers:
            x = layer(x, extended_attention_mask)[0]
        return x


class BertServer(nn.Module):
    def __init__(self, cut_layer: int = 4):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-uncased")
        base_model = BertModel.from_pretrained("bert-base-uncased", config=config)

        self.encoder_layers = nn.ModuleList(base_model.encoder.layer[cut_layer:])
        self.pooler = base_model.pooler  # may be None
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        extended_attention_mask = _get_extended_attention_mask(
            attention_mask, attention_mask.shape, attention_mask.device
        )
        x = hidden_states
        for layer in self.encoder_layers:
            x = layer(x, extended_attention_mask)[0]

        if self.pooler is not None:
            pooled_output = self.pooler(x)
        else:
            pooled_output = x[:, 0]  # fallback to CLS token

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, 2), labels.view(-1))

        return logits, loss
