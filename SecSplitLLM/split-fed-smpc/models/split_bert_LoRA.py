import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from peft import LoraConfig, get_peft_model, TaskType

class BertSplitConfig:
    def __init__(self, model_name="bert-base-uncased", split_layer=5):
        self.model_name = model_name
        self.split_layer = split_layer
        self.config = BertConfig.from_pretrained(model_name)

class BertModel_Client(nn.Module):
    def __init__(self, split_config: BertSplitConfig):
        super(BertModel_Client, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        base_model = BertModel.from_pretrained(split_config.model_name, config=split_config.config)

        # LoRA configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],  # LoRA will apply to these in attention layers
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )

        # Apply LoRA
        self.bert = get_peft_model(base_model, lora_config)

        # Freeze everything except LoRA layers
        for param in self.bert.parameters():
            param.requires_grad = False
        for name, param in self.bert.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        self.embeddings = self.bert.embeddings
        self.encoder_layers = nn.ModuleList(self.bert.encoder.layer[:split_config.split_layer])

        hidden_size = split_config.config.hidden_size
        self.pool_proj = nn.Linear(hidden_size, 128)

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        x = self.dropout(x)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=x.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for layer in self.encoder_layers:
            x = layer(x, attention_mask=extended_attention_mask)[0]

        # Masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        pooled_output = self.pool_proj(mean_pooled)
        pooled_output = self.dropout(pooled_output)

        return pooled_output

class BertModel_Server(nn.Module):
    def __init__(self, split_config: BertSplitConfig):
        super(BertModel_Server, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(128, 2)  # binary classification

    def forward(self, pooled_activations):
        x = self.dropout(pooled_activations)
        return self.classifier(x)
