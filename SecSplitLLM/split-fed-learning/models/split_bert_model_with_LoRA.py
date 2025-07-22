import torch.nn as nn
from transformers import BertModel, BertConfig
from peft import LoraConfig, get_peft_model, TaskType

class BertSplitConfig:
    def __init__(self, model_name="bert-base-uncased", split_layer=4):
        self.model_name = model_name
        self.split_layer = split_layer
        self.config = BertConfig.from_pretrained(model_name)

class BertModel_Client(nn.Module):
    def __init__(self, split_config: BertSplitConfig):
        super(BertModel_Client, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        # Load and wrap full BERT with LoRA
        base_model = BertModel.from_pretrained(split_config.model_name)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.bert = get_peft_model(base_model, lora_config)

        self.embeddings = self.bert.embeddings
        self.encoder_layers = nn.ModuleList(self.bert.encoder.layer[:split_config.split_layer])

        # Freeze everything except LoRA
        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.bert.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        x = self.dropout(x)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=x.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for layer in self.encoder_layers:
            x = layer(x, attention_mask=extended_attention_mask)[0]

        return x, attention_mask

class BertModel_Server(nn.Module):
    def __init__(self, split_config: BertSplitConfig):
        super(BertModel_Server, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.bert = BertModel.from_pretrained(split_config.model_name)
        self.encoder_layers = nn.ModuleList(self.bert.encoder.layer[split_config.split_layer:])
        self.pooler = self.bert.pooler
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # SST-2 has 2 classes

    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)[0].clone()

        pooled_output = self.pooler(hidden_states)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
