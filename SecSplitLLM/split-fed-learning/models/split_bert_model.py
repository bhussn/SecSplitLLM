import torch.nn as nn
from transformers import BertModel, BertConfig
import torch

class BertSplitConfig:
    def __init__(self, model_name="bert-base-uncased", split_layer=5):
        self.model_name = model_name
        self.split_layer = split_layer
        self.config = BertConfig.from_pretrained(model_name)

class BertModel_Client(nn.Module):
    def __init__(self, split_config: BertSplitConfig):
        super(BertModel_Client, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.bert = BertModel.from_pretrained(split_config.model_name)
        self.embeddings = self.bert.embeddings
        self.encoder_layers = nn.ModuleList(self.bert.encoder.layer[:split_config.split_layer])
        
        hidden_size = split_config.config.hidden_size
        self.pool_proj = nn.Linear(hidden_size, 128) 

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=x.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        for layer in self.encoder_layers:
            x = layer(x, attention_mask=extended_attention_mask)[0]
        
        # Masked mean pooling (taking attention mask into account)
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
        input_size = 128  # Must match client pool_proj output size
        self.classifier = nn.Linear(input_size, 3)  # Adjust num_classes if needed

    def forward(self, pooled_activations):
        x = self.dropout(pooled_activations)
        logits = self.classifier(x)
        return logits



# import torch
# import torch.nn as nn
# from transformers import BertModel, BertConfig

# class BertSplitConfig:
#     def __init__(self, model_name="bert-base-uncased", split_layer=5):
#         self.model_name = model_name
#         self.split_layer = split_layer
#         self.config = BertConfig.from_pretrained(model_name)

# class AttentionPooling(nn.Module):
#     def __init__(self, hidden_size):
#         super(AttentionPooling, self).__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_size, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1)
#         )
    
#     def forward(self, x, mask):
#         # x: (batch_size, seq_len, hidden_size)
#         # mask: (batch_size, seq_len)
        
#         scores = self.attention(x).squeeze(-1)  # (batch_size, seq_len)
#         scores = scores.masked_fill(mask == 0, float('-inf'))  # mask padding tokens
#         weights = torch.softmax(scores, dim=1)  # attention weights
#         pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)  # weighted sum
#         return pooled

# class BertModel_Client(nn.Module):
#     def __init__(self, split_config: BertSplitConfig):
#         super(BertModel_Client, self).__init__()
#         self.dropout = nn.Dropout(p=0.1)
#         self.bert = BertModel.from_pretrained(split_config.model_name)
#         self.embeddings = self.bert.embeddings
#         self.encoder_layers = nn.ModuleList(self.bert.encoder.layer[:split_config.split_layer])
        
#         hidden_size = split_config.config.hidden_size
#         self.attn_pool = AttentionPooling(hidden_size)
#         self.pool_proj = nn.Linear(hidden_size, 128)  # project to smaller embedding dimension

#     def forward(self, input_ids, attention_mask):
#         x = self.embeddings(input_ids)
        
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=x.dtype)
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
#         for layer in self.encoder_layers:
#             x = layer(x, attention_mask=extended_attention_mask)[0]
        
#         pooled = self.attn_pool(x, attention_mask)
        
#         pooled_output = self.pool_proj(pooled)
#         pooled_output = self.dropout(pooled_output)

#         return pooled_output

# class BertModel_Server(nn.Module):
#     def __init__(self, split_config: BertSplitConfig):
#         super(BertModel_Server, self).__init__()
#         self.dropout = nn.Dropout(p=0.1)
#         input_size = 128  # Must match client pool_proj output size
#         self.classifier = nn.Linear(input_size, 2)  # Adjust num_classes if needed

#     def forward(self, pooled_activations):
#         x = self.dropout(pooled_activations)
#         logits = self.classifier(x)
#         return logits
