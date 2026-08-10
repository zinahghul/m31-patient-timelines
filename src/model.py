import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class EHRTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, 
        vocab_size=209, 
        num_classes=80, 
        d_model=128, 
        nhead=4, 
        num_layers=3, 
        dim_feedforward=512, 
        max_seq_len=512, 
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # FIX: Removed padding_idx=0 to prevent LayerNorm division-by-zero
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        self.classifier = nn.Linear(d_model, num_classes)

    

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(input_ids)
        
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        # We explicitly do NOT pass a mask here to bypass the SDPA Windows bug
        x = self.transformer(x)
        
        cls_representation = x[:, 0, :]
        logits = self.classifier(cls_representation)
        
        return logits