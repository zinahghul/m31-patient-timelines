import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class EHRTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, 
        vocab_size=209, 
        num_classes=40, 
        d_model=128, 
        nhead=4, 
        num_layers=3, 
        dim_feedforward=512, 
        max_seq_len=512, 
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # FIX: Restored padding_idx=0
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
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

    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(input_ids)
        
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        # FIX: Apply attention mask so padded tokens are ignored
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        cls_representation = x[:, 0, :]
        logits = self.classifier(cls_representation)
        
        return logits