import torch.nn as nn
from transformers import ViTModel, ViTConfig, AutoImageProcessor

class MCViT(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2):
        super(MCViT, self).__init__()
        self.transformer = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k") #Feature extraction model https://huggingface.co/google/vit-base-patch16-224-in21k 
        
        # Add dropout to intermediate ff layers
        for layer in self.transformer.encoder.layer:
            layer.intermediate.dense = nn.Sequential(
                nn.Linear(in_features=768, out_features=3072, bias=True),
                nn.Dropout(dropout)
            )
            layer.output.dropout = nn.Dropout(dropout)
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.transformer.config.hidden_size, num_classes)
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.dropout(x.last_hidden_state[:, 0, :])  # Use CLS token
        x = self.fc(x)
        return x
    
    # Force dropout layers to train mode to activate dropout during evaluation
    def activate_dropout(self):
        for layer in self.transformer.encoder.layer:
            layer.intermediate.dense[1].train()
            layer.output.dropout.train()
