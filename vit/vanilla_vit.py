import torch.nn as nn
from transformers import ViTModel, ViTConfig, AutoImageProcessor

class ViT(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2):
        super(ViT, self).__init__()
        self.transformer = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k") #Feature extraction model https://huggingface.co/google/vit-base-patch16-224-in21k 
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.transformer.config.hidden_size, num_classes)
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.dropout(x.last_hidden_state[:, 0, :])  # Use CLS token
        x = self.fc(x)
        return x