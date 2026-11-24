import torch
from torch import nn
from torchvision.models import (
    vit_b_16, 
    ViT_B_16_Weights
)
class ViTImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.transform = weights.transforms()
        self.backbone = vit_b_16(weights=weights)
        self.out_dim = self.backbone.hidden_dim
        self.backbone.heads = nn.Identity()

        #freeze the backbone parameters
        self.backbone.requires_grad_(False)
        self.eval()
    def preprocess(self,images):
        tensors = [
            self.transform(image)
            for image in images
        ]
        return torch.stack(tensors,dim=0)

    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self,x):
        return self.backbone(x)