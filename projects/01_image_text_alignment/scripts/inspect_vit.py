import torch 
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_V1 # 一个枚举类（Enum），用于管理不同版本的预训练权重及其相关信息
model = vit_b_16(weights=weights)
print(model.conv_proj) # 输出模型的卷积投影层
print(model.heads)
print(model.hidden_dim)
print(model.class_token.shape)
print(model.encoder.pos_embedding.shape)

import inspect 
print(inspect.getfile(type(model)))
print(inspect.getsource(model._process_input))
print(inspect.getsource(model.forward))

from mmlab.data.flickr8k import Flickr8kDataset, flickr8k_clooate_fn
dataset = Flickr8kDataset(split="train")
transform = weights.transforms()
images=[
    transform(dataset[i]["image"])
    for i in range(4)
]
x = torch.stack(images,dim=0)
device = torch.device(
    "mps" if torch.backends.mps.is_available else "cpu"
)
x = x.to(device)
model.heads=nn.Identity()
model=model.to(device)
model.eval()
with torch.no_grad():
    patchs = model.conv_proj(x)
    tokens = model._process_input(x)
    features = model(x)

print("device:", device)
print("Input shape:", x.shape)
print("Patchs shape:", patchs.shape)
print("Tokens shape:", tokens.shape)
print("Features shape:", features.shape)
print("Requires grad:", features.requires_grad)
print("First 10 features:", features[0,:10])

assert features.shape == (4, 768)
assert torch.isfinite(features).all()
print("ViT feature extraction: SUCCESS")