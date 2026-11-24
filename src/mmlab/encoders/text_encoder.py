import torch 
from torch import nn
from transformers import AutoTokenizer, AutoModel

class DistilBERTTextEncoder(nn.Module):

    def __init__(
        self,
        model_name="distilbert/distilbert-base-uncased",
        max_length=64,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone
        self.max_length = max_length
        self.output_dim = self.backbone.config.dim

        #freeze backbone
        self.backbone.requires_grad_(False)

        self.eval()

    def preprocess(self,captions):
        return self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,  # 新增
            return_tensors="pt"
        )
    def train(self,mode=True):
        super().train(mode)

        self.backbone

        return self

    @torch.no_grad()
    def forward(self,input_ids,attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state
        mask=attention_mask.unsqueeze(-1)
        mask=mask.to(hidden_states.dtype)

        features = (
            (hidden_states * mask).sum(dim=1)
            /mask.sum(dim=1)
        )

        return features