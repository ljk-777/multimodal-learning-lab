import inspect

import torch
from transformers import AutoTokenizer,AutoModel

from mmlab.data.flickr8k import Flickr8kDataset

def main():
    model_name = "distilbert/distilbert-base-uncased"

    device = torch.device(
        "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

    #Load real captions
    dataset = Flickr8kDataset(split="train")

    sample = dataset[0]
    captions = sample["captions"]

    #Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model = model.to(device)
    model.eval()

    #Tokenization
    inputs = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    print("Captions:",captions)
    print("Input IDs:",inputs["input_ids"])
    print("Attention Mask:",inputs["attention_mask"])
    print("Input IDs shape:", inputs["input_ids"].shape)

    print(
        "First caption tokens:",
        tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0].tolist()
        )
    )

    #Move inputs to GPU
    inputs = inputs.to(device)

    with torch.no_grad():
        #Transformer forward
        outputs = model(**inputs)

        hidden_states = outputs.last_hidden_state

        #Masked mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1)
        mask = mask.to(hidden_states.dtype)
        masked_hidden = hidden_states * mask

        features=(
            masked_hidden.sum(dim=1)
            /mask.sum(dim=1).clamp_min(1)
        )

    print("Embedding layer:", model.get_input_embeddings())
    print("Transformer layers:", model.config.n_layers)

    print("Hidden states shape:", hidden_states.shape)
    print("Mask shape:", mask.shape)
    print("Text features shape:", features.shape)
    print("Device:", features.device)

    print("First 10 features:", features[0, :10])

    # 7. Inspect source code
    print(inspect.getsource(model.embeddings.forward))
    print(inspect.getsource(model.forward))

    assert features.shape == (5, 768)
    assert torch.isfinite(features).all()

    print("Text feature extraction: SUCCESS")


if __name__ == "__main__":
    main()