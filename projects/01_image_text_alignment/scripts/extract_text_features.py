import torch
from torch.utils.data import DataLoader
from mmlab.data.flickr8k import (
    Flickr8kDataset,
    flickr8k_clooate_fn
)
from mmlab.encoders.text_encoder import (
    DistilBERTTextEncoder,
)

def main():

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dataset = Flickr8kDataset(split="train")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=flickr8k_clooate_fn,
    )
    encoder = DistilBERTTextEncoder().to(device)

    batch = next(iter(loader))

    # Flatten caption groups
    captions=[]

    for group in batch["captions"]:
        captions.extend(group)
    #Tokenization
    inputs = encoder.preprocess(captions).to(device)
    #feature extraction
    features = encoder(**inputs)
    print("Number of images:", len(batch["images"]))
    print("Number of captions:", len(captions))
    print("Input shape:", inputs["input_ids"].shape)
    print("Feature shape:", features.shape)
    print("Device:", features.device)

    trainable_params = sum(
        p.numel()
        for p in encoder.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:", trainable_params)

    assert features.shape == (20, 768)
    assert torch.isfinite(features).all()
    assert trainable_params == 0
    assert not features.requires_grad

    print("Text encoder: SUCCESS")


if __name__ == "__main__":
    main()