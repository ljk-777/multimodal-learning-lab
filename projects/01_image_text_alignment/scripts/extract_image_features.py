import torch
from torch.utils.data import DataLoader
from mmlab.data.flickr8k import Flickr8kDataset, flickr8k_clooate_fn
from mmlab.encoders.image_encoder import ViTImageEncoder

def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    dataset = Flickr8kDataset(split="train")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=flickr8k_clooate_fn
    )

    encoder = ViTImageEncoder().to(device)
    batch = next(iter(loader))
    images = batch["images"]
    x=encoder.preprocess(images).to(device)

    features = encoder(x)

    print("Image IDs:", batch["image_ids"])
    print("Input shape:", x.shape)
    print("Features shape:", features.shape)
    print("Requires grad:", features.requires_grad)

    total_params = sum(
        p.numel() for p in encoder.parameters()
    )
    trainable_params = sum(
        p.numel() for p in encoder.parameters()
        if p.requires_grad
    )

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    assert features.shape == (4, 768)
    assert torch.isfinite(features).all()
    assert trainable_params == 0
    print("Image encoder: SUCCESS")


if __name__ == "__main__":
    main()