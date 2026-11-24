from torch.utils.data import DataLoader
from mmlab.data.flickr8k import Flickr8kDataset, flickr8k_clooate_fn

def main():
    dataset = Flickr8kDataset(split="train")
    print("Dataset length:", len(dataset))
    sample=dataset[0]
    print("\n--- Single Sample ---")
    print("Image ID:", sample["image_id"])
    print("Image size:", sample["image"].size)
    print("Image mode:", sample["image"].mode)

    for i,caption in enumerate(sample["captions"]):
        print(f"Caption {i+1}:", caption)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=flickr8k_clooate_fn
    )

    batch = next(iter(loader))

    print("\n--- Batch ---")
    print("Batch image IDs:", batch["image_ids"])
    print("Number of images in batch:", len(batch["images"]))
    print("Number of captions in batch:", len(batch["captions"]))
    print("Captions per image:", [
        len(captions)
        for captions in batch["captions"]
    ])

    assert len(dataset) == 6000
    assert len(batch["images"]) == 4
    assert all(len(c) == 5 for c in batch["captions"])

    print("\nData pipeline: SUCCESS")


if __name__ == "__main__":
    main()