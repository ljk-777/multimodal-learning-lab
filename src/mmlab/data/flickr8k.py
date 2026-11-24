from datasets import load_dataset
from torch.utils.data import Dataset

class Flick8kDataset(Dataset):
    def __init__(
            self,
            split="train",
            cache_dir="data/hf_cache"
    ):
        self.split = split

        self.samples = load_dataset(
            "jxie/flickr8k",
            split=split,
            cache_dir=cache_dir
        )
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample["image"].convert("RGB")

        captions = [
            sample[f"caption_{i}"].strip()
            for i in range(5)
        ]
        return {
            "image_id": f"{self.split}:{idx}",
            "image": image,
            "captions": captions
        }

def flickr8k_clooate_fn(batch):

    return {
        "image_ids":[
            sample["image_id"] for sample in batch
        ],
        "images":[
            sample["image"] for sample in batch
        ],
        "captions":[
            sample["captions"] for sample in batch
        ]
    }