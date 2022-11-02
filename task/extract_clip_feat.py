import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

import clip


class ImageDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path

        subdirs = os.listdir(self.root_path)
        self.images = []
        for subdir in subdirs:
            self.images.extend((self.root_path / subdir).glob("*.jpeg"))
        pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        image = Image.open(self.images[index])
        return image


def load_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


if __name__ == "__main__":
    DATASET_ROOT = Path("/CLIP/test_dataset")
    IMAGE_ROOT = DATASET_ROOT / "image"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = load_model(device)

    image_dataset = ImageDataset(root_path=IMAGE_ROOT)
