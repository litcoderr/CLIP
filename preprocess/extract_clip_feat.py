import os
import json
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import clip


class ImageDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path

        subdirs = os.listdir(self.root_path)
        self.images = []
        for subdir in subdirs:
            self.images.extend((self.root_path / subdir).glob("*.jpeg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        image = Image.open(self.images[index])
        return self.images[index], image


class TextDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        with open(self.root_path / "MSR_VTT.json", "r") as f:
            self.json = json.load(f)
        self.captions = self.json["annotations"]
        pass

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index: int):
        return index, self.captions[index]["caption"]


def load_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


if __name__ == "__main__":
    # existing dataset path
    DATASET_ROOT = Path("/CLIP/test_dataset")
    IMAGE_ROOT = DATASET_ROOT / "image"
    TEXT_ROOT = DATASET_ROOT / "text"

    # featrue path
    FEAT_ROOT = DATASET_ROOT / "feat"
    FEAT_IMAGE_ROOT = FEAT_ROOT / "image"
    FEAT_TEXT_ROOT = FEAT_ROOT / "text"
    FEAT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    FEAT_TEXT_ROOT.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model, preprocess = load_model(device)

    # define datasets
    image_dataset = ImageDataset(root_path=IMAGE_ROOT)
    text_dataset = TextDataset(root_path=TEXT_ROOT)

    # extract text feature
    for text_id, text in tqdm(text_dataset):
        pass

    # extract image feature
    for image_path, image in tqdm(image_dataset):
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # image_feat [batch_size=1, D=512]
            image_feat = model.encode_image(image)
            # image_feat [D=512]
            image_feat = image_feat.squeeze(0)
            image_feat = image_feat.detach().cpu().numpy()

        image_name = str(image_path).split("/")[-1].split(".")[0]
        video_name = str(image_path.parents[0]).split("/")[-1]

        feat_video_path = FEAT_IMAGE_ROOT / video_name
        feat_video_path.mkdir(parents=True, exist_ok=True)

        image_feat_path = feat_video_path / f"{image_name}.npy"

        with open(image_feat_path, 'wb') as f:
            np.save(f, image_feat)
