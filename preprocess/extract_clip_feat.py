import os
import json
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

import clip


class ImageDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path

        subdirs = os.listdir(self.root_path)
        self.images = []
        for subdir in subdirs:
            self.images.extend((self.root_path / subdir).glob("*.jpg"))

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

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index: int):
        return self.captions[index]["id"], self.captions[index]["caption"]


def load_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


if __name__ == "__main__":
    batch_size = 640

    # existing dataset path
    DATASET_ROOT = Path("/data/MSRVTT/")
    IMAGE_ROOT = DATASET_ROOT / "videos" / "frames"
    TEXT_ROOT = DATASET_ROOT / "annotation"

    # featrue path
    FEAT_ROOT = DATASET_ROOT / "feat" / "clip"
    FEAT_IMAGE_ROOT = FEAT_ROOT / "image"
    FEAT_TEXT_ROOT = FEAT_ROOT / "text"
    FEAT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    FEAT_TEXT_ROOT.mkdir(parents=True, exist_ok=True)
    """
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
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model, preprocess = load_model(device)

    """
    # define datasets
    def image_collate(batch):
        image_paths = []
        image_batch = []
        for image_path, image in batch:
            image_paths.append(image_path)
            image_batch.append(preprocess(image))
        image_batch = torch.stack(image_batch, dim=0)
        return image_paths, image_batch

    image_dataset = ImageDataset(root_path=IMAGE_ROOT)
    image_dataloader = DataLoader(image_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=image_collate,
                                  drop_last=False)
    """

    def text_collate(batch):
        text_ids = []
        texts = []
        for text_id, text in batch:
            text_ids.append(text_id)
            texts.append(text)

        text_tokens = clip.tokenize(texts)
        return text_ids, text_tokens

    text_dataset = TextDataset(root_path=TEXT_ROOT)
    text_dataloader = DataLoader(text_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=text_collate,
                                 drop_last=False)

    """
    # extract image feature
    for image_paths, image in tqdm(image_dataloader, desc="Extracting image features"):
        image = image.to(device)

        with torch.no_grad():
            # image_feat [batch_size, D=512]
            image_feat = model.encode_image(image)
            # image_feat [D=512]
            image_feat = image_feat.detach().cpu().numpy()

        for idx, image_path in enumerate(image_paths):
            image_name = str(image_path).split("/")[-1].split(".")[0]
            video_name = str(image_path.parents[0]).split("/")[-1]

            feat_video_path = FEAT_IMAGE_ROOT / video_name
            feat_video_path.mkdir(parents=True, exist_ok=True)

            image_feat_path = feat_video_path / f"{image_name}.npy"

            with open(image_feat_path, 'wb') as f:
                np.save(f, image_feat[idx])
    """

    # extract text feature
    for text_ids, text_token in tqdm(text_dataloader, desc="Extracting text features"):
        text_token = text_token.to(device)

        with torch.no_grad():
            # text_feat [batch_size, D=512]
            text_feat = model.encode_text(text_token)
            text_feat = text_feat.detach().cpu().numpy()

        for idx, text_id in enumerate(text_ids):
            text_feat_path = FEAT_TEXT_ROOT / f"{str(text_id).zfill(8)}.npy"

            with open(text_feat_path, 'wb') as f:
                np.save(f, text_feat[idx])
