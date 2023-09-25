from typing import List

import os
import clip
import torch
import decord
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def extract_clips(vidpath: str, clip_duration: float, num_frames: int) -> List[np.array]:
    """
    clip_duration: seconds for each clip
    num_frames: frames per clip
    """
    reader = decord.VideoReader(vidpath)

    vlen = len(reader)
    fps = reader.get_avg_fps()
    duration = vlen / fps  # in seconds
    num_clips = int(duration / clip_duration)

    clips = []
    for clip_id in range(num_clips):
        frames_per_clip = clip_duration * fps
        start = clip_id * frames_per_clip
        end = (clip_id+1) * frames_per_clip
        idx = np.linspace(start, end, num=num_frames, endpoint=False).astype(np.int64)
        clip = reader.get_batch(idx).asnumpy()
        clip = [clip[i] for i in range(len(clip))]
        clips.append(clip)

    return clips


def load_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


if __name__ == "__main__":
    batch_size = 64

    data_root = Path("/data/moma_web")
    raw_path = data_root / "raw"
    clip_path = data_root / "clip"
    os.makedirs(clip_path, exist_ok=True)
    vid_ids = [v.split('.')[0] for v in os.listdir(raw_path)]

    model, preprocess = load_model('cuda')

    pbar = tqdm(vid_ids)
    for vid_id in pbar:
        pbar.set_description(f"[{vid_id}]")

        clips = extract_clips(str(raw_path / f"{vid_id}.mp4"), 8, 8)

        # preprocess frames
        frames = []
        for c in clips:
            for f in c:
                frames.append(preprocess(Image.fromarray(f)))

        # make batch
        n_batch = len(frames) // batch_size
        remainder = len(frames) % batch_size
        batches = []
        for b_idx in range(n_batch):
            batch = torch.stack(frames[b_idx*batch_size:(b_idx+1)*batch_size],
                                dim=0)
            batches.append(batch)
        if remainder > 0:
            batch = torch.stack(frames[n_batch*batch_size:], dim=0)
            batches.append(batch)

        feat = []
        for batch in batches:
            batch = batch.to('cuda')
            with torch.no_grad():
                # image_feat [batch_size, D=512]
                image_feat = model.encode_image(batch)
                # image_feat [batch_size, D=512]
                image_feat = image_feat.detach().cpu()
                feat.append(image_feat)

        feat = torch.cat(feat, dim=0).numpy()

        # save feat
        with open(clip_path / f"{vid_id}.npy", "wb") as f:
            np.save(f, feat)
