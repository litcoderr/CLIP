import os
import subprocess
from tqdm import tqdm


def main():
    video_path = "/data/MSRVTT/videos/all"
    output_path = "/data/MSRVTT/videos/frames"
    for filename in tqdm(os.listdir(video_path)):
        vid = filename.split(".")[0]
        os.makedirs(os.path.join(output_path, vid), exist_ok=True)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i",
            os.path.join(video_path, filename),
            "-s",
            "224x224",
            "-vf",
            "fps=30",
            "-qscale:v",
            "1",
            os.path.join(output_path, vid, "frame_%4d.jpg")
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
