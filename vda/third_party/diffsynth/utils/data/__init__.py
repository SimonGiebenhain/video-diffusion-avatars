# From DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio), Apache-2.0, see
# vda/third_party/diffsynth/LICENSE. Changed for video-diffusion-avatars: reduced to what the released
# inference uses, plus the additions described below where present. These changes and additions are
# licensed under CC BY-NC-SA 4.0 (LICENSE at the repository root); the upstream code stays Apache-2.0.
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


class LowMemoryVideo:
    def __init__(self, file_name):
        self.reader = imageio.get_reader(file_name)
    
    def __len__(self):
        return self.reader.count_frames()

    def __getitem__(self, item):
        return Image.fromarray(np.array(self.reader.get_data(item))).convert("RGB")

    def __del__(self):
        self.reader.close()


class VideoData:
    """Frames of a video file, read on demand (trimmed from DiffSynth-Studio: no image folders,
    no crop-and-resize)."""

    def __init__(self, video_file):
        self.data = LowMemoryVideo(video_file)

    def raw_data(self):
        frames = []
        for i in range(self.__len__()):
            frames.append(self.__getitem__(i))
        return frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data.__getitem__(item)


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()
