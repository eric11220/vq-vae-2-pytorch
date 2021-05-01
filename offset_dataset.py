import numpy as np
import os
import random
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from PIL import Image

class OffsetDataset(Dataset):
    def __init__(self, frame_dir, offset=5, transform=None):
        self.transform = transform

        self.path_pairs = []
        for vid in os.listdir(frame_dir):
            vid_dir = os.path.join(frame_dir, vid)
            for frame in sorted(os.listdir(vid_dir)):
                frame_path = os.path.join(vid_dir, frame)
                frame_no, ext = os.path.splitext(frame)
                frame_no = int(frame_no)
                next_frame = frame_no + offset
                next_frame_path = os.path.join(vid_dir, f'{next_frame:04d}{ext}')
                if os.path.isfile(next_frame_path):
                    self.path_pairs.append((frame_path, next_frame_path))

        print(f'Total number of training data: {len(self.path_pairs)}')

    def __len__(self):
        return len(self.path_pairs)

    def __getitem__(self, index):
        frame_path, next_frame_path = self.path_pairs[index]
        frame = Image.open(frame_path)
        next_frame = Image.open(next_frame_path)

        if self.transform is not None:
            seed = np.random.randint(2147483647)
            random.seed(seed); torch.manual_seed(seed)
            frame_t = self.transform(frame)

            seed = np.random.randint(2147483647)
            random.seed(seed); torch.manual_seed(seed)
            next_frame_t = self.transform(next_frame)

        return frame_t, next_frame_t


if __name__ == '__main__':
    frame_dir = 'datasets/adobe_240fps/frames/train'

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = OffsetDataset(frame_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, num_workers=4)

    # for frame, next_frame in loader:
    #     print(frame.shape, next_frame.shape)
    #     input("")
