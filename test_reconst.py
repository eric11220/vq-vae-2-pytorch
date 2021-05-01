import argparse
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from matplotlib import pyplot as plt

import distributed as dist
from offset_dataset import OffsetDataset
from offset_encoder import OffsetNetwork
from vqvae import VQVAE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("ckpt")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--offset-encoder", action='store_true')
    parser.add_argument("--offset-weight", type=float, default=1.)
    parser.add_argument("--mode", choices=['plot', 'mse'], default='plot')

    args = parser.parse_args()
    return args

args = parse_args()
device = "cuda"

torch.manual_seed(820)
random.seed(820)
np.random.seed(820)

transform = transforms.Compose(
    [
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

dataset = OffsetDataset(args.path, transform=transform)
loader = DataLoader(
    dataset, batch_size=8, num_workers=2, shuffle=True
)

criterion = nn.MSELoss()

if args.offset_encoder:
    model = OffsetNetwork(None).to(device)
else:
    model = VQVAE().to(device)

try:
    model.load_state_dict(torch.load(args.ckpt))
except:
    print("Seems the checkpoint was trained with data parallel, try loading it that way")
    weights = torch.load(args.ckpt)
    renamed_weights = {}
    for key, value in weights.items():
        renamed_weights[key.replace('module.', '')] = value
    weights = renamed_weights
    model.load_state_dict(weights)

model.eval()
with torch.no_grad():
    total_frame = 0
    total_frame1_mse, total_frame2_mse = 0., 0.
    for i, (img, next_frame) in enumerate(loader):
        img_t = img.to(device)
        next_frame = next_frame.to(device)

        if args.offset_encoder:
            out, _ = model(img_t, next_frame)
            if args.offset_weight != 1.:
                offset_weighted_out, _ = model(img_t, next_frame, offset_weight=args.offset_weight)
            else:
                offset_weighted_out = None

            offset_out, _ = model(img_t, next_frame, offset_only=True, offset_weight=args.offset_weight)
        else:
            out, _ = model(next_frame)

        if args.mode == 'plot':
            to_plot = torch.cat([img_t, next_frame, out, offset_out], 0)
            if offset_weighted_out is not None:
                to_plot = torch.cat([to_plot, offset_weighted_out], 0)

            utils.save_image(
                to_plot,
                f"results/batch_{i}.png",
                nrow=len(img_t),
                normalize=True,
                range=(-1, 1),
            )
        elif args.mode == 'mse':
            if args.offset_encoder:
                frame1_mse = 0.
                frame2_mse = criterion(next_frame, out)

                total_frame1_mse += frame1_mse * len(img)
                total_frame2_mse += frame2_mse * len(img)
                total_frame += len(img)
            else:
                frame1_mse = criterion(img_t, out)
                frame2_mse = criterion(next_frame, out)

                total_frame1_mse += frame1_mse * len(img)
                total_frame2_mse += frame2_mse * len(img)
                total_frame += len(img)

    print(f'Pixel-wise MSE with frame 1: {total_frame1_mse/total_frame:.4f}, ' \
        + f'Pixel-wise MSE with frame 2: {total_frame2_mse/total_frame:.4f}')
