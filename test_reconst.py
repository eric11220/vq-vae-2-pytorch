import argparse
import numpy as np
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

    args = parser.parse_args()
    return args


# def plot_reconst(orig, reconst, prefix):
#     for i, (o, r) in enumerate(zip(orig, reconst)):
#         path = f'{prefix}_img{i}.png'
#         concat = np.concatenate((o, r), axis=1)
#         concat = (concat / 2 + 0.5)
#         concat[concat < 0.] = 0.
#         concat[concat > 1.] = 1.
#         plt.imsave(path, concat)


args = parse_args()
device = "cuda"

transform = transforms.Compose(
    [
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

if args.offset_encoder:
    dataset = OffsetDataset(args.path, transform=transform)
else:
    dataset = datasets.ImageFolder(args.path, transform=transform)
loader = DataLoader(
    dataset, batch_size=8, num_workers=2, shuffle=True
)

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
    for i, (img, label_or_nextf) in enumerate(loader):
        img_t = img.to(device)
        label_or_nextf = label_or_nextf.to(device)

        if args.offset_encoder:
            out, _ = model(img_t, label_or_nextf)
            offset_out, _ = model(img_t, label_or_nextf, offset_only=True)
        else:
            out, _ = model(img_t)

        # img = np.rollaxis(img.numpy(), 1, 4)
        # out = np.rollaxis(out.cpu().numpy(), 1, 4)
        # plot_reconst(img, out, prefix=f'results/batch_{i}')

        utils.save_image(
            torch.cat([img_t, label_or_nextf, out, offset_out], 0),
            f"results/batch_{i}.png",
            nrow=len(img_t),
            normalize=True,
            range=(-1, 1),
        )
