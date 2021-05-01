import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from offset_dataset import OffsetDataset
from offset_encoder import OffsetNetwork
from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist



def train(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (frames, next_frames) in enumerate(loader):
        model.zero_grad()

        frames = frames.to(device)
        next_frames = next_frames.to(device)

        out, latent_loss = model(frames, next_frames)
        recon_loss = criterion(out, next_frames)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * frames.shape[0]
        part_mse_n = frames.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                model.eval()

                sample_frames = frames[:sample_size]
                sample_next = next_frames[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample_frames, sample_next)

                utils.save_image(
                    torch.cat([sample_frames, out], 0),
                    f"offset_sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

                model.train()


def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = OffsetDataset(args.path, transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=args.bsize // args.n_gpu, sampler=sampler, num_workers=2
    )

    # Load pre-trained VQVAE
    vqvae = VQVAE().to(device)
    try:
        vqvae.load_state_dict(torch.load(args.ckpt))
    except:
        print("Seems the checkpoint was trained with data parallel, try loading it that way")
        weights = torch.load(args.ckpt)
        renamed_weights = {}
        for key, value in weights.items():
            renamed_weights[key.replace('module.', '')] = value
        weights = renamed_weights
        vqvae.load_state_dict(weights)

    # Init offset encoder
    model = OffsetNetwork(vqvae).to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
            find_unused_parameters=True
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint/offset_enc_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--bsize", type=int, default=128)

    parser.add_argument("path", type=str)
    parser.add_argument("ckpt", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
