import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from offset_dataset import OffsetDataset
from vqvae import VQVAE

class OffsetNetwork(VQVAE):
    def __init__(self, vqvae, in_channel=6, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99):

        super(OffsetNetwork, self).__init__(
            in_channel=in_channel,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            embed_dim=embed_dim,
            n_embed=n_embed,
            decay=decay
        )

        # Fix pre-trained VQVAE
        self.vqvae = vqvae if vqvae is not None else VQVAE()
        for params in self.vqvae.parameters():
            params.requires_grad = False

    def forward(self, frames, next_frames, offset_only=False, offset_weight=1):
        inputs = torch.cat((frames, next_frames), dim=1)
        offset_quant_t, offset_quant_b, diff, _, _ = self.encode(inputs)

        quant_t = offset_weight * offset_quant_t
        quant_b = offset_weight * offset_quant_b
        if not offset_only:
            frame1_quant_t, frame1_quant_b, _, _, _ = self.vqvae.encode(frames)
            quant_t += frame1_quant_t
            quant_b += frame1_quant_b

        dec = self.vqvae.decode(quant_t, quant_b)
        return dec, diff


if __name__ == '__main__':
    frame_dir = 'datasets/adobe_240fps/frames/train'
    ckpt = 'checkpoint/vqvae_175.pt'
    device = 'cuda'

    # Load pre-trained VQVAE
    vqvae = VQVAE().to(device)
    try:
        vqvae.load_state_dict(torch.load(ckpt))
    except:
        print("Seems the checkpoint was trained with data parallel, try loading it that way")
        weights = torch.load(ckpt)
        renamed_weights = {}
        for key, value in weights.items():
            renamed_weights[key.replace('module.', '')] = value
        weights = renamed_weights
        vqvae.load_state_dict(weights)
    vqvae.eval()

    # Init offset encoder
    model = OffsetNetwork(vqvae).to(device)

    # Load Offset Dataset
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = OffsetDataset(frame_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, num_workers=4)

    for frames, next_frames in loader:
        frames, next_frames = frames.to(device), next_frames.to(device)
        out, latent_loss = model(frames, next_frames)
        print(out.shape, latent_loss)
        input("")
