import numpy as np

from tqdm import tqdm

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger

from pytorch_optimizer import Prodigy

from vocos.dataset import VocosDataset
from vocos.feature_extractors import MelSpectrogramFeatures
from vocos.models import VocosBackbone
from vocos.heads import ISTFTHead
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss

BATCH_SIZE_PER_GB = 2.66
NUM_EPOCHS = 1000
NUM_SAMPLES = 16384
NUM_WORKERS = 8
LEARNING_RATE = 5e-4
USE_PRODIGY = True

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = MelSpectrogramFeatures(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, padding="center")
        self.backbone = VocosBackbone(input_channels=100, dim=512, intermediate_dim=1536, num_layers=8)
        self.head = ISTFTHead(dim=512, n_fft=1024, hop_length=256, padding="center")

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(self.backbone(features))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.multi_period = MultiPeriodDiscriminator()
        self.multi_resolution = MultiResolutionDiscriminator()

def main():
    #torch.set_float32_matmul_precision("high")
    gpu = torch.cuda.get_device_properties(0)
    batch_size = int((round(gpu.total_memory / (1024 ** 3)) - 1) * BATCH_SIZE_PER_GB)
    mr_loss_coeff = 0.1
    mel_loss_coeff = 45

    logger = TensorBoardLogger(root_dir="logs")

    fabric = Fabric(loggers=logger, precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(4444)

    dataset = VocosDataset(filelist_path="filelist.train", sampling_rate=24000, num_samples=NUM_SAMPLES, train=True)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=NUM_WORKERS)

    with fabric.init_module():
        generator = Generator()
        discriminator = Discriminator()

        discriminator_loss = DiscriminatorLoss()
        generator_loss = GeneratorLoss()
        feature_loss = FeatureMatchingLoss()
        mel_loss = MelSpecReconstructionLoss()

    if USE_PRODIGY:
        optimizer_g = Prodigy(generator.parameters(), lr=1.0, betas=(0.8, 0.9), weight_decay=0.01, safeguard_warmup=True, bias_correction=True)
        optimizer_d = Prodigy(discriminator.parameters(), lr=1.0, betas=(0.8, 0.9), weight_decay=0.01, safeguard_warmup=True, bias_correction=True)
    else:
        optimizer_g = AdamW(generator.parameters(), lr=LEARNING_RATE, betas=(0.8, 0.9))
        optimizer_d = AdamW(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.8, 0.9))

    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=NUM_EPOCHS)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=NUM_EPOCHS)

    generator, optimizer_g = fabric.setup(generator, optimizer_g)
    discriminator, optimizer_d = fabric.setup(discriminator, optimizer_d)

    dataloader = fabric.setup_dataloaders(dataloader)

    step = 0
    losses = {
        "discriminator/total_loss": [],
        "discriminator/multi_period_loss": [],
        "discriminator/multi_resolution_loss": [],
        "generator/multi_period_loss": [],
        "generator/multi_resolution_loss": [],
        "generator/fm_mp_loss": [],
        "generator/fm_mr_loss": [],
        "generator/mel_loss": [],
        "generator/total_loss": []
    }
    for epoch in range(NUM_EPOCHS):
        with tqdm(dataloader) as t:
            t.set_description(f"Epoch {epoch}")
            for data in t:
                audio_real = data
                audio_fake = generator(audio_real)

                optimizer_d.zero_grad()

                real_score_mp, fake_score_mp, _, _ = discriminator.multi_period(audio_real, audio_fake.detach())
                real_score_mr, fake_score_mr, _, _ = discriminator.multi_resolution(audio_real, audio_fake.detach())

                loss_mp, loss_mp_real, _ = discriminator_loss(real_score_mp, fake_score_mp)
                loss_mr, loss_mr_real, _ = discriminator_loss(real_score_mr, fake_score_mr)
                loss_mp /= len(loss_mp_real)
                loss_mr /= len(loss_mr_real)
                loss_d = loss_mp + mr_loss_coeff * loss_mr

                fabric.backward(loss_d)
                optimizer_d.step()

                optimizer_g.zero_grad()

                _, fake_score_mp, fmap_real_mp, fmap_fake_mp = discriminator.multi_period(audio_real, audio_fake)
                _, fake_score_mr, fmap_real_mr, fmap_fake_mr = discriminator.multi_resolution(audio_real, audio_fake)

                loss_fake_mp, list_loss_fake_mp = generator_loss(fake_score_mp)
                loss_fake_mr, list_loss_fake_mr = generator_loss(fake_score_mr)
                loss_fake_mp /= len(list_loss_fake_mp)
                loss_fake_mr /= len(list_loss_fake_mr)
                loss_fmap_mp = feature_loss(fmap_real_mp, fmap_fake_mp)
                loss_fmap_mr = feature_loss(fmap_real_mr, fmap_fake_mr)

                loss_mel = mel_loss(audio_fake, audio_real)
                loss_g = loss_fake_mp + mr_loss_coeff * loss_fake_mr + loss_fmap_mp + mr_loss_coeff * loss_fmap_mr + mel_loss_coeff * loss_mel

                fabric.backward(loss_g)
                optimizer_g.step()

                losses["discriminator/total_loss"].append(loss_d.item())
                losses["discriminator/multi_period_loss"].append(loss_mp.item())
                losses["discriminator/multi_resolution_loss"].append(loss_mr.item())
                losses["generator/multi_period_loss"].append(loss_fake_mp.item())
                losses["generator/multi_resolution_loss"].append(loss_fake_mr.item())
                losses["generator/fm_mp_loss"].append(loss_fmap_mp.item())
                losses["generator/fm_mr_loss"].append(loss_fmap_mr.item())
                losses["generator/mel_loss"].append(loss_mel.item())
                losses["generator/total_loss"].append(loss_g.item())

                if step % 10 == 0:
                    t.set_postfix({
                        "D": loss_d.item(),
                        "G": loss_g.item()
                    })

                if step % 100 == 0:
                    record = {}
                    for key in losses:
                        record[key] = np.mean(losses[key])
                        losses[key] = []
                    fabric.log_dict(record, step)

                if step % 1000 == 0:
                    fabric.logger.experiment.add_audio("train/audio_fake", audio_fake[0].data.cpu(), step, 24000)

                step += 1

        scheduler_d.step()
        scheduler_g.step()


if __name__ == "__main__":
    main()