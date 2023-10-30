from tqdm import trange
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import TensorBoardLogger

from pytorch_optimizer import Prodigy

from vocos.dataset import VocosDataset
from vocos.feature_extractors import MelSpectrogramFeatures
from vocos.models import VocosBackbone
from vocos.heads import ISTFTHead
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss

BATCH_SIZE = 16
NUM_SAMPLES = 16384
NUM_WORKERS = 8
NUM_EPOCHS = 1000
LEARNING_RATE = 5e-4
MAX_STEPS = 1_000_000
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
    seed_everything(4444)

    torch.set_float32_matmul_precision("high")

    logger = TensorBoardLogger(root_dir="logs")

    fabric = Fabric(accelerator="auto", loggers=logger)
    fabric.launch()

    dataset = VocosDataset(filelist_path="filelist.train", sampling_rate=24000, num_samples=NUM_SAMPLES, train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    generator = Generator()
    discriminator = Discriminator()

    discriminator_loss = DiscriminatorLoss()
    generator_loss = GeneratorLoss()
    feature_loss = FeatureMatchingLoss()
    mel_loss = MelSpecReconstructionLoss()

    mr_loss_coeff = 0.1
    mel_loss_coeff = 45

    if USE_PRODIGY:
        optimizer_g = Prodigy(generator.parameters(), lr=1.0, betas=(0.8, 0.9), weight_decay=0.01, safeguard_warmup=True, bias_correction=True)
        optimizer_d = Prodigy(discriminator.parameters(), lr=1.0, betas=(0.8, 0.9), weight_decay=0.01, safeguard_warmup=True, bias_correction=True)
    else:
        optimizer_g = AdamW(generator.parameters(), lr=LEARNING_RATE, betas=(0.8, 0.9))
        optimizer_d = AdamW(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.8, 0.9))

    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=MAX_STEPS)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=MAX_STEPS)

    generator, optimizer_g = fabric.setup(generator, optimizer_g)
    discriminator, optimizer_d = fabric.setup(discriminator, optimizer_d)

    dataloader = fabric.setup_dataloaders(dataloader)

    fabric.to_device(discriminator_loss)
    fabric.to_device(generator_loss)
    fabric.to_device(feature_loss)
    fabric.to_device(mel_loss)

    step = 0
    for epoch in range(NUM_EPOCHS):
        with trange(len(dataloader)) as t:
            for i, data in zip(t, dataloader):
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

                fabric.log_dict({
                    "discriminator/total_loss": loss_d,
                    "discriminator/multi_period_loss": loss_mp,
                    "discriminator/multi_resolution_loss": loss_mr
                }, step)

                fabric.backward(loss_d)
                optimizer_d.step()

                optimizer_g.zero_grad()

                _, fake_score_mp, fmap_real_mp, fmap_fake_mp = discriminator.multi_period(audio_real, audio_fake)
                _, fake_score_mr, fmap_real_mr, fmap_fake_mr = discriminator.multi_resolution(audio_real, audio_fake)

                loss_fake_mp, list_loss_fake_mp = generator_loss(fake_score_mp)
                loss_fake_mr, list_loss_fake_mr = generator_loss(fake_score_mr)
                loss_fake_mp = loss_fake_mp / len(list_loss_fake_mp)
                loss_fake_mr = loss_fake_mr / len(list_loss_fake_mr)
                loss_fmap_mp = feature_loss(fmap_real_mp, fmap_fake_mp)
                loss_fmap_mr = feature_loss(fmap_real_mr, fmap_fake_mr)

                loss_mel = mel_loss(audio_fake, audio_real)
                loss_g = loss_fake_mp + mr_loss_coeff * loss_fake_mr + loss_fmap_mp + mr_loss_coeff * loss_fmap_mr + mel_loss_coeff * loss_mel

                fabric.log_dict({
                    "generator/multi_period_loss": loss_fake_mp,
                    "generator/multi_resolution_loss": loss_fake_mr,
                    "generator/feature_matching_mp": loss_fmap_mp,
                    "generator/feature_matching_mr": loss_fmap_mr,
                    "generator/mel_loss": loss_mel,
                    "generator/total_loss": loss_g
                }, step)

                fabric.backward(loss_g)
                optimizer_g.step()

                t.set_postfix({
                    "D": loss_d.item(),
                    "G": loss_g.item()
                })

                if step % 100 == 0:
                    fabric.logger.experiment.add_audio("train/audio_fake", audio_fake[0].data.cpu(), step, 24000)

                step += 1

                scheduler_d.step()
                scheduler_g.step()


if __name__ == "__main__":
    main()