import math

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
import transformers

from pytorch_optimizer import Prodigy

from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.feature_extractors import FeatureExtractor
from vocos.heads import FourierHead
from vocos.helpers import plot_spectrogram_to_numpy
from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss
from vocos.models import Backbone
from vocos.modules import safe_log

class VocosTraining(pl.LightningModule):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        sample_rate: int,
        initial_learning_rate: float,
        num_warmup_steps: int = 0,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head"])

        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()

        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)

        self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff

        self.automatic_optimization = False
        self.validation_step_outputs = []

    def configure_optimizers(self):
        disc_params = [
            {"params": self.multiperioddisc.parameters()},
            {"params": self.multiresddisc.parameters()}
        ]
        gen_params = [
            {"params": self.backbone.parameters()},
            {"params": self.head.parameters()}
        ]

        opt_disc = Prodigy(disc_params, lr=1.0, weight_decay=0.01, safeguard_warmup=True, bias_correction=True)
        opt_gen = Prodigy(gen_params, lr=1.0, weight_decay=0.01, safeguard_warmup=True, bias_correction=True)

        max_steps = self.trainer.max_steps // 2  # Max steps per optimizer
        scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_disc, T_max=max_steps
        )
        scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_gen, T_max=max_steps
        )

        return (
            [opt_disc, opt_gen],
            [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
        )

    def training_step(self, batch, batch_idx):
        opt_disc, opt_gen = self.optimizers()
        
        audio_input = batch
        audio_hat = self(audio_input)

        self.toggle_optimizer(opt_disc)
        opt_disc.zero_grad()

        real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat.detach(), flg_train=True)
        real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat.detach(), flg_train=True)

        loss_mp, _, _ = self.disc_loss(disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp)
        loss_mrd, _, _ = self.disc_loss(disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd)
        loss_disc = loss_mp + loss_mrd

        self.log("discriminator/total", loss_disc, prog_bar=True)
        self.log("discriminator/multi_period_loss", loss_mp)
        self.log("discriminator/multi_res_loss", loss_mrd)

        if self.global_step % 1000 == 0 and self.global_rank == 0:
            self.logger.experiment.add_audio(
                "train/audio_in", audio_input[0].data.cpu(), self.global_step, self.hparams.sample_rate
            )
            self.logger.experiment.add_audio(
                "train/audio_pred", audio_hat[0].data.cpu(), self.global_step, self.hparams.sample_rate
            )
            with torch.no_grad():
                mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
                mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
            self.logger.experiment.add_image(
                "train/mel_target",
                plot_spectrogram_to_numpy(mel.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "train/mel_pred",
                plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )

        self.manual_backward(loss_disc)
        opt_disc.step()
        self.untoggle_optimizer(opt_disc)

        self.toggle_optimizer(opt_gen)
        opt_gen.zero_grad()

        _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(y=audio_input, y_hat=audio_hat)
        _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(y=audio_input, y_hat=audio_hat)

        loss_gen_mp, _ = self.gen_loss(disc_outputs=gen_score_mp)
        loss_gen_mrd, _ = self.gen_loss(disc_outputs=gen_score_mrd)
        loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp)
        loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd)

        self.log("generator/multi_period_loss", loss_gen_mp)
        self.log("generator/multi_res_loss", loss_gen_mrd)
        self.log("generator/feature_matching_mp", loss_fm_mp)
        self.log("generator/feature_matching_mrd", loss_fm_mrd)

        mel_loss = self.melspec_loss(audio_hat, audio_input)
        loss_gen = loss_gen_mp + loss_gen_mrd + loss_fm_mp + loss_fm_mrd + mel_loss

        self.log("generator/total_loss", loss_gen, prog_bar=True)
        self.log("mel_loss_coeff", self.mel_loss_coeff)
        self.log("generator/mel_loss", mel_loss)

        self.manual_backward(loss_gen)
        opt_gen.step()
        self.untoggle_optimizer(opt_gen)

    def on_validation_epoch_start(self):
        if self.hparams.evaluate_utmos:
            from metrics.UTMOS import UTMOSScore

            if not hasattr(self, "utmos_model"):
                self.utmos_model = UTMOSScore(device=self.device)

    def validation_step(self, batch, batch_idx):
        audio_input = batch
        audio_hat = self(audio_input)

        audio_16_khz = torchaudio.functional.resample(audio_input, orig_freq=self.hparams.sample_rate, new_freq=16000)
        audio_hat_16khz = torchaudio.functional.resample(audio_hat, orig_freq=self.hparams.sample_rate, new_freq=16000)

        if self.hparams.evaluate_periodicty:
            from metrics.periodicity import calculate_periodicity_metrics

            periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(audio_16_khz, audio_hat_16khz)
        else:
            periodicity_loss = pitch_loss = f1_score = 0

        if self.hparams.evaluate_utmos:
            utmos_score = self.utmos_model.score(audio_hat_16khz.unsqueeze(1)).mean()
        else:
            utmos_score = torch.zeros(1, device=self.device)

        if self.hparams.evaluate_pesq:
            from pesq import pesq

            pesq_score = 0
            for ref, deg in zip(audio_16_khz.cpu().numpy(), audio_hat_16khz.cpu().numpy()):
                pesq_score += pesq(16000, ref, deg, "wb", on_error=1)
            pesq_score /= len(audio_16_khz)
            pesq_score = torch.tensor(pesq_score)
        else:
            pesq_score = torch.zeros(1, device=self.device)

        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        total_loss = mel_loss + (5 - utmos_score) + (5 - pesq_score)

        losses = {
            "val_loss": total_loss,
            "mel_loss": mel_loss,
            "utmos_score": utmos_score,
            "pesq_score": pesq_score,
            "periodicity_loss": periodicity_loss,
            "pitch_loss": pitch_loss,
            "f1_score": f1_score,
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
        }
        self.validation_step_outputs.append(losses)
        return losses
    
    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            *_, audio_in, audio_pred = self.validation_step_outputs[0].values()
            self.logger.experiment.add_audio(
                "val_in", audio_in.data.cpu().numpy(), self.global_step, self.hparams.sample_rate
            )
            self.logger.experiment.add_audio(
                "val_pred", audio_pred.data.cpu().numpy(), self.global_step, self.hparams.sample_rate
            )
            mel_target = safe_log(self.melspec_loss.mel_spec(audio_in))
            mel_hat = safe_log(self.melspec_loss.mel_spec(audio_pred))
            self.logger.experiment.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(mel_target.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "val_mel_hat",
                plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        mel_loss = torch.stack([x["mel_loss"] for x in self.validation_step_outputs]).mean()
        utmos_score = torch.stack([x["utmos_score"] for x in self.validation_step_outputs]).mean()
        pesq_score = torch.stack([x["pesq_score"] for x in self.validation_step_outputs]).mean()
        periodicity_loss = np.array([x["periodicity_loss"] for x in self.validation_step_outputs]).mean()
        pitch_loss = np.array([x["pitch_loss"] for x in self.validation_step_outputs]).mean()
        f1_score = np.array([x["f1_score"] for x in self.validation_step_outputs]).mean()

        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val/mel_loss", mel_loss, sync_dist=True)
        self.log("val/utmos_score", utmos_score, sync_dist=True)
        self.log("val/pesq_score", pesq_score, sync_dist=True)
        self.log("val/periodicity_loss", periodicity_loss, sync_dist=True)
        self.log("val/pitch_loss", pitch_loss, sync_dist=True)
        self.log("val/f1_score", f1_score, sync_dist=True)

    @property
    def global_step(self):
        """
        Override global_step so that it returns the total number of batches processed
        """
        return self.trainer.fit_loop.epoch_loop.total_batch_idx
    
    def on_train_batch_start(self, batch, batch_idx):
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        def mel_loss_coeff_decay(current_step, num_cycles=0.5):
            max_steps = self.trainer.max_steps // 2
            if current_step < self.hparams.num_warmup_steps:
                return 1.0
            progress = float(current_step - self.hparams.num_warmup_steps) / float(
                max(1, max_steps - self.hparams.num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        if self.hparams.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * mel_loss_coeff_decay(self.global_step + 1)

    def forward(self, audio_input):
        features = self.feature_extractor(audio_input)
        x = self.backbone(features)
        audio_output = self.head(x)
        return audio_output