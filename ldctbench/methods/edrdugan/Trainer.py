"""
Trainer cho EDR-DUGAN
======================
Kế thừa toàn bộ DUGAN Trainer (dual discriminators, cutmix, gradient domain)
Chỉ thay đổi: Model (Generator) từ RED-CNN → EDR-REDCNN.

So sánh với DUGAN gốc:
  - Generator: RED-CNN → EDR-REDCNN (FixedSobelLayer + EdgeDilatedResidualBlock)
  - D_im: UNet + SpectralNorm (giữ nguyên)
  - D_grad: UNet + SpectralNorm trên Sobel(ảnh) (giữ nguyên)
  - G loss: lam_adv × (adv_img + adv_grad) + lam_px_im × MSE + lam_px_grad × L1(Sobel)
            (giữ nguyên — DUGAN đã có gradient loss mạnh, không cần thêm SobelEdgeLoss)
  - Gradient clipping: THÊM MỚI — clip_grad_norm_(G, 1.0) để ổn định training

Tham chiếu:
  Z. Huang et al., "DU-GAN", IEEE Trans. Instrum. Meas., 2022.
"""

import copy
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from ldctbench import utils
from ldctbench.methods.base import BaseTrainer

from .network import Model, UNet
from ldctbench.methods.dugan.utils import (
    SobelOperator,
    cutmix,
    ls_gan,
    mask_src_tgt,
    turn_on_spectral_norm,
)


class Trainer(BaseTrainer):
    """Trainer cho EDR-DUGAN.

    Dual-domain GAN với EDR-REDCNN làm Generator:
      - D_im: UNet discriminator trên image domain
      - D_grad: UNet discriminator trên gradient domain (Sobel)
      - G: EDR-REDCNN (FixedSobelLayer + EdgeDilatedResidualBlock + RED-CNN backbone)
    """

    def __init__(self, args: Namespace, device: torch.device):
        """Khởi tạo Trainer.

        Parameters
        ----------
        args : Namespace
            Cần có: lr, n_d_train, lam_adv, lam_px_im, lam_px_grad,
                    lam_cutmix, cutmix_prob, cutmix_warmup_iter,
                    num_edge_blocks, use_sobel_input.
        device : torch.device
            GPU/CPU device.
        """
        super().__init__(args, device)

        # ── Generator: EDR-REDCNN ─────────────────────────────────────────────────
        self.model = Model(args).to(self.dev)

        # ── Discriminators: UNet × 2 (image + gradient domain) ───────────────────
        self.im_discriminator = UNet(
            repeat_num=6,
            use_discriminator=True,
            conv_dim=64,
            use_sigmoid=False,
        ).to(self.dev)
        self.im_discriminator = turn_on_spectral_norm(self.im_discriminator)
        self.grad_discriminator = copy.deepcopy(self.im_discriminator)

        if isinstance(self.args.devices, list):
            self.model = nn.DataParallel(self.model, device_ids=self.args.devices)
            self.im_discriminator = nn.DataParallel(
                self.im_discriminator, device_ids=self.args.devices
            )
            self.grad_discriminator = nn.DataParallel(
                self.grad_discriminator, device_ids=self.args.devices
            )

        # ── Optimizers ────────────────────────────────────────────────────────────
        self.g_optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.im_d_optimizer = optim.Adam(
            self.im_discriminator.parameters(), self.args.lr
        )
        self.grad_d_optimizer = optim.Adam(
            self.grad_discriminator.parameters(), self.args.lr
        )

        # ── Loss (giữ nguyên DUGAN) ───────────────────────────────────────────────
        self.criterion = ls_gan

        # ── Sobel (cho gradient discriminator) ───────────────────────────────────
        self.sobel = SobelOperator().to(self.dev)

        # ── CutMix schedule ───────────────────────────────────────────────────────
        max_iter_upper = (
            self.args.max_iterations
            + self.args.iterations_before_val
            - (self.args.max_iterations % self.args.iterations_before_val)
        )
        self.apply_cutmix_prob = torch.rand(max_iter_upper)

        # ── Logging ───────────────────────────────────────────────────────────────
        self.losses = utils.metrics.Losses(
            self.dataloader,
            [
                "D loss (img)",
                "D loss (grad)",
                "G loss (pix)",
                "G loss (grad)",
                "G loss",
            ],
        )

    def warmup(self):
        return min(
            self.iteration * self.args.cutmix_prob / self.args.cutmix_warmup_iter,
            self.args.cutmix_prob,
        )

    def train_discriminator(self, discriminator, optimizer, inputs, targets, fakes):
        """Train một discriminator theo LS-GAN với CutMix (giữ nguyên DUGAN)."""
        optimizer.zero_grad()
        
        scaler = getattr(self, "scaler", None)
        if scaler is None:
            self.scaler = torch.cuda.amp.GradScaler()
            scaler = self.scaler

        with torch.cuda.amp.autocast():
            real_enc, real_dec = discriminator(targets)
            fake_enc, fake_dec = discriminator(fakes.detach())
            source_enc, source_dec = discriminator(inputs)

            d_loss = (
                self.criterion(real_enc, 1.0)
                + self.criterion(real_dec, 1.0)
                + self.criterion(fake_enc, 0.0)
                + self.criterion(fake_dec, 0.0)
                + self.criterion(source_enc, 0.0)
                + self.criterion(source_dec, 0.0)
            )

            apply_cutmix = self.apply_cutmix_prob[self.iteration - 1] < self.warmup()
            if apply_cutmix:
                mask = cutmix(real_dec.size()).to(real_dec)
                cutmix_enc, cutmix_dec = discriminator(
                    mask_src_tgt(targets, fakes.detach(), mask)
                )
                cutmix_disc_loss = self.criterion(cutmix_enc, 0.0) + self.criterion(
                    cutmix_dec, mask
                )
                cr_loss = F.mse_loss(cutmix_dec, mask_src_tgt(real_dec, fake_dec, mask))
                d_loss += cutmix_disc_loss + cr_loss * self.args.lam_cutmix

        scaler.scale(d_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return d_loss.data

    def train_step(self, batch):
        """Training step cho EDR-DUGAN.

        Giữ nguyên DUGAN training loop.
        Thêm gradient clipping cho Generator để ổn định training.
        """
        inputs, targets = batch["x"], batch["y"]
        self.iteration += 1

        # [THÊM MỚI] Dùng AMP (Automatic Mixed Precision) để giảm VRAM và tăng tốc
        scaler = getattr(self, "scaler", None)
        if scaler is None:
            self.scaler = torch.cuda.amp.GradScaler()

        with torch.cuda.amp.autocast():
            gen_full_dose = self.model(inputs)
            grad_gen_full_dose = self.sobel(gen_full_dose)
            grad_low_dose = self.sobel(inputs)
            grad_full_dose = self.sobel(targets)

        # ── Train Image Discriminator ─────────────────────────────────────────────
        for _ in range(self.args.n_d_train):
            im_d_loss = self.train_discriminator(
                self.im_discriminator,
                self.im_d_optimizer,
                inputs,
                targets,
                gen_full_dose,
            )

        # ── Train Generator ───────────────────────────────────────────────────────
        self.g_optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # Image adversarial loss
            img_gen_enc, img_gen_dec = self.im_discriminator(gen_full_dose)
            img_gen_loss = self.criterion(img_gen_enc, 1.0) + self.criterion(
                img_gen_dec, 1.0
            )

            # Train Gradient Discriminator + get gradient adversarial loss
            grad_d_loss = self.train_discriminator(
                self.grad_discriminator,
                self.grad_d_optimizer,
                grad_low_dose,
                grad_full_dose,
                grad_gen_full_dose,
            )
            grad_gen_enc, grad_gen_dec = self.grad_discriminator(grad_gen_full_dose)
            grad_gen_loss = self.criterion(grad_gen_enc, 1.0) + self.criterion(
                grad_gen_dec, 1.0
            )

            # Pixelwise losses
            pix_loss = F.mse_loss(gen_full_dose, targets)
            grad_loss = F.l1_loss(grad_gen_full_dose, grad_full_dose)

            # Tổng G loss
            total_loss = (
                grad_gen_loss * self.args.lam_adv
                + img_gen_loss * self.args.lam_adv
                + pix_loss * self.args.lam_px_im
                + grad_loss * self.args.lam_px_grad
            )

        self.scaler.scale(total_loss).backward()
        
        # Unscale trước khi clip_grad_norm_
        self.scaler.unscale_(self.g_optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.scaler.step(self.g_optimizer)
        self.scaler.update()

        self.losses.push(
            loss={
                "D loss (img)": im_d_loss,
                "D loss (grad)": grad_d_loss,
                "G loss (pix)": pix_loss.data,
                "G loss (grad)": grad_loss.data,
                "G loss": total_loss.data,
            },
            phase="train",
        )

    @torch.no_grad()
    def val_step(self, batch_idx, batch):
        """Validation step (giữ nguyên DUGAN)."""
        inputs, targets = batch["x"], batch["y"]
        outputs = self.model(inputs)
        self.metrics.push(targets, outputs)

        im_d = self.im_discriminator(outputs)[1]
        grad_d = self.grad_discriminator(self.sobel(outputs))[1]

        if batch_idx < self.args.valsamples:
            self.log_wandb_images(
                {
                    "low dose": inputs,
                    "high dose": targets,
                    "prediction": outputs,
                    "D_im(prediction)": im_d,
                    "D_grad(prediction)": grad_d,
                }
            )

    def train(self):
        self.model.train()
        self.im_discriminator.train()
        self.grad_discriminator.train()
        for batch in tqdm(self.dataloader["train"]):
            batch = {
                k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()
            }
            self.train_step(batch)

        self.losses.summarize("train")

    def validate(self):
        self.images = {}
        self.model.eval()
        self.im_discriminator.eval()
        self.grad_discriminator.eval()
        for batch_idx, batch in enumerate(tqdm(self.dataloader["val"])):
            batch = {
                k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()
            }
            self.val_step(batch_idx, batch)

        self.losses.summarize("val")
        self.metrics.summarize()
