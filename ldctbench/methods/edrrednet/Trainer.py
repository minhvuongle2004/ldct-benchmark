"""
Trainer cho EDR-REDNet
=======================
Override train_step của BaseTrainer để dùng CombinedLoss thay vì MSELoss.
Giữ nguyên toàn bộ data pipeline, validation, checkpoint từ BaseTrainer.
"""

from argparse import Namespace

import torch
import torch.nn as nn
import wandb

from ldctbench.methods.base import BaseTrainer
from ldctbench.utils.training_utils import setup_optimizer

from .loss import CombinedLoss
from .network import Model


class Trainer(BaseTrainer):
    """Trainer cho mô hình EDR-REDNet.

    Thay MSELoss bằng CombinedLoss (Charbonnier + SobelEdgeLoss).
    Log từng thành phần loss riêng để ablation study.
    """

    def __init__(self, args: Namespace, device: torch.device):
        """Khởi tạo Trainer.

        Parameters
        ----------
        args : Namespace
            Cần có: loss_alpha, loss_beta, loss_gamma, num_edge_blocks.
        device : torch.device
            GPU/CPU device.
        """
        # Gọi BaseTrainer để thiết lập dataloader, logger, device, vòng lặp train
        super().__init__(args, device)

        # ── CombinedLoss thay thế MSELoss ───────────────────────────────────────
        alpha = getattr(args, "loss_alpha", 0.1)
        beta = getattr(args, "loss_beta", 0.0)
        gamma = getattr(args, "loss_gamma", 0.0)

        self.criterion = CombinedLoss(alpha=alpha, beta=beta, gamma=gamma).to(self.dev)

        # ── Model EDR-REDNet ─────────────────────────────────────────────────────
        self.model = Model(args).to(self.dev)

        if isinstance(self.args.devices, list):
            self.model = nn.DataParallel(self.model, device_ids=self.args.devices)

        # ── Optimizer ────────────────────────────────────────────────────────────
        self.optimizer = setup_optimizer(args, self.model.parameters())

    def train_step(self, batch):
        """Override train_step để dùng CombinedLoss.

        BaseTrainer gọi hàm này mỗi iteration với batch là dict {"x": tensor, "y": tensor}.
        Các tensor đã được BaseTrainer.train() đưa lên device rồi.
        """
        ldct = batch["x"]   # ảnh LDCT (low-dose)
        ndct = batch["y"]   # ảnh NDCT (ground truth)

        self.optimizer.zero_grad()
        pred = self.model(ldct)

        # CombinedLoss trả về (total_loss, components_dict)
        total_loss, components = self.criterion(pred, ndct)

        total_loss.backward()
        self.optimizer.step()

        # Cập nhật iteration counter và loss tracker (BaseTrainer dùng cho log)
        self.iteration += 1
        self.losses.push(total_loss, "train")

        # Log từng thành phần để theo dõi ablation study
        wandb.log(components, step=self.iteration)

    @torch.no_grad()
    def val_step(self, batch_idx, batch):
        """Override val_step để unpack tuple từ CombinedLoss.

        base.val_step expect criterion trả về scalar tensor,
        nhưng CombinedLoss trả về (total_loss, components_dict).
        """
        inputs = batch["x"]
        targets = batch["y"]

        outputs = self.model(inputs)

        # Unpack tuple — chỉ dùng total_loss cho val metrics
        total_loss, _ = self.criterion(outputs, targets)

        self.losses.push(total_loss, "val")
        self.metrics.push(targets, outputs)
        if batch_idx < self.args.valsamples:
            self.log_wandb_images(
                {"low dose": inputs, "prediction": outputs, "high dose": targets}
            )

