"""
CombinedLoss cho EDR-ResNet
============================
Giống hệt edrrednet/loss.py — Charbonnier + SobelEdgeLoss.
Copy nguyên để tránh circular import.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1 variant).

    L = sqrt((pred - target)^2 + eps^2)
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class SobelEdgeLoss(nn.Module):
    """L1 Loss giữa Sobel gradient của pred và target.

    L_Sobel = ||Sobel(pred) - Sobel(target)||_1
    """

    def __init__(self):
        super().__init__()
        kernels = torch.tensor(
            [
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],    # Horizontal
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],    # Vertical
                [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],    # Diagonal 45°
                [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],    # Diagonal 135°
            ],
            dtype=torch.float32,
        ).unsqueeze(1)  # (4, 1, 3, 3)
        self.register_buffer("weight", kernels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad_pred   = F.conv2d(pred,   self.weight, padding=1)
        grad_target = F.conv2d(target, self.weight, padding=1)
        return F.l1_loss(grad_pred, grad_target)


class CombinedLoss(nn.Module):
    """Loss kết hợp: Charbonnier + alpha * SobelEdgeLoss.

    Parameters
    ----------
    alpha : float
        Trọng số SobelEdgeLoss (default=0.1).
    beta : float
        Trọng số PerceptualLoss VGG — ablation only (default=0.0).
    gamma : float
        Trọng số HU Loss (default=0.0).
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.0, gamma: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        self.charbonnier = CharbonnierLoss()
        self.sobel_loss  = SobelEdgeLoss()

        self.vgg = None
        if beta > 0:
            self._init_vgg()

    def _init_vgg(self):
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features[:16]
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg
        except Exception as e:
            print(f"[WARNING] Failed to load VGG: {e}. beta will be ignored.")
            self.beta = 0.0

    def _perceptual_loss(self, pred, target):
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        self.vgg = self.vgg.to(pred.device)
        pred_3ch   = pred.repeat(1, 3, 1, 1)
        target_3ch = target.repeat(1, 3, 1, 1)
        return F.l1_loss(self.vgg(pred_3ch), self.vgg(target_3ch))

    def forward(self, pred, target, mean=None, std=None):
        """Tính tổng loss.

        Returns
        -------
        total : Tensor (scalar)
        components : dict
        """
        l_charb = self.charbonnier(pred, target)
        l_sobel = self.sobel_loss(pred, target)
        total   = l_charb + self.alpha * l_sobel

        l_perceptual = torch.tensor(0.0, device=pred.device)
        if self.beta > 0:
            l_perceptual = self._perceptual_loss(pred, target)
            total = total + self.beta * l_perceptual

        l_hu = torch.tensor(0.0, device=pred.device)
        if self.gamma > 0 and mean is not None and std is not None:
            pred_hu   = pred   * std + mean
            target_hu = target * std + mean
            l_hu  = F.l1_loss(pred_hu, target_hu)
            total = total + self.gamma * l_hu

        components = {
            "loss/total":       total.item(),
            "loss/charbonnier": l_charb.item(),
            "loss/sobel":       l_sobel.item(),
            "loss/perceptual":  l_perceptual.item(),
            "loss/hu":          l_hu.item(),
        }
        return total, components
