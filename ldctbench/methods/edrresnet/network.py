"""
EDR-ResNet: Edge-Dilated Residual ResNet
=========================================
Kế thừa kiến trúc ResNet (noise-subtraction) và bổ sung:
  1. FixedSobelLayer  — non-trainable edge extractor (4 hướng: H, V, Diag45, Diag135)
  2. EdgeDilatedResidualBlock — dilated conv (rate=2,3) + residual ở giữa backbone

Điểm khác biệt vs EDR-CNN10 / EDR-REDNet:
  - ResNet học NOISE MAP, không học ảnh sạch trực tiếp
  - output = input - predicted_noise  (noise subtraction)
  - n_channels = 128 (rộng hơn CNN10's 64)
  - EdgeBlock đặt ở giữa (sau ResBlock thứ 5)

Tham khảo:
  - ResNet gốc: Missert et al., ICIFX 2018
  - FixedSobelLayer: EDR-REDNet (dự án này)
  - Dilated Conv: Yu & Koltun, ICLR 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace


# ──────────────────────────────────────────────────────────────────────────────
# 1. Fixed Sobel Layer (copy từ edrrednet/network.py)
# ──────────────────────────────────────────────────────────────────────────────

class FixedSobelLayer(nn.Module):
    """Non-trainable layer trích xuất edge map theo 4 hướng.

    Weights Sobel hoàn toàn cố định (register_buffer), không được cập nhật
    trong quá trình train. Output là 4-channel edge map (H, V, Diag45, Diag135).

    Input:  (B, 1, H, W)
    Output: (B, 4, H, W)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=1)


# ──────────────────────────────────────────────────────────────────────────────
# 2. EdgeDilatedResidualBlock (copy từ edrrednet/network.py, ch=128)
# ──────────────────────────────────────────────────────────────────────────────

class EdgeDilatedResidualBlock(nn.Module):
    """Residual block với dilated convolution để mở rộng receptive field.

    Cấu trúc:
        x → Conv(ch, ch, 3, dilation=d) → ReLU → Conv(ch, ch, 3, dilation=1) → + x

    Input/Output: (B, ch, H, W)
    """

    def __init__(self, ch: int, dilation: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)


# ──────────────────────────────────────────────────────────────────────────────
# 3. ResBlock gốc (giữ nguyên từ resnet/network.py)
# ──────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Single Residual block — giữ nguyên từ ResNet gốc.

    Each block consists of Conv → BN → ReLU → GroupConv → BN → ReLU → Conv + skip
    """

    def __init__(self, ch: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, groups=8),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + x


# ──────────────────────────────────────────────────────────────────────────────
# 4. EDR-ResNet Model
# ──────────────────────────────────────────────────────────────────────────────

class Model(nn.Module):
    """EDR-ResNet: ResNet + FixedSobelLayer + EdgeDilatedResidualBlock.

    Kiến trúc:
        in_conv(1→128, k=9)
            ↓
        [FixedSobelLayer → edge_proj(4→128)] inject vào features
            ↓
        ResBlock 1 → 2 → 3 → 4 → 5
            ↓
        EdgeDilatedResidualBlock(128, dilation=2)
        EdgeDilatedResidualBlock(128, dilation=3)
            ↓
        ResBlock 6 → 7 → 8 → 9 → 10
            ↓
        out_conv(128→1, k=3)
            ↓
        output = original_input - predicted_noise   ← NOISE SUBTRACTION

    Parameters
    ----------
    args : Namespace
        Cần có: use_sobel_input (bool), num_edge_blocks (int).
    n_channels : int
        Số channels (default=128, giống ResNet gốc).
    n_blocks : int
        Tổng số ResBlocks (default=10, giống ResNet gốc).
    """

    def __init__(self, args: Namespace, n_channels: int = 128, n_blocks: int = 10):
        super().__init__()

        self.n_blocks    = n_blocks
        self.use_sobel   = getattr(args, "use_sobel_input", True)
        n_edge_blocks    = getattr(args, "num_edge_blocks", 2)
        self.split_point = n_blocks // 2   # Đặt EdgeBlocks ở giữa (sau block 5)

        # ── Backbone gốc ──────────────────────────────────────────────────────
        self.in_conv  = nn.Conv2d(1, n_channels, 9, padding=4)
        self.out_conv = nn.Conv2d(n_channels, 1, 3, padding=1)
        self.blocks   = nn.ModuleList([ResBlock(n_channels) for _ in range(n_blocks)])

        # ── FixedSobelLayer + edge_proj ───────────────────────────────────────
        if self.use_sobel:
            self.sobel     = FixedSobelLayer()
            self.edge_proj = nn.Conv2d(4, n_channels, 1)  # 4 → 128

        # ── EdgeDilatedResidualBlocks (bottleneck giữa network) ───────────────
        dilations = [2, 3][:n_edge_blocks]
        self.edge_blocks = nn.ModuleList(
            [EdgeDilatedResidualBlock(n_channels, d) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x  # Lưu để noise subtraction ở cuối

        # ── in_conv ───────────────────────────────────────────────────────────
        feat = self.in_conv(x)

        # ── Inject Sobel edge features ────────────────────────────────────────
        if self.use_sobel:
            edges = self.sobel(original_x)       # (B, 4, H, W)
            feat  = feat + self.edge_proj(edges) # (B, 128, H, W)

        # ── ResBlocks 1~5 ─────────────────────────────────────────────────────
        for i in range(self.split_point):
            feat = self.blocks[i](feat)

        # ── EdgeDilatedResidualBlocks (bottleneck) ────────────────────────────
        for eb in self.edge_blocks:
            feat = eb(feat)

        # ── ResBlocks 6~10 ────────────────────────────────────────────────────
        for i in range(self.split_point, self.n_blocks):
            feat = self.blocks[i](feat)

        # ── out_conv + noise subtraction ──────────────────────────────────────
        noise = self.out_conv(feat)
        return original_x - noise
