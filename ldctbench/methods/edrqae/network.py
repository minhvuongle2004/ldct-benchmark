"""
EDR-QAE: Edge-Dilated Residual Quadratic Autoencoder
======================================================
Kế thừa kiến trúc QAE (Quadratic Autoencoder) và bổ sung:
  1. FixedSobelLayer  — non-trainable edge extractor (4 hướng)
  2. EdgeDilatedResidualBlock — dilated conv (rate=2,3) tại bottleneck

Điểm khác biệt vs EDR-CNN10 / EDR-ResNet:
  - QAE dùng QuadConv (bậc 2): output = (W_r·x + b_r)(W_g·x + b_g) + W_b·x² + b_b
  - Chỉ 15 channels — lightweight nhất trong bộ 3
  - Encoder-decoder với skip connections (x4→decoder[0], x2→decoder[2])
  - encoder[4] dùng 'valid' padding → spatial size giảm 2px (36→34)
  - output = decoder[4](x9) + input (residual từ input gốc)

Tham khảo:
  - QAE gốc: Fan et al., IEEE TMI 2020
  - FixedSobelLayer: EDR-REDNet (dự án này)
  - Dilated Conv: Yu & Koltun, ICLR 2016
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from scipy.stats import truncnorm


# ──────────────────────────────────────────────────────────────────────────────
# 1. QuadConv / QuadDeconv — copy nguyên từ qae/network.py
# ──────────────────────────────────────────────────────────────────────────────

class QuadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding="valid"):
        super(QuadConv, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        self.padding = padding
        self.valid_padding = ["valid", "same"]
        if self.padding not in self.valid_padding:
            raise ValueError(
                f"Padding must be one of {self.valid_padding}. Got {self.padding} instead!"
            )

        self.W_r = nn.Parameter(
            torch.Tensor(
                truncnorm.rvs(-2, 2, scale=0.1, size=[out_channels, in_channels, *kernel_size])
            )
        )
        self.W_g = nn.Parameter(
            torch.zeros(size=[out_channels, in_channels, *kernel_size], dtype=torch.float32)
        )
        self.W_b = nn.Parameter(
            torch.zeros(size=[out_channels, in_channels, *kernel_size], dtype=torch.float32)
        )
        self.b_r = nn.Parameter(torch.zeros(size=[out_channels], dtype=torch.float32))
        self.b_g = nn.Parameter(torch.ones(size=[out_channels], dtype=torch.float32))
        self.b_b = nn.Parameter(torch.zeros(size=[out_channels], dtype=torch.float32))

    def forward(self, x):
        # Clip input to prevent activation explosion in quadratic terms
        x_c = torch.clamp(x, min=-10.0, max=10.0)
        x1 = F.conv2d(x_c,     self.W_r, self.b_r, stride=1, padding=self.padding)
        x2 = F.conv2d(x_c,     self.W_g, self.b_g, stride=1, padding=self.padding)
        x3 = F.conv2d(x_c * x_c, self.W_b, self.b_b, stride=1, padding=self.padding)
        return x1 * x2 + x3


class QuadDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding="valid"):
        super(QuadDeconv, self).__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size

        self.padding = padding
        self.valid_padding = ["valid", "same"]
        if self.padding not in self.valid_padding:
            raise ValueError(
                f"Padding must be one of {self.valid_padding}. Got {self.padding} instead!"
            )

        self.W_r = nn.Parameter(
            torch.Tensor(
                truncnorm.rvs(-2, 2, scale=0.1,
                              size=[in_channels, out_channels, *self.kernel_size])
            )
        )
        self.W_g = nn.Parameter(
            torch.zeros(size=[in_channels, out_channels, *self.kernel_size], dtype=torch.float32)
        )
        self.W_b = nn.Parameter(
            torch.zeros(size=[in_channels, out_channels, *self.kernel_size], dtype=torch.float32)
        )
        self.b_r = nn.Parameter(torch.zeros(size=[out_channels], dtype=torch.float32))
        self.b_g = nn.Parameter(torch.ones(size=[out_channels], dtype=torch.float32))
        self.b_b = nn.Parameter(torch.zeros(size=[out_channels], dtype=torch.float32))

    def forward(self, x):
        pad = (
            int(np.ceil((self.kernel_size[0] - 1) / 2)) if self.padding == "same" else 0
        )
        # Clip input to prevent activation explosion in quadratic terms
        x_c = torch.clamp(x, min=-10.0, max=10.0)
        x1 = F.conv_transpose2d(x_c,     self.W_r, self.b_r, stride=1, padding=(pad, pad))
        x2 = F.conv_transpose2d(x_c,     self.W_g, self.b_g, stride=1, padding=(pad, pad))
        x3 = F.conv_transpose2d(x_c * x_c, self.W_b, self.b_b, padding=(pad, pad))
        return x1 * x2 + x3


# ──────────────────────────────────────────────────────────────────────────────
# 2. FixedSobelLayer (copy từ edrrednet/network.py)
# ──────────────────────────────────────────────────────────────────────────────

class FixedSobelLayer(nn.Module):
    """Non-trainable layer trích xuất edge map theo 4 hướng.

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
# 3. EdgeDilatedResidualBlock (copy từ edrrednet/network.py, ch=15)
# ──────────────────────────────────────────────────────────────────────────────

class EdgeDilatedResidualBlock(nn.Module):
    """Residual block với dilated convolution.

    Input/Output: (B, ch, H, W) — spatial size KHÔNG thay đổi (same padding)
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
# 4. EDR-QAE Model
# ──────────────────────────────────────────────────────────────────────────────

class Model(nn.Module):
    """EDR-QAE: QAE + FixedSobelLayer + EdgeDilatedResidualBlock.

    Kiến trúc:
        Encoder:
          x1 = ReLU(QuadConv(x, 1→15, same))
          x1 = x1 + edge_proj(FixedSobelLayer(x))   ← THÊM MỚI
          x2 = ReLU(QuadConv(x1, 15→15, same))
          x3 = ReLU(QuadConv(x2, 15→15, same))
          x4 = ReLU(QuadConv(x3, 15→15, same))
          x5 = ReLU(QuadConv(x4, 15→15, valid))     ← bottleneck (size -2px)

          x5 = EdgeDilatedResidualBlock(x5, d=2)    ← THÊM MỚI
          x5 = EdgeDilatedResidualBlock(x5, d=3)    ← THÊM MỚI

        Decoder:
          x6 = ReLU(QuadDeconv(x5, valid) + x4)
          x7 = ReLU(QuadDeconv(x6, same))
          x8 = ReLU(QuadDeconv(x7, same) + x2)
          x9 = ReLU(QuadDeconv(x8, same))
          out = QuadDeconv(x9, same) + input        ← residual từ input

    Parameters
    ----------
    args : Namespace
        Cần có: use_sobel_input (bool), num_edge_blocks (int).
    """

    def __init__(self, args: Namespace):
        super(Model, self).__init__()

        self.use_sobel  = getattr(args, "use_sobel_input", True)
        n_edge_blocks   = getattr(args, "num_edge_blocks", 2)
        ch = 15  # QAE luôn dùng 15 channels

        # ── Encoder (QuadConv, giữ nguyên từ QAE gốc) ─────────────────────────
        self.encoder = nn.ModuleList([
            QuadConv(1,  ch, 3, "same"),   # [0] x1
            QuadConv(ch, ch, 3, "same"),   # [1] x2
            QuadConv(ch, ch, 3, "same"),   # [2] x3
            QuadConv(ch, ch, 3, "same"),   # [3] x4
            QuadConv(ch, ch, 3, "valid"),  # [4] x5 — bottleneck (size -2px)
        ])

        # ── Decoder (QuadDeconv, giữ nguyên từ QAE gốc) ───────────────────────
        self.decoder = nn.ModuleList([
            QuadDeconv(ch, ch, 3, "valid"),  # [0] x6
            QuadDeconv(ch, ch, 3, "same"),   # [1] x7
            QuadDeconv(ch, ch, 3, "same"),   # [2] x8
            QuadDeconv(ch, ch, 3, "same"),   # [3] x9
            QuadDeconv(ch, 1,  3, "same"),   # [4] output
        ])

        self.relu = nn.ReLU()

        # ── FixedSobelLayer + edge_proj ────────────────────────────────────────
        if self.use_sobel:
            self.sobel     = FixedSobelLayer()
            self.edge_proj = nn.Conv2d(4, ch, 1)  # 4 → 15

        # ── EdgeDilatedResidualBlocks (bottleneck) ─────────────────────────────
        dilations = [2, 3][:n_edge_blocks]
        self.edge_blocks = nn.ModuleList(
            [EdgeDilatedResidualBlock(ch, d) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x  # Lưu cho residual output và Sobel input

        # ── Encoder ──────────────────────────────────────────────────────────
        x1 = self.relu(self.encoder[0](original_x))

        # Inject Sobel edge features vào x1
        if self.use_sobel:
            edges = self.sobel(original_x)        # (B, 4, H, W)
            x1    = x1 + self.edge_proj(edges)    # (B, 15, H, W)

        x2 = self.relu(self.encoder[1](x1))
        x3 = self.relu(self.encoder[2](x2))
        x4 = self.relu(self.encoder[3](x3))
        x5 = self.relu(self.encoder[4](x4))       # (B, 15, H-2, W-2) — valid padding

        # ── EdgeDilatedResidualBlocks (bottleneck) ────────────────────────────
        for eb in self.edge_blocks:
            x5 = eb(x5)

        # ── Decoder (giữ nguyên từ QAE gốc) ──────────────────────────────────
        x6 = self.relu(self.decoder[0](x5) + x4)  # valid deconv → size +2px
        x7 = self.relu(self.decoder[1](x6))
        x8 = self.relu(self.decoder[2](x7) + x2)  # skip từ x2
        x9 = self.relu(self.decoder[3](x8))

        return self.decoder[4](x9) + original_x   # residual từ input gốc
