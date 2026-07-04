"""
EDR-CNN10: Edge-Dilated Residual CNN10
========================================
Áp dụng 2 module từ EDR-REDNet vào kiến trúc CNN10:
  1. FixedSobelLayer  — non-trainable edge extractor (4 hướng: H, V, Diag45, Diag135)
  2. EdgeDilatedResidualBlock — dilated conv (rate=2,3) + residual tại bottleneck

Thay đổi so với CNN10 gốc:
  - Thêm 1 lớp Conv(64→64) làm bottleneck (CNN10 gốc không có)
  - Inject FixedSobelLayer tại bottleneck
  - Thêm 2 EdgeDilatedResidualBlock tại bottleneck
  - Thêm global residual skip (input → output)
  - Loss: CombinedLoss (Charbonnier + SobelEdgeLoss) thay MSE

Tham khảo:
  - CNN10 gốc: Chen et al., Biomed Opt Express 2017
  - EDR-REDNet: Lê Minh Vương, 2026 (nội bộ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedSobelLayer(nn.Module):
    """Non-trainable layer trích xuất edge map theo 4 hướng.

    Copy y hệt từ edrrednet/network.py — không sửa.

    Weights Sobel hoàn toàn cố định (register_buffer), không được cập nhật
    trong quá trình train. Output là 4-channel edge map (H, V, Diag45, Diag135).

    Input:  (B, 1, H, W)
    Output: (B, 4, H, W)
    """

    def __init__(self):
        super().__init__()

        kernels = torch.tensor(
            [
                # Horizontal edges
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                # Vertical edges
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                # Diagonal 45°
                [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                # Diagonal 135°
                [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)  # (4, 1, 3, 3)

        self.register_buffer("weight", kernels)

    def forward(self, x):
        # x: (B, 1, H, W) → output: (B, 4, H, W)
        return F.conv2d(x, self.weight, padding=1)


class EdgeDilatedResidualBlock(nn.Module):
    """Dilated Residual Block nhận biết biên.

    Copy y hệt từ edrrednet/network.py — không sửa.

    Cấu trúc:
      - Conv dilated rate=dilation: mở rộng receptive field không tăng params
      - Conv 3x3 (dilation=1): tinh chỉnh đặc trưng
      - Residual connection: bảo toàn thông tin gốc

    Input/Output: (B, channels, H, W) — giữ nguyên shape.
    """

    def __init__(self, channels: int, dilation: int = 2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,   # padding = dilation để giữ nguyên H, W
                dilation=dilation,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class Model(nn.Module):
    """EDR-CNN10: Edge-Dilated Residual CNN10.

    Kiến trúc (so với CNN10 gốc 3 lớp):

    [Input LDCT]
        ├─→ FixedSobelLayer → edge_map (4ch) → proj_edge (64ch)
        │                                              ↓ (ADD vào bottleneck)
        └─→ Conv(1→64,  9×9) + ReLU   [lớp 1 — giống CNN10 gốc]
                ↓
            Conv(64→64, 3×3) + ReLU   [lớp 2 — BOTTLENECK MỚI, CNN10 gốc dùng 64→32]
                ↓ + edge_map_proj
            EdgeDilatedResidualBlock(64, d=2)   [MỚI]
            EdgeDilatedResidualBlock(64, d=3)   [MỚI]
                ↓
            Conv(64→32, 3×3) + ReLU   [lớp 3 — MỚI, thu hẹp về 32ch]
                ↓
            Conv(32→1,  5×5)           [lớp 4 — output, giống CNN10 gốc]
                ↓ + input (global skip) [MỚI]
            Output

    Parameters
    ----------
    args : Namespace
        Cần có:
        - args.num_edge_blocks (int, default=2): Số EdgeDilatedResidualBlock
        - args.use_sobel_input (bool, default=True): Có dùng FixedSobelLayer không
    """

    def __init__(self, args):
        super().__init__()

        self.use_sobel_input = getattr(args, "use_sobel_input", True) if args is not None else True
        num_edge_blocks = getattr(args, "num_edge_blocks", 2) if args is not None else 2

        # ── [MỚI] Edge extractor (non-trainable) ────────────────────────────────
        if self.use_sobel_input:
            self.sobel = FixedSobelLayer()          # output: (B, 4, H, W)
            self.proj_edge = nn.Conv2d(4, 64, kernel_size=1, bias=False)  # 4→64ch

        # ── Lớp 1: giống CNN10 gốc ──────────────────────────────────────────────
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)

        # ── Lớp 2: BOTTLENECK MỚI (CNN10 gốc: 64→32, ở đây giữ 64→64) ─────────
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # ── [MỚI] EdgeDilatedResidualBlocks tại bottleneck ──────────────────────
        dilations = [2, 3][:num_edge_blocks]
        self.edge_blocks = nn.Sequential(
            *[EdgeDilatedResidualBlock(64, dilation=d) for d in dilations]
        )

        # ── Lớp 3: Thu hẹp từ bottleneck 64ch → 32ch ────────────────────────────
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # ── Lớp 4: Output — giống CNN10 gốc ─────────────────────────────────────
        self.conv4 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # ── [MỚI] Global skip (residual learning) ───────────────────────────────
        residual = x

        # ── [MỚI] Trích xuất edge map ────────────────────────────────────────────
        if self.use_sobel_input:
            edge_map = self.sobel(x)                 # (B, 4, H, W)
            edge_proj = self.proj_edge(edge_map)     # (B, 64, H, W) — cùng H,W vì padding=1

        # ── Lớp 1 ────────────────────────────────────────────────────────────────
        out = self.relu(self.conv1(x))               # (B, 64, H, W)

        # ── Lớp 2 (bottleneck) ───────────────────────────────────────────────────
        out = self.relu(self.conv2(out))             # (B, 64, H, W)

        # ── [MỚI] Inject edge map vào bottleneck ─────────────────────────────────
        if self.use_sobel_input:
            out = out + edge_proj                    # ADD (không concat)

        # ── [MỚI] EdgeDilatedResidualBlocks ──────────────────────────────────────
        out = self.edge_blocks(out)                  # (B, 64, H, W)

        # ── Lớp 3 ────────────────────────────────────────────────────────────────
        out = self.relu(self.conv3(out))             # (B, 32, H, W)

        # ── Lớp 4 (output) ───────────────────────────────────────────────────────
        out = self.conv4(out)                        # (B, 1, H, W)

        # ── [MỚI] Global residual ─────────────────────────────────────────────────
        out = out + residual

        return out
