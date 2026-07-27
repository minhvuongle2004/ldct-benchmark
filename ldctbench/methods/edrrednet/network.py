"""
EDR-REDNet: Edge-Dilated Residual RED-CNN
==========================================
Kế thừa kiến trúc RED-CNN và bổ sung:
  1. FixedSobelLayer  — non-trainable edge extractor (4 hướng: H, V, Diag45, Diag135)
  2. EdgeDilatedResidualBlock — dilated conv (rate=2,3) + residual ở bottleneck

Tham khảo:
  - RED-CNN gốc: Chen et al., IEEE TMI 2017
  - ER-Net: Gholizadeh-Ansari et al., J. Digital Imaging 2020
  - Dilated Conv: Yu & Koltun, ICLR 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedSobelLayer(nn.Module):
    """Non-trainable layer trích xuất edge map theo 4 hướng.

    Weights Sobel hoàn toàn cố định (register_buffer), không được cập nhật
    trong quá trình train. Output là 4-channel edge map (H, V, Diag45, Diag135).

    Input:  (B, 1, H, W)
    Output: (B, 4, H, W)
    """

    def __init__(self):
        super().__init__()

        # 4 Sobel kernels cố định: Horizontal, Vertical, Diagonal 45°, Diagonal 135°
        kernels = torch.tensor(
            [
                # Horizontal edges (phát hiện cạnh ngang)
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                # Vertical edges (phát hiện cạnh dọc)
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                # Diagonal 45° (cạnh chéo trái-phải)
                [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                # Diagonal 135° (cạnh chéo phải-trái)
                [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
            ],
            dtype=torch.float32,
        )  # shape: (4, 3, 3)

        # Reshape thành (out_channels=4, in_channels=1, kH=3, kW=3)
        kernels = kernels.unsqueeze(1)

        # register_buffer: lưu vào state_dict nhưng không train
        self.register_buffer("weight", kernels)

    def forward(self, x):
        # x: (B, 1, H, W) → output: (B, 4, H, W)
        return F.conv2d(x, self.weight, padding=1)


class EdgeDilatedResidualBlock(nn.Module):
    """Dilated Residual Block nhận biết biên.

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
    """EDR-REDNet: Edge-Dilated Residual RED-CNN.

    Kiến trúc:
      [Input LDCT]
          ├─→ FixedSobelLayer → edge_map (4ch) → proj_edge (96ch)
          │                                              ↓ (add vào bottleneck)
          └─→ Encoder (Conv1→Conv5, RED-CNN style)
                  ↓ bottleneck
              [EdgeBlock d=2] → [EdgeBlock d=3]
                  ↓
              Decoder (TConv1→TConv5)
                  ↓
              Output + residual_1 (global skip)

    Parameters
    ----------
    args : Namespace
        Cần có: args.num_edge_blocks (số EdgeBlock, default=2)
    out_ch : int
        Số feature channels (default=96, giống RED-CNN gốc)
    """

    def __init__(self, args, out_ch: int = 96):
        super().__init__()

        self.use_sobel_input = getattr(args, "use_sobel_input", True) if args is not None else True
        num_edge_blocks = getattr(args, "num_edge_blocks", 2) if args is not None else 2

        # ── Edge extractor (non-trainable) ──────────────────────────────────────
        if self.use_sobel_input:
            self.sobel = FixedSobelLayer()  # output: (B, 4, H, W)
            # Project 4-channel edge map → out_ch để cộng vào bottleneck
            self.proj_edge = nn.Conv2d(4, out_ch, kernel_size=1, bias=False)

        # ── Encoder (giống RED-CNN) ──────────────────────────────────────────────
        # ... (giữ nguyên các lớp conv1-5) ...
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        # ── EdgeDilatedResidualBlocks tại bottleneck ─────────────────────────────
        # Dùng dilation 2 và 3 để nắm bắt ngữ cảnh rộng hơn (mạch máu, cấu trúc nhỏ)
        dilations = [2, 3][:num_edge_blocks]
        self.edge_blocks = nn.Sequential(
            *[EdgeDilatedResidualBlock(out_ch, dilation=d) for d in dilations]
        )

        # ── Decoder (giống RED-CNN) ──────────────────────────────────────────────
        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # ── Global skip (residual learning) ─────────────────────────────────────
        residual_1 = x
        
        # ── Encoder ─────────────────────────────────────────────────────────────
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out

        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out

        out = self.relu(self.conv5(out))  # bottleneck feature: (B, 96, H', W')

        # ── Tích hợp edge map vào bottleneck (nếu bật) ──────────────────────────
        if self.use_sobel_input:
            edge_map = self.sobel(x)  # (B, 4, H, W)
            # Exact central crop for coordinate alignment (resolving valid convolution spatial shift)
            diff_h = edge_map.shape[2] - out.shape[2]
            diff_w = edge_map.shape[3] - out.shape[3]
            pad_h = diff_h // 2
            pad_w = diff_w // 2
            if diff_h > 0 and diff_w > 0:
                edge_cropped = edge_map[:, :, pad_h : edge_map.shape[2] - pad_h, pad_w : edge_map.shape[3] - pad_w]
            else:
                edge_cropped = edge_map
            edge_proj = self.proj_edge(edge_cropped)  # (B, 96, H', W')
            out = out + edge_proj  # ADD (không concat để không tăng params decoder)

        # ── EdgeDilatedResidualBlocks ────────────────────────────────────────────
        out = self.edge_blocks(out)

        # ── Decoder ─────────────────────────────────────────────────────────────
        out = self.tconv1(out)
        out += residual_3

        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2

        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))

        # ── Residual learning: cộng lại ảnh gốc ─────────────────────────────────
        out += residual_1
        return out
