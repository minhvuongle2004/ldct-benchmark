"""
EDR-DUGAN: network.py
======================
Generator của EDR-DUGAN là EDR-REDCNN (từ edrrednet/network.py).
Tái sử dụng trực tiếp — không viết lại.

Cấu trúc Generator (EDR-REDCNN):
  - FixedSobelLayer: inject edge map vào bottleneck
  - EdgeDilatedResidualBlock × 2: xử lý edge tại bottleneck
  - Kiến trúc RED-CNN: 5 × Conv + 5 × ConvTranspose, 96ch, k=5, valid padding

Discriminator:
  - UNet × 2 (image domain + gradient domain) từ dugan/network.py
  - Giữ nguyên hoàn toàn
"""

# Generator = EDR-REDCNN (tái sử dụng hoàn toàn)
from ldctbench.methods.edrrednet.network import (
    EdgeDilatedResidualBlock,
    FixedSobelLayer,
    Model,
)

# Discriminator = UNet từ DUGAN (giữ nguyên)
from ldctbench.methods.dugan.network import UNet

__all__ = ["Model", "UNet", "FixedSobelLayer", "EdgeDilatedResidualBlock"]
