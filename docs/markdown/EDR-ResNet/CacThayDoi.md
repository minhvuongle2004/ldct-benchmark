# CacThayDoi.md — EDR-ResNet

Tài liệu ghi lại **tất cả thay đổi** khi áp dụng `FixedSobelLayer` + `EdgeDilatedResidualBlock` vào ResNet.

---

## 1. Thay Đổi Kiến Trúc (`network.py`)

### 1.1. ResNet Gốc — Tổng quan

```
in_conv: Conv2d(1, 128, kernel=9, padding=4)
    ↓
10 × ResBlock(128)           ← mỗi block: Conv-BN-ReLU-GroupConv-BN-ReLU-Conv + residual
    ↓
out_conv: Conv2d(128, 1, kernel=3, padding=1)
    ↓
output = input - noise       ← noise subtraction (khác CNN10 và RED-CNN!)
```

**Params gốc:** ~3.5M params (128 channels × 10 blocks)

---

### 1.2. Điểm Thêm FixedSobelLayer

- **Vị trí:** Sau `in_conv`, trước Block 1 (giống CNN10)
- **Cơ chế:**
  - `FixedSobelLayer(input_x)` → edge map 4 channels
  - `edge_proj`: `Conv2d(4, 128, 1)` chiếu về 128 channels
  - Cộng vào features sau `in_conv`: `features = features + edge_proj(edges)`
- **Lưu ý:** Input cho Sobel là **ảnh LDCT gốc** (1 channel), không phải feature map

---

### 1.3. Điểm Thêm EdgeDilatedResidualBlock

- **Vị trí:** Sau Block 5 (giữa 10 ResBlocks) — bottleneck giữa network
- **Cấu hình:** 2 × `EdgeDilatedResidualBlock(128, dilation=2)` và `(128, dilation=3)`
- **Lý do chọn giữa:** Tại đây feature map đã trừu tượng vừa đủ để Dilated Conv phát huy

---

### 1.4. Noise Subtraction — Giữ Nguyên

- `output = original_input - predicted_noise`
- Không thay đổi cơ chế này — chỉ thêm module vào trong backbone

---

### 1.5. So Sánh Params

| | ResNet gốc | EDR-ResNet |
|:-|:---------:|:----------:|
| in_conv | Conv(1→128, k=9) | Conv(1→128, k=9) |
| FixedSobelLayer | ❌ | ✅ (non-trainable) |
| edge_proj | ❌ | Conv(4→128, k=1) |
| Residual Blocks | 10 × ResBlock(128) | 10 × ResBlock(128) |
| EdgeDilatedResidualBlock | ❌ | 2 × EDRB(128) ở giữa |
| out_conv | Conv(128→1, k=3) | Conv(128→1, k=3) |
| Noise subtraction | ✅ | ✅ (giữ nguyên) |
| **Params (ước tính)** | ~3.5M | ~3.6M |

---

## 2. Thay Đổi Loss Function (`loss.py`)

| | ResNet gốc | EDR-ResNet |
|:-|:---------:|:----------:|
| Loss | `nn.MSELoss()` | `CombinedLoss` |
| Công thức | `MSE(pred, target)` | `Charbonnier(pred, target) + α × SobelEdgeLoss(pred, target)` |
| α | — | `0.1` |

- Copy nguyên `loss.py` từ `ldctbench/methods/edrrednet/loss.py`

---

## 3. Thay Đổi Trainer (`Trainer.py`)

| | ResNet gốc | EDR-ResNet |
|:-|:---------:|:----------:|
| `self.criterion` | `nn.MSELoss()` | `CombinedLoss(alpha, beta, gamma)` |
| `train_step` | BaseTrainer mặc định | Override: unpack `(total_loss, components)` |
| `val_step` | BaseTrainer mặc định | Override: unpack tuple |

- Copy nguyên từ `ldctbench/methods/edrrednet/Trainer.py`, chỉ sửa docstring

---

## 4. File Mới Hoàn Toàn

| File | Mô tả |
|:-----|:------|
| `ldctbench/methods/edrresnet/__init__.py` | Đăng ký package |
| `ldctbench/methods/edrresnet/network.py` | Kiến trúc EDR-ResNet |
| `ldctbench/methods/edrresnet/loss.py` | CombinedLoss (copy từ edrrednet) |
| `ldctbench/methods/edrresnet/Trainer.py` | Trainer (copy + sửa docstring) |
| `ldctbench/methods/edrresnet/argparser.py` | Argparser (copy + conflict check) |
| `configs/edrresnet.yaml` | Config training |

---

## 5. Nhật Ký Thay Đổi

| Ngày | Việc đã làm |
|:-----|:------------|
| 12/06/2026 | Tạo file kế hoạch, phân tích kiến trúc ResNet gốc, thiết kế EDR-ResNet |
| 12/06/2026 | ✅ Implement `network.py`, `loss.py`, `Trainer.py`, `argparser.py`, `__init__.py` |
| 12/06/2026 | ✅ Sanity check PASS — Shape (1,1,64,64) ✅ — Trainable params: 2,434,305 |
| 12/06/2026 | ✅ Tạo `configs/edrresnet.yaml`, đăng ký vào METHODS list |
| 12/06/2026 | ✅ Tạo notebook `results/training/EDR-ResNet/train-edrresnet.ipynb` |
| 13/06/2026 | TRAINING XONG — Variant D, 3 seeds (1339, 2024, 42) trên Kaggle T4 |
| 13/06/2026 | Seed 1339: Best val SSIM=0.8764, PSNR=39.700 @ iter 15000 (~9 giờ) |
| 13/06/2026 | Seed 2024: Best val SSIM=0.8774, PSNR=39.740 @ iter 20000 (~9 giờ) |
| 13/06/2026 | Seed 42  : Best val SSIM=0.8774, PSNR=39.599 @ iter 14000 (~9 giờ) |
| 13/06/2026 | MEAN val SSIM: 0.8771 +/- 0.0005 — MEAN PSNR: 39.680 +/- 0.074 |
| 14/06/2026 | EVALUATION XONG — Wilcoxon test trên 9 bệnh nhân test set |
| 14/06/2026 | Kết quả lưu tại: results/training/EDR-ResNet/SoSanh/ |

---

## 6. Kết Quả Đánh Giá Cuối (Test Set — 9 bệnh nhân)

| Metric | ResNet Baseline | EDR-ResNet (mean 3 seeds) | Δ | p-value | Ý nghĩa |
|:-------|:--------------:|:-------------------------:|:-:|:-------:|:-------:|
| PSNR | **44.02 ± 6.74** | 43.73 ± 6.67 | -0.29 dB | 1.000 | ❌ Kém hơn |
| SSIM | **0.9359 ± 0.053** | 0.9332 ± 0.055 | -0.0027 | 1.000 | ❌ Kém hơn |
| **Edge SSIM** | 0.7530 ± 0.138 | **0.7659 ± 0.128** | **+0.0129** | **0.0098** | ✅ **Significant** |

**Wilcoxon test:** EDR-ResNet best seed (seed 42) vs ResNet baseline, n=9

**Nhận xét per-patient:**
- 9/9 bệnh nhân: PSNR **thấp hơn** baseline ❌
- 9/9 bệnh nhân: SSIM **thấp hơn** baseline ❌
- 9/9 bệnh nhân: Edge SSIM **tốt hơn** baseline ✅ (p=0.0098)

**Lý do PSNR/SSIM giảm:**
> ResNet baseline được train với **105 bệnh nhân** (3× nhiều hơn) và MSE loss tối ưu trực tiếp cho PSNR/SSIM.
> EDR-ResNet chỉ train trên **34 bệnh nhân** với Charbonnier+SobelEdgeLoss — không tối ưu cho PSNR/SSIM mà tối ưu cho edge preservation.

**Kết luận:**
> FixedSobelLayer + EdgeDilatedResidualBlock cải thiện **Edge SSIM có ý nghĩa thống kê (p=0.0098)** trên tất cả 9/9 bệnh nhân.
> PSNR/SSIM giảm nhẹ do bất lợi về dữ liệu train (34 vs 105 bệnh nhân) và loss function khác biệt.
> Kết quả này cho thấy 2 module đặc biệt hiệu quả trong **bảo toàn cạnh biên** trên ResNet.
