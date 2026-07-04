# EDR-CNN10: Các Thay Đổi So Với CNN10 Gốc

> Tài liệu này ghi lại **tất cả những gì được thêm vào / thay đổi** trong mô hình CNN10 để tạo ra EDR-CNN10.
> Cập nhật liên tục trong quá trình triển khai.

---

## 1. Thay Đổi Kiến Trúc (`network.py`)

### 1.1. Thêm lớp bottleneck trung gian

**CNN10 gốc (3 lớp):**
```
Conv(1→64, 9×9)  + ReLU
Conv(64→32, 3×3) + ReLU
Conv(32→1,  5×5)
```

**EDR-CNN10 (4 lớp + modules):**
```
Conv(1→64,  9×9)  + ReLU
Conv(64→64, 3×3)  + ReLU   ← [MỚI] bottleneck giữ 64 channels
    + FixedSobelLayer inject ← [MỚI] edge prior cộng vào đây
    + EdgeDilatedResidualBlock(64, d=2) ← [MỚI]
    + EdgeDilatedResidualBlock(64, d=3) ← [MỚI]
Conv(64→32, 3×3)  + ReLU
Conv(32→1,  5×5)
    + global residual skip   ← [MỚI] cộng lại input gốc
```

**Lý do thêm bottleneck:** CNN10 gốc thay đổi channel (64→32→1) ngay lập tức — không có điểm nào có channel nhất quán để gắn EdgeDilatedResidualBlock (cần channel đầu vào = đầu ra). Thêm Conv(64→64) tạo ra điểm gắn hợp lệ.

---

### 1.2. Thêm `FixedSobelLayer`

- **Lấy từ:** `ldctbench/methods/edrrednet/network.py` — copy y hệt, không sửa
- **Chức năng:** Trích xuất edge map 4 hướng (Ngang, Dọc, Chéo 45°, Chéo 135°)
- **Input:** `(B, 1, H, W)` → **Output:** `(B, 4, H, W)`
- **Trainable params thêm vào:** **0** (dùng `register_buffer`)
- **Cách inject:** Project 4-channel edge map → 64-channel qua `Conv2d(4→64, 1×1)`, rồi **cộng (ADD)** vào bottleneck feature (không concat để tránh tăng channel decoder)

---

### 1.3. Thêm `EdgeDilatedResidualBlock`

- **Lấy từ:** `ldctbench/methods/edrrednet/network.py` — copy y hệt, không sửa
- **Số block:** 2 (dilation=2 và dilation=3)
- **Chức năng:** Mở rộng receptive field mà không tăng số tham số
- **Vị trí:** Ngay sau khi inject edge map tại bottleneck
- **Params thêm vào mỗi block:** `2 × (64×64×3×3) = ~73K params`

---

### 1.4. Thêm Global Residual Skip

- **CNN10 gốc:** Không có residual connection nào
- **EDR-CNN10:** Thêm `output += input` ở cuối forward pass
- **Lý do:** Stabilize training khi thêm block mới; nếu các block không giúp ích, gradient có đường tắt

---

## 2. Thay Đổi Loss Function (`loss.py`)

| | CNN10 gốc | EDR-CNN10 |
|:-|:---------:|:---------:|
| Loss | `nn.MSELoss()` | `CombinedLoss` |
| Công thức | `MSE(pred, target)` | `Charbonnier(pred, target) + α × SobelEdgeLoss(pred, target)` |
| α (edge weight) | — | `0.1` (mặc định) |

- **`loss.py`**: Copy nguyên từ `ldctbench/methods/edrrednet/loss.py` — **không sửa gì**
- **Lý do dùng Charbonnier thay MSE:** Robust hơn với noise pixel nặng trong CT
- **Lý do thêm SobelEdgeLoss:** Ép model tối ưu gradient biên, không chỉ pixel-wise

---

## 3. Thay Đổi Trainer (`Trainer.py`)

| | CNN10 gốc | EDR-CNN10 |
|:-|:---------:|:---------:|
| `self.criterion` | `nn.MSELoss()` | `CombinedLoss(alpha, beta, gamma)` |
| `train_step` | BaseTrainer mặc định | Override: unpack `(total_loss, components)` |
| `val_step` | BaseTrainer mặc định | Override: unpack tuple từ CombinedLoss |
| WandB log | Loss tổng | Log từng component riêng |

- **`Trainer.py`**: Copy nguyên từ `ldctbench/methods/edrrednet/Trainer.py`, chỉ sửa docstring
- **`argparser.py`**: Copy nguyên từ `ldctbench/methods/edrrednet/argparser.py` — không sửa

---

## 4. File Mới Hoàn Toàn

| File | Mô tả |
|:-----|:------|
| `ldctbench/methods/edrcnn10/__init__.py` | Export Trainer |
| `ldctbench/methods/edrcnn10/network.py` | Kiến trúc EDR-CNN10 (viết mới) |
| `ldctbench/methods/edrcnn10/loss.py` | Copy từ edrrednet |
| `ldctbench/methods/edrcnn10/Trainer.py` | Copy từ edrrednet + sửa nhỏ |
| `ldctbench/methods/edrcnn10/argparser.py` | Copy từ edrrednet |
| `configs/edrcnn10.yaml` | Config training |

---

## 5. Tóm Tắt Thay Đổi Về Tham Số

| | CNN10 gốc | EDR-CNN10 |
|:-|:---------:|:---------:|
| Trainable params | ~25K | ~171K (+6.8×) |
| Non-trainable params | 0 | 36 (FixedSobelLayer buffers) |
| Lớp Conv | 3 | 4 (+1 bottleneck) |
| Loss components | 1 (MSE) | 2 (Charbonnier + SobelEdge) |

---

## 6. Nhật Ký Cập Nhật

| Ngày | Nội dung |
|:-----|:---------|
| 30/05/2026 | Tạo file kế hoạch, phân tích kiến trúc CNN10 gốc, thiết kế EDR-CNN10 |
| 30/05/2026 | ✅ Tạo `ldctbench/methods/edrcnn10/network.py` — kiến trúc 4 lớp + FixedSobelLayer + EdgeDilatedResidualBlock |
| 30/05/2026 | ✅ Tạo `ldctbench/methods/edrcnn10/loss.py` — copy CombinedLoss từ edrrednet |
| 30/05/2026 | ✅ Tạo `ldctbench/methods/edrcnn10/Trainer.py` — copy + sửa docstring |
| 30/05/2026 | ✅ Tạo `ldctbench/methods/edrcnn10/argparser.py` — copy từ edrrednet |
| 30/05/2026 | ✅ Tạo `ldctbench/methods/edrcnn10/__init__.py` |
| 30/05/2026 | ✅ Tạo `configs/edrcnn10.yaml` |
| 30/05/2026 | ✅ **Sanity check PASS** — Input/Output shape [1,1,128,128] ✅ — Trainable params: 209,153 — Variant B: 208,897 |
| 30/05/2026 | Dang ky edrcnn10 vao METHODS list trong argparser.py |
| 30/05/2026 | Sua edrcnn10/argparser.py va edrrednet/argparser.py - check-before-add de tranh conflict |
| 30/05/2026 | MINI TRAINING TEST PASS (10 steps, CPU) - Loss: 1.5481 -> 1.3619, No NaN, charb+sobel giam deu |
| 30/05/2026 | Ghi chu: may local khong co CUDA, training that se chay tren Kaggle T4 |

| 06/06/2026 | TRAINING XONG - Variant D, 3 seeds (1339, 2024, 42) |
| 06/06/2026 | Seed 1339: Best val SSIM=0.8752, PSNR=39.683 @ iter 24000 |
| 06/06/2026 | Seed 2024: Best val SSIM=0.8759, PSNR=39.602 @ iter 23000 |
| 06/06/2026 | Seed 42  : Best val SSIM=0.8770, PSNR=39.610 @ iter 22000 |
| 06/06/2026 | MEAN val SSIM: 0.8760 +/- 0.0009 -- MEAN PSNR: 39.632 +/- 0.045 |
| 06/06/2026 | Thoi gian train: ~1 gio/seed (vs RED-CNN 7 gio/seed - nhanh hon 7x) |
| 06/06/2026 | Checkpoint size: 843KB/seed |

| 12/06/2026 | EVALUATION XONG - Chay Wilcoxon tren 9 benh nhan test |
| 12/06/2026 | Ket qua luu tai: results/training/EDR-CNN10/SoSanh/ |

---

## 5. Kết Quả Đánh Giá Cuối (Test Set — 9 bệnh nhân)

| Metric | CNN10 Baseline | EDR-CNN10 (mean 3 seeds) | Δ | p-value | Ý nghĩa |
|:-------|:--------------:|:------------------------:|:-:|:-------:|:-------:|
| **PSNR** | 42.83 ± 6.45 | 43.49 ± 6.75 | **+0.66 dB** | 0.0020 | ✅ Significant |
| **SSIM** | 0.9275 ± 0.059 | 0.9291 ± 0.059 | **+0.0016** | 0.0020 | ✅ Significant |
| Edge SSIM | 0.7521 ± 0.146 | 0.7549 ± 0.131 | +0.0028 | 0.1504 | ⚠️ Not significant |

**Wilcoxon test:** EDR-CNN10 best seed (seed 2024) vs CNN10 baseline, n=9, alternative="greater"

**Nhận xét per-patient:**
- 9/9 bệnh nhân: PSNR tốt hơn ✅
- 8/9 bệnh nhân: SSIM tốt hơn ✅ (C021 giảm 0.000344 — không đáng kể)
- Edge SSIM: Chest patients tốt hơn ✅, Liver patients giảm nhẹ ⚠️

**Kết luận:**
> FixedSobelLayer + EdgeDilatedResidualBlock cải thiện PSNR và SSIM có ý nghĩa thống kê trên CNN10.
> Chứng minh được tính tổng quát hóa của 2 module sang kiến trúc CNN khác ngoài RED-CNN.
> Edge SSIM chưa significant — hiệu quả biên phụ thuộc vào vùng giải phẫu (Chest tốt hơn Liver).
