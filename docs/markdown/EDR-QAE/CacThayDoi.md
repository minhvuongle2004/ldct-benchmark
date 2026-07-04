# CacThayDoi.md — EDR-QAE

Tài liệu ghi lại **tất cả thay đổi** khi áp dụng `FixedSobelLayer` + `EdgeDilatedResidualBlock` vào QAE.

---

## 1. Thay Đổi Kiến Trúc (`network.py`)

### 1.1. QAE Gốc — Tổng quan

```
Encoder:
  QuadConv(1→15,  k=3, same)   ← x1
  QuadConv(15→15, k=3, same)   ← x2
  QuadConv(15→15, k=3, same)   ← x3
  QuadConv(15→15, k=3, same)   ← x4
  QuadConv(15→15, k=3, valid)  ← x5  ← bottleneck (spatial -2px mỗi chiều)

Decoder:
  QuadDeconv(15→15, k=3, valid)  + x4  ← x6
  QuadDeconv(15→15, k=3, same)         ← x7
  QuadDeconv(15→15, k=3, same)  + x2   ← x8
  QuadDeconv(15→15, k=3, same)         ← x9
  QuadDeconv(15→1,  k=3, same)  + x    ← output (residual từ input)
```

**Cơ chế QuadConv (đặc biệt):**
```
output = (W_r·x_c + b_r) × (W_g·x_c + b_g) + (W_b·x_c² + b_b)
với x_c = torch.clamp(x, min=-10.0, max=10.0)
```
→ Neuron bậc 2 (quadratic), KHÔNG phải Conv thông thường!
**Lưu ý:** Thêm `torch.clamp` vào x trước khi tính toán để chống activation explosion (lỗi NaN/inf do tích lũy giá trị cực lớn qua nhiều layer).

**Params gốc:** ~29K params (rất nhẹ — chỉ 15 channels)

---

### 1.2. Điểm Thêm FixedSobelLayer

- **Vị trí:** Sau encoder[0] (x1 đầu tiên)
- **Cơ chế:**
  - `FixedSobelLayer(input_x)` → edge map 4 channels
  - `edge_proj`: `Conv2d(4, 15, 1)` chiếu về 15 channels (QAE dùng 15ch)
  - Cộng vào x1: `x1 = x1 + edge_proj(edges)`
- **Lưu ý:** Input cho Sobel là **ảnh LDCT gốc**, không phải feature map

---

### 1.3. Điểm Thêm EdgeDilatedResidualBlock

- **Vị trí:** Sau encoder[4] (x5, bottleneck) — trước decoder[0]
- **Cấu hình:** 2 × `EdgeDilatedResidualBlock(15, dilation=2)` và `(15, dilation=3)`
- **Lý do chọn bottleneck:** x5 là điểm nén thông tin nhất, receptive field rộng nhất
- **Lưu ý về shape:** encoder[4] dùng `valid` padding → x5 nhỏ hơn input 2px mỗi chiều
  - Patch 36×36 → x5: 34×34 → EdgeBlock (same) → 34×34 → decoder[0] → 36×36

---

### 1.4. Residual Skip từ Input — Giữ Nguyên

- `output = decoder[4](x9) + input`
- Không thay đổi cơ chế residual học từ input

---

### 1.5. So Sánh Params

| | QAE gốc | EDR-QAE |
|:-|:-------:|:-------:|
| Encoder | 5 × QuadConv(15ch) | 5 × QuadConv(15ch) |
| FixedSobelLayer | ❌ | ✅ (non-trainable) |
| edge_proj | ❌ | Conv(4→15, k=1) |
| EdgeDilatedResidualBlock | ❌ | 2 × EDRB(15) ở bottleneck |
| Decoder | 5 × QuadDeconv(15ch) | 5 × QuadDeconv(15ch) |
| Residual skip | ✅ | ✅ (giữ nguyên) |
| **Params (ước tính)** | ~29K | ~31K |

---

## 2. Thay Đổi Loss Function (`loss.py`)

| | QAE gốc | EDR-QAE |
|:-|:-------:|:-------:|
| Loss | `nn.MSELoss()` | `CombinedLoss` |
| Công thức | `MSE(pred, target)` | `Charbonnier(pred, target) + α × SobelEdgeLoss(pred, target)` |
| α | — | `0.1` |

- Copy nguyên `loss.py` từ `ldctbench/methods/edrrednet/loss.py`

---

## 3. Thay Đổi Trainer (`Trainer.py`)

| | QAE gốc | EDR-QAE |
|:-|:-------:|:-------:|
| `self.criterion` | `nn.MSELoss()` | `CombinedLoss(alpha, beta, gamma)` |
| `train_step` | BaseTrainer mặc định | Override: unpack `(total_loss, components)` |
| `val_step` | BaseTrainer mặc định | Override: unpack tuple |

- Copy nguyên từ `ldctbench/methods/edrrednet/Trainer.py`, chỉ sửa docstring

---

## 4. File Mới Hoàn Toàn

| File | Mô tả |
|:-----|:------|
| `ldctbench/methods/edrqae/__init__.py` | Đăng ký package |
| `ldctbench/methods/edrqae/network.py` | Kiến trúc EDR-QAE |
| `ldctbench/methods/edrqae/loss.py` | CombinedLoss (copy từ edrrednet) |
| `ldctbench/methods/edrqae/Trainer.py` | Trainer (copy + sửa docstring) |
| `ldctbench/methods/edrqae/argparser.py` | Argparser (copy + conflict check) |
| `configs/edrqae.yaml` | Config training |
| `paper_scripts/evaluate_edrqae.py` | Script đánh giá EDR-QAE vs Baseline |

---

## 5. Kết Quả Training (Kaggle T4)

Đã hoàn thành training trên Kaggle (3 seeds) với các hyperparameter: `mbs=64`, `max_iterations=50000`, `patchsize=36`.

| Seed | Max Validation SSIM | Kỷ lục tại Iteration | Ghi chú |
|:----:|:-------------------:|:--------------------:|:-------|
| 1339 | ~0.8708 | 46000 | Ổn định, hội tụ tốt |
| 2024 | ~0.8718 | 41000 | Cao nhất |
| 42   | ~0.8718 | 14000 | Hội tụ rất sớm |

*Lưu ý:* QAE Baseline chưa được EDR cải tiến thường có SSIM thấp hơn và training dễ bị bất ổn định. Việc kết hợp `FixedSobelLayer` + `EdgeDilatedResidualBlock` và đặc biệt là **Gradient/Input Clamping** đã giúp mô hình QAE train ổn định và đạt SSIM > 0.87 trên validation set.

---

## 6. Kết Quả Đánh Giá (Test Set 9 Bệnh Nhân)

Đánh giá thực tế trên tập Test độc lập thông qua script `paper_scripts/evaluate_edrqae.py`:

| Model | PSNR | SSIM | Edge SSIM |
|:------|:----:|:----:|:---------:|
| QAE Baseline | 42.6953 | 0.9257 | 0.7726 |
| **EDR-QAE (Trung bình 3 seeds)** | **42.9856** | **0.9274** | **0.7734** |

**Kết luận:**
- **PSNR và SSIM tăng đáng kể:** p-value của SSIM là 0.009766 và PSNR là 0.001953 (cả hai đều < 0.05), chứng tỏ sự nâng cấp có ý nghĩa thống kê.
- Như vậy, module EDR không chỉ cải thiện các mạng tích chập thông thường (CNN, ResNet) mà còn chứng minh được hiệu quả tăng cường thông tin không gian trên các cấu trúc đặc biệt như **Quadratic Autoencoder (QAE)**.

---

## 7. Nhật Ký Thay Đổi

| Ngày | Việc đã làm |
|:-----|:------------|
| 14/06/2026 | Tạo file kế hoạch, phân tích kiến trúc QAE gốc, thiết kế EDR-QAE |
| 18/06/2026 | Khắc phục gradient/activation explosion ở QuadConv bằng Clamping |
| 21/06/2026 | Hoàn thành training Kaggle 3 seeds, cập nhật kết quả Max SSIM |
| 22/06/2026 | Hoàn thành đánh giá test set, EDR-QAE vượt qua baseline QAE (p<0.05) |
