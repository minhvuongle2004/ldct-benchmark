# EDR-DUGAN: Các Thay Đổi So Với DUGAN Gốc

> Tài liệu này ghi lại **tất cả những gì được thêm vào / thay đổi** trong mô hình DUGAN để tạo ra EDR-DUGAN.
> Cập nhật liên tục trong quá trình triển khai.

---

## 1. Kiến Trúc DUGAN Gốc

DUGAN là mô hình **Dual-Domain GAN** — điểm độc đáo nằm ở hai Discriminator song song:

```
DUGAN:

  Generator G:        RED-CNN (encoder-decoder với skip connections)
                      5 × Conv + 5 × ConvTranspose, 96 channels, k=5
                      
  Image Discriminator D_im:   UNet(repeat_num=6, conv_dim=64) + Spectral Norm
                               Discriminate real vs fake ảnh trực tiếp
                               
  Gradient Discriminator D_grad:  Deep copy của D_im (cùng kiến trúc)
                                   Discriminate Sobel(real) vs Sobel(fake)
                                   ← ĐÂY là điểm độc đáo của DUGAN
```

**Generator Loss của DUGAN:**
```
G_total = lam_adv × (G_adv_img + G_adv_grad)
        + lam_px_im   × MSE(G(x), y)
        + lam_px_grad × L1(Sobel(G(x)), Sobel(y))
```

**Đặc điểm quan trọng:**
- Generator = RED-CNN (KHÔNG phải UNet — UNet chỉ là Discriminator)
- DUGAN đã tích hợp Sobel trong training: `grad_discriminator` + `lam_px_grad × L1(Sobel)`
- CutMix augmentation trong discriminator training để tránh overfitting
- Discriminator dùng Spectral Normalization

---

## 2. Lý Do DUGAN và RED-CNN Là 2 Mô Hình Khác Nhau

Dù dùng cùng Generator backbone (RED-CNN), nhưng DUGAN là một mô hình **hoàn toàn khác**:

| | RED-CNN | DUGAN |
|:-|:-------:|:-----:|
| Training paradigm | Supervised (MSE loss) | Adversarial (dual GAN) |
| Discriminator | ❌ | ✅ 2 UNet discriminator |
| Gradient domain | ❌ | ✅ D_grad trên Sobel(ảnh) |
| Gradient loss | ❌ | ✅ `lam_px_grad × L1(Sobel)` |
| CutMix | ❌ | ✅ |

---

## 3. Thay Đổi Kiến Trúc — Generator (`network.py`)

### 3.1. Thêm `FixedSobelLayer`

- **Vị trí:** Sau `conv1` (Conv 1→96, đầu tiên của Encoder)
- **Cơ chế:** `edge_proj: Conv2d(4, 96, 1)` chiếu 4 kênh Sobel → 96 kênh
- **Inject:** `out = relu(conv1(x)) + edge_proj(Sobel(x))`
- **Synergy với DUGAN:** Generator học edge từ kiến trúc; D_grad ép buộc edge từ training signal → Cùng chiều hướng

### 3.2. Thêm `EdgeDilatedResidualBlock`

- **Vị trí:** Tại bottleneck sau `conv5` (layer cuối encoder, feature map nhỏ nhất)
- **Cấu hình:** 2 × `EdgeDilatedResidualBlock(96, dilation=2)` và `(96, dilation=3)`
- **Lưu ý về shape:** RED-CNN dùng `valid` padding (k=5, no pad) → spatial size thay đổi:
  - Mỗi Conv k=5 valid: `H → H-4`; sau 5 lớp: `H → H-20`
  - patchsize=92: 92 → 72 (bottleneck) → EdgeBlock (same) → 72 → ... → 92

### 3.3. Kiến Trúc So Sánh

```
RED-CNN (Generator gốc):          EDR-DUGAN Generator:

conv1: Conv(1→96,k=5,valid)+ReLU  conv1: Conv(1→96,k=5,valid)+ReLU
                                   [THÊM] + edge_proj(Sobel(x))
conv2: Conv(96→96,k=5,valid)+ReLU conv2: (giữ nguyên)
conv3: Conv(96→96,k=5,valid)+ReLU conv3: (giữ nguyên)
conv4: Conv(96→96,k=5,valid)+ReLU conv4: (giữ nguyên)
conv5: Conv(96→96,k=5,valid)+ReLU conv5: (giữ nguyên)
                                   [THÊM] EdgeDilatedResidualBlock(96, d=2)
                                   [THÊM] EdgeDilatedResidualBlock(96, d=3)
tconv1 → tconv5 (decoder)         tconv1 → tconv5 (giữ nguyên)
output += input                    output += input (giữ nguyên)
```

### 3.4. So Sánh Params

| | DUGAN gốc (Generator) | EDR-DUGAN Generator |
|:-|:---------------------:|:-------------------:|
| Generator | RED-CNN (~1.85M) | RED-CNN + EDR |
| FixedSobelLayer | ❌ | ✅ (non-trainable) |
| edge_proj | ❌ | Conv(4→96, k=1) +96×4+96=480 params |
| EdgeDilatedResidualBlock | ❌ | 2 × EDRB(96) ~200K |
| Discriminators (×2) | UNet + SpectralNorm | **Giữ nguyên** |
| **Generator Params** | ~1.85M | ~2.05M |

---

## 4. Thay Đổi Trainer (`Trainer.py`)

| | DUGAN gốc | EDR-DUGAN |
|:-|:---------:|:---------:|
| Generator | RED-CNN | **EDR-REDCNN** (FixedSobel + EDRB) |
| D_im training | LS-GAN + CutMix | **Giữ nguyên** |
| D_grad training | LS-GAN trên Sobel + CutMix | **Giữ nguyên** |
| G loss | adv + MSE + L1(Sobel) | **Giữ nguyên** |
| Gradient clipping | ❌ | ✅ `clip_grad_norm_(G, 1.0)` |
| AMP (Mixed Precision) | ❌ | ✅ `torch.cuda.amp.autocast` + `GradScaler` |
| Batch size | 92 | 16 (do giới hạn VRAM T4 15GB) |

> **Lý do không đổi G loss:** DUGAN đã có `lam_px_grad × L1(Sobel)` — đây chính là gradient edge loss. Thêm SobelEdgeLoss nữa sẽ bị trùng lặp và gây mất cân bằng G loss.

> **AMP:** Thêm Automatic Mixed Precision (FP16) để giảm VRAM ~50% và tăng tốc training ~1.5x trên Tensor Cores GPU T4.

---

## 5. Synergy Đặc Biệt: FixedSobelLayer + D_grad

Đây là điểm nổi bật nhất của EDR-DUGAN so với các EDR model khác:

```
[Kiến trúc] FixedSobelLayer → Generator học cạnh từ spatial features
     ↓
[Training]  D_grad phân biệt Sobel(real) vs Sobel(G(x)) → ép G tạo ra cạnh tự nhiên
     ↓
[Loss]      lam_px_grad × L1(Sobel(G(x)), Sobel(y)) → pixel-level gradient supervision
```

Ba tầng cùng hướng đến bảo toàn cạnh → **Synergy mạnh nhất** trong tất cả 5 mô hình EDR.

---

## 6. File Mới Hoàn Toàn

| File | Mô tả |
|:-----|:------|
| `ldctbench/methods/edrdugan/__init__.py` | Export Trainer |
| `ldctbench/methods/edrdugan/network.py` | EDR-REDCNN Generator (import từ edrrednet hoặc viết lại) |
| `ldctbench/methods/edrdugan/Trainer.py` | Copy từ dugan/Trainer.py, chỉ đổi import Model |
| `ldctbench/methods/edrdugan/argparser.py` | Copy từ dugan/argparser.py + EDR args |
| `configs/edrdugan.yaml` | Config training |
| `paper_scripts/evaluate_edrdugan.py` | Script đánh giá |

---

## 7. Kết Quả Đánh Giá

### 7.1. So Sánh EDR-DUGAN vs DUGAN Baseline (Test Set — 9 bệnh nhân)

| Model | PSNR | SSIM | Edge SSIM |
|:------|:----:|:----:|:---------:|
| **DUGAN (Baseline)** | **42.60 ± 6.82** | **0.9191 ± 0.0666** | **0.8446 ± 0.0855** |
| EDR-DUGAN (seed 1339) | 41.42 ± 6.01 | 0.9157 ± 0.0690 | 0.8357 ± 0.0918 |
| EDR-DUGAN (seed 2024) | 42.16 ± 6.73 | 0.9132 ± 0.0712 | 0.8335 ± 0.0913 |
| EDR-DUGAN (seed 42) | 41.95 ± 6.28 | 0.9171 ± 0.0670 | 0.8298 ± 0.0939 |

### 7.2. Wilcoxon Signed-Rank Test

| So sánh | PSNR p-value | SSIM p-value | Edge SSIM p-value |
|:--------|:------------:|:------------:|:-----------------:|
| EDR-DUGAN (best) vs DUGAN | 1.0 | 0.998 | 1.0 |

**Kết luận:** EDR-DUGAN **không cải thiện** so với DUGAN baseline trên cả 3 metrics.

### 7.3. Phân Tích Nguyên Nhân

1. **Thiếu iterations:** EDR-DUGAN chỉ train **25.000 iterations** (do giới hạn 12h Kaggle T4), trong khi DUGAN baseline được train đầy đủ **50.000 iterations** trên GPU mạnh hơn.
2. **Batch size nhỏ (mbs=16):** DUGAN gốc dùng mbs=92, nhưng do 3 model song song (G + D_im + D_grad) tốn quá nhiều VRAM trên T4 (15GB), phải giảm xuống 16. Batch nhỏ → gradient nhiễu → GAN khó hội tụ tối ưu.
3. **AMP (FP16):** Mixed Precision có thể gây mất chính xác nhỏ ở một số phép tính loss/gradient — đặc biệt trong GAN, sự ổn định rất nhạy cảm.
4. **DUGAN đã tích hợp edge supervision:** DUGAN gốc đã có D_grad (Sobel discriminator) + `lam_px_grad × L1(Sobel)` rất mạnh (hệ số 27.8). Việc thêm FixedSobelLayer + EdgeDilatedResidualBlock vào Generator bị "dư thừa" vì training signal đã cover rồi → không tạo thêm hiệu quả.

> **Lưu ý quan trọng:** Đây là kết quả trong điều kiện training không công bằng hoàn toàn (25k vs 50k iter, mbs 16 vs 92). Kết quả vẫn có giá trị học thuật vì cho thấy: khi mô hình gốc ĐÃ CÓ edge supervision mạnh (như DUGAN), thì việc thêm EDR modules vào kiến trúc không mang lại thêm cải thiện đáng kể.

---

## 8. Nhật Ký Thay Đổi

| Ngày | Việc đã làm |
|:-----|:------------|
| 22/06/2026 | Phân tích TransCT (loại) → WGAN-VGG (loại) → DUGAN (chọn) |
| 22/06/2026 | Xác nhận DUGAN Generator = RED-CNN; DUGAN vẫn là mô hình riêng biệt do training khác |
| 22/06/2026 | Tạo file kế hoạch, phân tích kiến trúc DUGAN, thiết kế EDR-DUGAN |
| 22/06/2026 | Implementation code: network.py, Trainer.py, argparser.py, edrdugan.yaml |
| 22/06/2026 | Fix argparse conflicts (wganvgg, dugan), fix OOM (mbs 64→16), thêm AMP |
| 22-24/06/2026 | Train 3 seeds (1339, 2024, 42) × 25.000 iterations trên Kaggle T4 |
| 24/06/2026 | Evaluation: EDR-DUGAN vs DUGAN baseline — không cải thiện (p > 0.05) |
