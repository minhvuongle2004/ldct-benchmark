# EDR-DUGAN: Kế Hoạch Thực Hiện

> Đánh dấu `[x]` khi hoàn thành từng bước.

---

## Tổng Quan Tiến Độ

| Giai đoạn | Trạng thái |
|:----------|:----------:|
| 1. Phân tích & Thiết kế | ✅ |
| 2. Implementation (code) | ✅ |
| 3. Sanity check & debug | ✅ |
| 4. Training (Kaggle T4) | ✅ |
| 5. Evaluation & So sánh | ✅ |
| 6. Push GitHub | ✅ |

---

## GIAI ĐOẠN 1 — Phân Tích & Thiết Kế ✅

### [x] Bước 1.1 — Xác nhận kiến trúc DUGAN

```
DUGAN = Generator (RED-CNN) + 2 × UNet Discriminator (image + gradient)

Generator G (RED-CNN):
  Encoder: 5 × Conv(96ch, k=5, valid padding) → feature map giảm 4px mỗi lớp
  Decoder: 5 × ConvTranspose(96ch, k=5, valid) + skip connections
  Residual: output += input
  
G Loss = lam_adv × (G_adv_img + G_adv_grad)
       + lam_px_im   × MSE(G(x), y)      ← pixel loss trên ảnh
       + lam_px_grad × L1(Sobel(G(x)), Sobel(y))  ← gradient loss
```

**Patchsize constraint:**
- valid padding k=5: mỗi lớp -4px mỗi chiều
- 5 encoder layers: patch giảm 20px (10px mỗi bên)
- patchsize gốc: 128 → bottleneck: 108 → output: 128 ✅
- EdgeDilatedResidualBlock (same padding): 108 → 108 ✅

### [x] Bước 1.2 — Thiết kế EDR-DUGAN Generator

```python
# forward(self, x):
original_x = x
residual_1 = x

out = relu(conv1(x))
# THÊM MỚI: Sobel injection
out = out + self.edge_proj(self.sobel(original_x))

out = relu(conv2(out))
residual_2 = out
out = relu(conv3(out))
out = relu(conv4(out))
residual_3 = out
out = relu(conv5(out))   # bottleneck

# THÊM MỚI: EdgeDilatedResidualBlocks
for eb in self.edge_blocks:
    out = eb(out)

# Decoder (giữ nguyên)
out = tconv1(out) + residual_3
out = tconv2(relu(out))
out = tconv3(relu(out)) + residual_2
out = tconv4(relu(out))
out = tconv5(relu(out))
out += residual_1   # global residual

return out
```

### [x] Bước 1.3 — Xác nhận Trainer strategy

- **Giữ nguyên** toàn bộ DUGAN Trainer (dual discriminators, cutmix, gradient loss)
- **Chỉ đổi** `from .network import Model` → import từ `edrdugan.network`
- **Không thêm** SobelEdgeLoss vào G vì `lam_px_grad × L1(Sobel)` đã cover

---

## GIAI ĐOẠN 2 — Implementation

### [x] Bước 2.1 — Tạo package `ldctbench/methods/edrdugan/`

```
ldctbench/methods/edrdugan/
├── __init__.py
├── network.py       ← VIẾT MỚI: EDR-REDCNN Generator (copy từ edrrednet/network.py + EDR modules)
├── Trainer.py       ← Copy từ dugan/Trainer.py, chỉ đổi import Model
└── argparser.py     ← Copy từ dugan/argparser.py + EDR args (num_edge_blocks, use_sobel_input)
```

### [x] Bước 2.2 — Viết `network.py` (Option A: import từ edrrednet)

**Option A (Khuyến nghị):** Import Model từ `edrrednet.network` trực tiếp:
```python
from ldctbench.methods.edrrednet.network import Model
```
→ Tái sử dụng hoàn toàn, không cần viết lại.

**Option B:** Copy và viết lại (nếu muốn tùy chỉnh riêng).

Sau khi chọn, kiểm tra shape:
```python
from ldctbench.methods.edrdugan.network import Model
args = Namespace(use_sobel_input=True, num_edge_blocks=2, loss_alpha=0.1)
model = Model(args)
x = torch.randn(1, 1, 128, 128)
y = model(x)
assert y.shape == x.shape
```

### [x] Bước 2.3 — Tạo `configs/edrdugan.yaml`

```yaml
trainer: edrdugan
lr: 1.2436216786633454e-05    # Giữ nguyên LR của DUGAN gốc
adam_b1: 0.653411010331042    # Giữ nguyên β1
mbs: 64                       # Giảm từ 92 (generator nặng hơn chút)
max_iterations: 50000
patchsize: 128                # Giữ nguyên của DUGAN gốc
# DUGAN-specific params (giữ nguyên từ gốc):
n_d_train: 2
lam_adv: 0.080335201069619
lam_px_im: 1.0
lam_px_grad: 27.81452397771084
lam_cutmix: 2.647865385857211
cutmix_prob: 0.5
cutmix_warmup_iter: 5000
# EDR params:
num_edge_blocks: 2
use_sobel_input: true
```

### [x] Bước 2.4 — Đăng ký `edrdugan` vào `ldctbench/utils/argparser.py`

```python
METHODS = [..., "edrdugan", ...]
```

---

## GIAI ĐOẠN 3 — Sanity Check & Debug

### [x] Bước 3.1 — Forward pass Generator ✅

```
[OK] Generator: in=(1,1,128,128), out=(1,1,128,128)
     Trainable params: 2,181,025
[OK] Discriminator: enc=scalar, dec=(1,1,128,128)
[OK] Grad discriminator: enc=scalar, dec=(1,1,128,128)
```

### [x] Bước 3.2 — Kiểm tra toàn bộ DUGAN training loop ✅
### [x] Bước 3.3 — Fix bugs: argparse conflict, OOM, thêm AMP

---

## GIAI ĐOẠN 4 — Training (Kaggle T4)

### Chuẩn bị
- [x] Tạo notebook `results/training/EDR-DUGAN/train-edrdugan.ipynb`
- [x] Upload lên Kaggle, add dataset LDCT

### [x] Bước 4.1 — Train seed 1339 ✅ (25.000 iter, mbs=16, AMP)
### [x] Bước 4.2 — Train seed 2024 ✅
### [x] Bước 4.3 — Train seed 42 ✅

> Giới hạn Kaggle T4: chỉ train 25.000/50.000 iter do timeout 12h

---

## GIAI ĐOẠN 5 — Evaluation & So Sánh

### [x] Bước 5.1 — Tạo evaluation script ✅

```bash
python paper_scripts/evaluate_edrdugan.py
```

### [x] Bước 5.2 — Kết quả

| Model | PSNR | SSIM | Edge SSIM |
|:------|:----:|:----:|:---------:|
| **DUGAN baseline** | **42.60 ± 6.82** | **0.9191 ± 0.0666** | **0.8446 ± 0.0855** |
| EDR-DUGAN (seed 1339) | 41.42 ± 6.01 | 0.9157 ± 0.0690 | 0.8357 ± 0.0918 |
| EDR-DUGAN (seed 2024) | 42.16 ± 6.73 | 0.9132 ± 0.0712 | 0.8335 ± 0.0913 |
| EDR-DUGAN (seed 42) | 41.95 ± 6.28 | 0.9171 ± 0.0670 | 0.8298 ± 0.0939 |

**Wilcoxon:** Tất cả p > 0.05 → Không có ý nghĩa thống kê.

---

## GIAI ĐOẠN 6 — Push GitHub

### [x] Bước 6.1 — Commit & Push ✅

---

## Ghi Chú Quan Trọng

> **patchsize = 128:** RED-CNN dùng `valid` padding k=5 → 5 encoder layers làm giảm 20px (10px mỗi bên). patchsize=128 cho bottleneck 108×108. EdgeBlock (same) giữ nguyên 108. Decoder khôi phục về 128 ✅

> **DUGAN nặng hơn các EDR model khác:** 3 model song song (G, D_im, D_grad) → VRAM cao hơn. Nếu OOM giảm mbs xuống 32.

> **lam_px_grad = 27.81 (rất cao):** DUGAN đặt trọng số gradient loss rất cao → gradient supervision mạnh. FixedSobelLayer trong Generator sẽ synergy trực tiếp với loss này.

> **Không thêm SobelEdgeLoss vào G:** DUGAN đã có gradient loss đủ mạnh. Thêm nữa sẽ làm mất cân bằng giữa các loss terms.

---

*Cập nhật lần cuối: 24/06/2026*
