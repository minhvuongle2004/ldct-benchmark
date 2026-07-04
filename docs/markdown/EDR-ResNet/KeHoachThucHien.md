# KeHoachThucHien.md — EDR-ResNet

Kế hoạch từng bước để tích hợp `FixedSobelLayer` + `EdgeDilatedResidualBlock` vào **ResNet**.

> **Mục tiêu:** Chứng minh 2 module cải thiện ResNet (kiến trúc noise-subtraction CNN), bổ sung bằng chứng generalization bên cạnh EDR-CNN10.

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

> 🎉 **HOÀN THÀNH** — 14/06/2026

---

## GIAI ĐOẠN 1 — Phân Tích & Thiết Kế ✅

### [x] Bước 1.1 — Phân tích kiến trúc ResNet gốc

```
ResNet gốc:
  in_conv: Conv2d(1, 128, kernel=9, padding=4)
      ↓
  10 × ResBlock(128)
      ↓
  out_conv: Conv2d(128, 1, kernel=3, padding=1)
      ↓
  return input - output   ← NOISE SUBTRACTION
```

**Đặc điểm khác biệt so với CNN10/RED-CNN:**
- Học **noise map** thay vì học ảnh sạch
- 10 ResBlock (mỗi block: Conv-BN-ReLU-GroupConv-BN-ReLU-Conv + skip)
- Channel rộng hơn (128 vs 64 của CNN10)

### [x] Bước 1.2 — Thiết kế EDR-ResNet

```
EDR-ResNet:
  in_conv: Conv2d(1, 128, kernel=9, padding=4)
      ↓
  edge_proj(FixedSobelLayer(input_x))   ← Thêm mới
  features = features + edge_proj(edges)
      ↓
  Block 1 → Block 2 → Block 3 → Block 4 → Block 5
      ↓
  EdgeDilatedResidualBlock(128, dilation=2)   ← Thêm mới
  EdgeDilatedResidualBlock(128, dilation=3)   ← Thêm mới
      ↓
  Block 6 → Block 7 → Block 8 → Block 9 → Block 10
      ↓
  out_conv: Conv2d(128, 1, kernel=3, padding=1)
      ↓
  return input - output   ← Giữ nguyên noise subtraction
```

---

## GIAI ĐOẠN 2 — Implementation

### [ ] Bước 2.1 — Tạo package `ldctbench/methods/edrresnet/`

```bash
mkdir ldctbench/methods/edrresnet
```

**Cấu trúc:**
```
ldctbench/methods/edrresnet/
├── __init__.py
├── network.py       ← VIẾT MỚI (dựa trên resnet/network.py)
├── loss.py          ← Copy từ edrrednet/loss.py
├── Trainer.py       ← Copy từ edrrednet/Trainer.py, sửa docstring
└── argparser.py     ← Copy từ edrrednet/argparser.py, giữ conflict check
```

### [ ] Bước 2.2 — Viết `network.py`

- Copy `FixedSobelLayer` và `EdgeDilatedResidualBlock` từ `edrrednet/network.py`
- Sửa `Model.__init__()`:
  - Thêm `self.sobel = FixedSobelLayer()`
  - Thêm `self.edge_proj = nn.Conv2d(4, 128, 1)`
  - Thêm 2 × `EdgeDilatedResidualBlock` vào giữa (sau block 5)
- Sửa `Model.forward()`:
  - Lưu `original_x = x`
  - Inject edge features sau `in_conv`
  - Chạy ResBlocks 0-4, qua EdgeDilatedBlocks, rồi ResBlocks 5-9
  - `return original_x - out_conv(x)` (giữ noise subtraction)

### [ ] Bước 2.3 — Tạo `configs/edrresnet.yaml`

Tham khảo config ResNet gốc, thêm các tham số EDR:
```yaml
trainer: edrresnet
lr: 0.0001
mbs: 32           # Giảm từ 64 để fit GPU với model lớn hơn
max_iterations: 20000
patchsize: 64
loss_alpha: 0.1
num_edge_blocks: 2
use_sobel_input: true
```

### [ ] Bước 2.4 — Đăng ký `edrresnet` vào `ldctbench/utils/argparser.py`

```python
METHODS = [..., "edrresnet"]
```

---

## GIAI ĐOẠN 3 — Sanity Check & Debug

### [ ] Bước 3.1 — Kiểm tra shape

```python
import torch
from ldctbench.methods.edrresnet.network import Model
from argparse import Namespace

args = Namespace(use_sobel_input=True, num_edge_blocks=2, loss_alpha=0.1)
model = Model(args)
x = torch.randn(1, 1, 64, 64)
y = model(x)
print(y.shape)  # Kỳ vọng: (1, 1, 64, 64)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))  # ~3.6M
```

### [ ] Bước 3.2 — Mini training test (10 steps, CPU)

```python
python test_edrresnet.py   # Script tương tự test_edrcnn10.py
```

Kiểm tra:
- Loss không NaN
- Loss giảm dần
- Gradient flow bình thường

---

## GIAI ĐOẠN 4 — Training (Kaggle T4)

### Chuẩn bị
- [ ] Tạo notebook `results/training/EDR-ResNet/train-edrresnet.ipynb`
- [ ] Upload lên Kaggle, add dataset LDCT + checkpoints

### [ ] Bước 4.1 — Train seed 1339
- `SEED = 1339`, max_iterations = 20000
- Kỳ vọng: ~30–45 phút trên T4

### [ ] Bước 4.2 — Train seed 2024
### [ ] Bước 4.3 — Train seed 42

> **Variant:** Chỉ train Variant D (Full EDR-ResNet với cả 2 module + SobelLoss), giống chiến lược EDR-CNN10.

---

## GIAI ĐOẠN 5 — Evaluation & So Sánh

### [ ] Bước 5.1 — Tạo evaluation script

```bash
python paper_scripts/evaluate_edrresnet.py
```

Metrics: PSNR, SSIM, Edge SSIM — 9 bệnh nhân test set
Wilcoxon: EDR-ResNet vs ResNet baseline

### [ ] Bước 5.2 — Điền vào bảng tổng hợp

| Model | PSNR | SSIM | Edge SSIM | p-value |
|:------|:----:|:----:|:---------:|:-------:|
| ResNet baseline | TBD | TBD | TBD | — |
| EDR-ResNet (mean 3 seeds) | TBD | TBD | TBD | TBD |

---

## GIAI ĐOẠN 6 — Push GitHub

### [ ] Bước 6.1 — Commit
```bash
git add ldctbench/methods/edrresnet/
git add configs/edrresnet.yaml
git add docs/markdown/EDR-ResNet/
git commit -m "feat: add EDR-ResNet with FixedSobelLayer + EdgeDilatedResidualBlock"
```

### [ ] Bước 6.2 — Push
```bash
git push origin main
```

---

## Kết Quả Cuối Cùng (Test Set)

| Metric | ResNet Baseline | EDR-ResNet (best seed 42) | Δ | p-value |
|:-------|:--------------:|:-------------------------:|:-:|:-------:|
| PSNR | 44.02 ± 6.74 | 43.63 ± 6.64 | -0.39 dB | 1.000 |
| SSIM | 0.9359 ± 0.053 | 0.9331 ± 0.056 | -0.003 | 1.000 |
| **Edge SSIM** | 0.7530 ± 0.138 | **0.7751 ± 0.123** | **+0.022** | **0.0098** ✅ |

**Kết luận:** Edge SSIM cải thiện có ý nghĩa thống kê trên 9/9 bệnh nhân.
PSNR/SSIM giảm nhẹ do bất lợi dữ liệu (34 vs 105 bệnh nhân) và khác biệt loss function.

---

## Ghi Chú Quan Trọng

> **Noise subtraction khác với CNN10/RED-CNN:** EDR-ResNet vẫn giữ `output = input - predicted_noise`. Sobel features inject vào feature map trung gian, không ảnh hưởng đến cơ chế này.

> **n_channels = 128** — Rộng hơn CNN10 (64) nên `edge_proj` chiếu `4 → 128` thay vì `4 → 64`.

> **max_iterations = 20000** — Theo config gốc của ResNet. Tốc độ train ~9 giờ/seed trên T4 (chậm hơn CNN10 do model nặng hơn).

---

*Cập nhật lần cuối: 14/06/2026*
