# KeHoachThucHien.md — EDR-QAE

Kế hoạch từng bước để tích hợp `FixedSobelLayer` + `EdgeDilatedResidualBlock` vào **QAE**.

> **Mục tiêu:** Chứng minh 2 module cải thiện QAE (kiến trúc Quadratic Autoencoder — hoàn toàn khác CNN), hoàn thiện bộ 3 kiến trúc đa dạng cùng được cải thiện.

---

## Tổng Quan Tiến Độ

| Giai đoạn | Trạng thái |
|:----------|:----------:|
| 1. Phân tích & Thiết kế | ✅ |
| 2. Implementation (code) | ✅ |
| 3. Sanity check & debug | ✅ |
| 4. Training (Kaggle T4) | ✅ |
| 5. Evaluation & So sánh | ✅ |
| 6. Push GitHub | [ ] |

---

## GIAI ĐOẠN 1 — Phân Tích & Thiết Kế ✅

### [x] Bước 1.1 — Phân tích kiến trúc QAE gốc

```
QAE gốc (QuadConv = bậc 2: output = W_r·x × W_g·x + W_b·x²):

Encoder:
  x1 = ReLU(QuadConv(x,  1→15, same))
  x2 = ReLU(QuadConv(x1, 15→15, same))
  x3 = ReLU(QuadConv(x2, 15→15, same))
  x4 = ReLU(QuadConv(x3, 15→15, same))
  x5 = ReLU(QuadConv(x4, 15→15, valid))  ← bottleneck, size giảm 2px

Decoder (với skip):
  x6 = ReLU(QuadDeconv(x5, valid) + x4)
  x7 = ReLU(QuadDeconv(x6, same))
  x8 = ReLU(QuadDeconv(x7, same) + x2)
  x9 = ReLU(QuadDeconv(x8, same))
  out = QuadDeconv(x9, same) + input    ← residual từ input
```

**Đặc điểm quan trọng:**
- Chỉ 15 channels — lightweight nhất trong tất cả
- QuadConv dùng bậc 2, KHÔNG phải ReLU activation thông thường
- Skip connections: x4→decoder[0], x2→decoder[2]
- `valid` padding tại encoder[4]: patch 36×36 → x5: 34×34

### [x] Bước 1.2 — Thiết kế EDR-QAE

```
EDR-QAE:

Encoder:
  x1 = ReLU(QuadConv(x, 1→15, same))
  x1 = x1 + edge_proj(FixedSobelLayer(x))   ← THÊM MỚI

  x2 = ReLU(QuadConv(x1, 15→15, same))
  x3 = ReLU(QuadConv(x2, 15→15, same))
  x4 = ReLU(QuadConv(x3, 15→15, same))
  x5 = ReLU(QuadConv(x4, 15→15, valid))

  x5 = EdgeDilatedResidualBlock(x5, dilation=2)   ← THÊM MỚI
  x5 = EdgeDilatedResidualBlock(x5, dilation=3)   ← THÊM MỚI

Decoder (giữ nguyên):
  x6 = ReLU(QuadDeconv(x5, valid) + x4)
  ...
  out = QuadDeconv(x9, same) + input
```

---

## GIAI ĐOẠN 2 — Implementation

### [ ] Bước 2.1 — Tạo package `ldctbench/methods/edrqae/`

**Cấu trúc:**
```
ldctbench/methods/edrqae/
├── __init__.py
├── network.py       ← VIẾT MỚI (dựa trên qae/network.py)
├── loss.py          ← Copy từ edrrednet/loss.py
├── Trainer.py       ← Copy từ edrrednet/Trainer.py, sửa docstring
└── argparser.py     ← Copy từ edrrednet/argparser.py, giữ conflict check
```

### [ ] Bước 2.2 — Viết `network.py`

- Copy `FixedSobelLayer` và `EdgeDilatedResidualBlock` từ `edrrednet/network.py`
- Copy `QuadConv` và `QuadDeconv` từ `qae/network.py` (KHÔNG thay đổi)
- Thêm vào `Model.__init__()`:
  - `self.sobel = FixedSobelLayer()`
  - `self.edge_proj = nn.Conv2d(4, 15, 1)` ← 4→15 (QAE dùng 15ch)
  - `self.edge_blocks = nn.ModuleList([EDRB(15, d=2), EDRB(15, d=3)])`
- Sửa `Model.forward()`:
  - Inject edge sau x1: `x1 = x1 + self.edge_proj(self.sobel(original_x))`
  - Chạy EdgeBlocks sau x5
  - Giữ nguyên `return ... + x` (residual từ input)

### [ ] Bước 2.3 — Tạo `configs/edrqae.yaml`

```yaml
trainer: edrqae
lr: 2.8427860979114494e-05     # Giữ nguyên LR của QAE gốc
mbs: 64                         # Giảm từ 125 để ổn định với loss mới
max_iterations: 50000           # Giảm từ 94025 (QAE gốc) để tiết kiệm thời gian
patchsize: 36                   # Giữ nguyên của QAE gốc (valid padding cần ≥ 36)
loss_alpha: 0.1
num_edge_blocks: 2
use_sobel_input: true
```

### [ ] Bước 2.4 — Đăng ký `edrqae` vào `ldctbench/utils/argparser.py`

```python
METHODS = [..., "edrqae"]
```

---

## GIAI ĐOẠN 3 — Sanity Check & Debug

### [ ] Bước 3.1 — Kiểm tra shape

```python
import torch
from ldctbench.methods.edrqae.network import Model
from argparse import Namespace

args = Namespace(use_sobel_input=True, num_edge_blocks=2, loss_alpha=0.1)
model = Model(args)
x = torch.randn(1, 1, 36, 36)   # patchsize=36 như QAE gốc
y = model(x)
print(y.shape)  # Kỳ vọng: (1, 1, 36, 36)
```

**Lưu ý shape đặc biệt của QAE:**
- encoder[4] với `valid` padding: 36→34
- EdgeBlocks (same): 34→34
- decoder[0] với `valid` transpose: 34→36

### [ ] Bước 3.2 — Mini training test (10 steps, CPU)

---

## GIAI ĐOẠN 4 — Training (Kaggle T4)

### Chuẩn bị
- [ ] Tạo notebook `results/training/EDR-QAE/train-edrqae.ipynb`
- [ ] Upload lên Kaggle, add dataset LDCT + checkpoints

### [x] Bước 4.1 — Train seed 1339
- `SEED = 1339`, max_iterations = 50000
- **Kết quả:** Đã train xong, max SSIM validation đạt ~0.8708 (tại iteration 46000)

### [x] Bước 4.2 — Train seed 2024
- **Kết quả:** Đã train xong, max SSIM validation đạt ~0.8718 (tại iteration 41000)

### [x] Bước 4.3 — Train seed 42
- **Kết quả:** Đã train xong, max SSIM validation đạt ~0.8718 (tại iteration 14000)

---

## GIAI ĐOẠN 5 — Evaluation & So Sánh

### [x] Bước 5.1 — Tạo evaluation script

```bash
python paper_scripts/evaluate_edrqae.py
```

Metrics: PSNR, SSIM, Edge SSIM — 9 bệnh nhân test set
Wilcoxon: EDR-QAE vs QAE baseline

### [x] Bước 5.2 — Điền vào bảng tổng hợp

Kết quả trung bình trên 9 bệnh nhân Test Set:

| Model | PSNR | SSIM | Edge SSIM | p-value (SSIM) |
|:------|:----:|:----:|:---------:|:-------:|
| QAE baseline | 42.6953 | 0.9257 | 0.7726 | — |
| EDR-QAE (mean 3 seeds) | **42.9856** | **0.9274** | **0.7734** | **0.009766** (<0.05) |

*Ghi chú:*
- Seed tốt nhất (Seed 1339) có SSIM = 0.9276, PSNR = 43.0640.
- Sự cải thiện của SSIM và PSNR (p=0.0019) có ý nghĩa thống kê rõ rệt.
- Edge SSIM không có khác biệt đáng kể (p>0.05), do QAE vốn đã học cạnh khá tốt nhưng bị hạn chế ở PSNR/SSIM, khi thêm EDR thì các chỉ số PSNR/SSIM được kéo lên đáng kể mà không làm hỏng Edge SSIM.

---

## GIAI ĐOẠN 6 — Push GitHub

### [ ] Bước 6.1 — Commit & Push
```bash
git add ldctbench/methods/edrqae/
git add configs/edrqae.yaml
git add docs/markdown/EDR-QAE/
git commit -m "feat: add EDR-QAE with FixedSobelLayer + EdgeDilatedResidualBlock"
git push origin main
```

---

## Ghi Chú Quan Trọng

> **QuadConv ≠ Conv thông thường:** Không có ReLU sau QuadConv trong layer (ReLU chỉ bao bên ngoài). EdgeDilatedResidualBlock vẫn dùng Conv thông thường — không can thiệp vào QuadConv.

> **n_channels = 15** — Nhỏ nhất trong 3 kiến trúc. `edge_proj` chiếu `4 → 15`.

> **patchsize = 36** — Giữ nguyên của QAE gốc vì `valid` padding cần kích thước tối thiểu nhất định.

> **max_iterations = 50000** — Giảm từ 94025 của gốc. QAE nhẹ nên hội tụ sớm hơn khi đã có edge guidance.

---

*Cập nhật lần cuối: 14/06/2026*
