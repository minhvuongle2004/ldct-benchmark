# Kế Hoạch Thực Hiện: EDR-REDNet

> **Mô hình đề xuất:** Edge-Dilated Residual RED-CNN (EDR-REDNet)  
> **Dựa trên bài báo:** Gholizadeh-Ansari et al., *"Deep Learning for Low-Dose CT Denoising Using Perceptual Loss and Edge Detection Layer"*, J. Digital Imaging, 2020. [DOI: 10.1007/s10278-019-00274-4](https://doi.org/10.1007/s10278-019-00274-4)  
> **Framework nền:** [`ldct-benchmark`](https://github.com/eeulig/ldct-benchmark) (Eulig et al., Medical Physics 2024)  
> **Dataset:** LDCT-and-Projection-data (TCIA) — **Chest / Abdomen** (100 scans, 50 mỗi nhóm)  
> *(Head data bị restricted access tại TCIA → loại khỏi phạm vi. Liver là organ/ROI trong Abdomen, không phải exam type riêng)*

---

## Tổng Quan Ý Tưởng

RED-CNN và các mô hình denoising dùng pixel-wise loss dễ bị **over-smoothing**, đặc biệt ở các cấu trúc giải phẫu nhỏ và biên low-contrast (mạch máu nhỏ, cạnh tổ chức). Điều này thúc đẩy việc tích hợp **edge prior** và **dilated residual refinement** để cải thiện *small-structure preservation*.

> ⚠️ **Lưu ý:** Không claim "mất nốt phổi" vì benchmark hiện tại không có nodule annotation. Chỉ nói **small structure / vascular boundary preservation**.

Bài báo gốc đề xuất:

```
Dilated Conv (multi-rate) + Residual Learning + Edge Detection Layer + Perceptual Loss
```

EDR-REDNet kế thừa kiến trúc RED-CNN hiện có trong repo và bổ sung **4 thành phần** (3 bắt buộc + 1 optional):

| Thành phần | Bắt buộc? | Mô tả | File |
|---|---|---|---|
| **FixedSobelLayer** | ✅ Bắt buộc | Non-trainable layer trích xuất edge map (H/V/Diagonal) — đưa vào network như input phụ | `network.py` (mới) |
| **EdgeDilatedResidualBlock** | ✅ Bắt buộc | Dilated conv (rate=2,3) + residual ở bottleneck | `network.py` (mới) |
| **SobelEdgeLoss** | ✅ Bắt buộc | Ép output giữ gradient giống NDCT | `loss.py` (mới) |
| **Perceptual Loss (VGG)** | ⚪ Optional | Ablation only — VGG train trên ảnh tự nhiên, cần kiểm chứng với CT | `loss.py` (mới) |

---

## Các Bước Thực Hiện

### Bước 1 — Xác nhận baseline RED-CNN đang chạy được ✅

> **Mục tiêu:** Đảm bảo môi trường và data pipeline hoạt động trước khi thêm code mới.

**Kiểm tra:**
```bash
# Chạy thử RED-CNN trong 100 iterations (dryrun)
python -m ldctbench.scripts.train --config configs/redcnn.yaml --dryrun true --max_iterations 100
```

**Kết quả mong đợi:** In ra loss sau mỗi 100 iterations, không lỗi shape hoặc CUDA.

**Nếu chưa train lần nào:** Chạy đủ RED-CNN baseline trước (seed 1339 theo config).

---

### Bước 2 — Tạo thư mục method mới: `edrrednet`

> **Mục tiêu:** Tạo module model mới theo đúng cấu trúc của repo, không sửa code gốc.

**Cấu trúc cần tạo:**
```
ldctbench/methods/edrrednet/
├── __init__.py
├── network.py      ← kiến trúc EDR-REDNet
├── loss.py         ← SobelEdgeLoss + PerceptualLoss
├── Trainer.py      ← tích hợp loss kết hợp
└── argparser.py    ← tham số riêng (alpha, beta, gamma)
```

**Copy từ RED-CNN làm nền:**
```powershell
Copy-Item -Recurse ldctbench\methods\redcnn ldctbench\methods\edrrednet
```

---

### Bước 3 — Viết `network.py`: Kiến trúc EDR-REDNet

> **Mục tiêu:** Thêm `FixedSobelLayer` + `EdgeDilatedResidualBlock` vào RED-CNN.

**3a. FixedSobelLayer** (non-trainable, bắt buộc — theo đúng bài báo gốc):
```python
class FixedSobelLayer(nn.Module):
    """
    Non-trainable layer trích xuất biên theo 4 hướng: H, V, Diagonal 45°, Diagonal 135°.
    Weights không được cập nhật trong quá trình train (requires_grad=False).
    Output: edge map cùng kích thước với input, dùng như thông tin phụ trợ cho mạng.
    """
    def __init__(self):
        super().__init__()
        # Định nghĩa 4 Sobel kernels cố định
        kernels = [...]  # Horizontal, Vertical, Diag45, Diag135
        self.register_buffer('weight', kernels)  # không train

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1)  # shape: (B, 4, H, W)
```

**3b. EdgeDilatedResidualBlock** (bắt buộc):
```python
class EdgeDilatedResidualBlock(nn.Module):
    """
    Dilated Residual Block nhận biết biên.
    - Conv dilated rate=2: mở rộng receptive field mà không tăng params
    - Conv dilated rate=3: nắm bắt ngữ cảnh rộng hơn (mạch máu, cạnh cấu trúc nhỏ)
    - Residual connection: bảo toàn thông tin gốc
    """
    def __init__(self, channels, dilation=2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=1, bias=False)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))
```

**Luồng dữ liệu tổng thể:**
```
Input (LDCT) ──┬──────────────────────────────────────────────────────┐
               │                                                      │
               ↓                                                      │
         FixedSobelLayer → edge_map (4 channels)                      │
               │                                                      │
               ↓ (concat hoặc add vào bottleneck feature)             │
         [Encoder: Conv1→Conv5]                                       │
               ↓                                                      │
         [EdgeBlock d=2] → [EdgeBlock d=3]  ← bottleneck              │
               ↓                                                      │
         [Decoder: TConv1→TConv5]                                     │
               ↓                                                      │
          Output + residual_1 ←─────────────────────────────────────-─┘
```

**Test nhanh (phải pass trước khi train):**
```python
import torch
from ldctbench.methods.edrrednet.network import Model

model = Model(args=None)
x = torch.randn(2, 1, 64, 64)
y = model(x)
assert y.shape == (2, 1, 64, 64), f"Shape lỗi: {y.shape}"
print("✅ Forward pass OK:", y.shape)
```

---

### Bước 4 — Viết `loss.py`: CombinedLoss

> **Mục tiêu:** Implement loss kết hợp theo từng version, không dùng MSE đơn thuần.

**Lộ trình thêm loss theo version (an toàn hơn):**

| Version | Loss | Ghi chú |
|---|---|---|
| V1 | `Charbonnier + SobelEdgeLoss` | **Bắt đầu từ đây** — ổn định, không cần VGG |
| V2 | V1 + `SSIM loss` | Thêm structure awareness |
| V3 | V2 + `HU loss` | Thêm clinical consistency |
| V4 | V3 + `Perceptual (VGG)` | **Ablation only** — không bắt buộc |

**SobelEdgeLoss** (bắt buộc từ V1):
```python
class SobelEdgeLoss(nn.Module):
    """
    Tính L1 loss giữa gradient Sobel của output và target.
    Ép mô hình giữ biên theo cả hướng ngang, dọc, chéo.
    """
    # Kernels: Horizontal, Vertical, Diagonal (45°, 135°)
    # forward: ||Sobel(pred) - Sobel(target)||_1
```

**Loss tổng hợp V1 (mặc định khi train lần đầu):**
```
L_total = L_Charbonnier(pred, target)     # thay MSE, ít over-smooth hơn
        + α × L_Sobel(pred, target)        # giữ biên (α ≈ 0.1)
```

**HU Loss — cần định nghĩa đúng:**
```python
# ⚠️ Data đã normalize meanstd → pred/target KHÔNG ở thang HU gốc
# Phải inverse normalization trước khi tính HU loss
def hu_loss(pred_norm, target_norm, mean, std):
    pred_hu = pred_norm * std + mean    # chuyển về HU
    target_hu = target_norm * std + mean
    return F.l1_loss(pred_hu, target_hu)
# Nếu không inverse → L_HU chỉ là L1 thông thường, không có ý nghĩa lâm sàng
```

**Perceptual Loss (VGG) — chỉ dùng trong ablation:**
```python
# ⚠️ VGG được train trên ImageNet (ảnh tự nhiên RGB)
# CT là grayscale/HU, low-contrast → perceptual loss có thể gây sai lệch
# Dùng để so sánh trong ablation, không đưa vào model chính
# Trong bài báo ghi: "Perceptual loss is evaluated as an auxiliary
#  component in the ablation study rather than being assumed to be
#  universally beneficial for CT images."
```

> ⚠️ **Lưu ý:** Không dùng BatchNorm gần output — ảnh hưởng phân bố HU trong CT.

---

### Bước 5 — Viết `Trainer.py`: Tích hợp vào training loop

> **Mục tiêu:** Override hàm `train_step` để dùng loss kết hợp thay vì MSE.

**Thay đổi so với RED-CNN Trainer:**
```python
# Thay self.criterion = nn.MSELoss()
# Bằng:
self.criterion = CombinedLoss(alpha=0.1, beta=0.01, gamma=0.01)

# Trong train_step:
loss = self.criterion(pred, target)  # tự động tính tổng 4 thành phần
```

**Log thêm từng thành phần loss** để ablation study sau:
```python
self.log({
    "loss/total": total_loss,
    "loss/charbonnier": l_charb,
    "loss/sobel": l_edge,
    "loss/perceptual": l_perceptual,
})
```

---

### Bước 6 — Tạo `configs/edrrednet.yaml`

> **Mục tiêu:** Cấu hình training theo 3 giai đoạn — không dùng hyperparameter RED-CNN ngay cho final run.

```yaml
# configs/edrrednet.yaml  (giai đoạn PILOT — chạy thử)
trainer: edrrednet
seed: 1339

# Optimizer
optimizer: adam
lr: 9.583e-05
adam_b1: 0.9
adam_b2: 0.999

# Loss weights (V1: chỉ Charbonnier + Sobel)
loss_alpha: 0.1        # Sobel edge weight
# loss_beta: 0.01      # Perceptual — BẬT khi ablation
# loss_gamma: 0.01     # HU loss — BẬT sau khi định nghĩa inverse norm

# Training — 3 giai đoạn:
# Debug:  mbs=8,  max_iterations=2000
# Pilot:  mbs=16, max_iterations=20000
# Final:  mbs=16, max_iterations=92994  (giống RED-CNN để so sánh)
mbs: 8                 # bắt đầu nhỏ để kiểm tra VRAM
max_iterations: 2000   # tăng dần
patchsize: 128
iterations_before_val: 500

# Hyperparameter grid nhỏ (nếu có thời gian):
# loss_alpha in [0.01, 0.05, 0.1]
# num_edge_blocks in [1, 2]
# dilation in [(2), (2,3)]

# Data
data_norm: meanstd
data_subset: 0.1       # bắt đầu với 10% data → tăng lên 1.0 khi stable
num_workers: 8
cuda: true
devices: 0
```

---

### Bước 7 — Đăng ký model vào hệ thống

> **Mục tiêu:** Để lệnh `--trainer edrrednet` hoạt động.

**Tìm nơi đăng ký:**
```powershell
Select-String -Path "ldctbench\**\*.py" -Pattern "trainer.*redcnn|get_trainer|available" -Recurse
```

**Thêm vào registry** (thường ở `ldctbench/scripts/train.py` hoặc `base.py`):
```python
from ldctbench.methods.edrrednet.Trainer import Trainer as EDRREDNetTrainer
TRAINERS = {
    ...,
    "edrrednet": EDRREDNetTrainer,
}
```

---

### Bước 8 — Train thử subset nhỏ (sanity check)

> **Mục tiêu:** Bắt lỗi shape/loss trước khi train full.

```bash
python -m ldctbench.scripts.train \
  --config configs/edrrednet.yaml \
  --data_subset 0.1 \
  --max_iterations 500 \
  --dryrun false
```

**Kiểm tra:**
- [ ] Loss giảm trong 500 iterations đầu
- [ ] Không có `NaN` trong loss
- [ ] VRAM không vượt quá giới hạn GPU
- [ ] Shape output = shape input

**📁 Lưu kết quả vào:**
```
results/sanity_check/
├── loss_curve_500iter.csv     ← loss theo từng iteration
└── sanity_check_notes.md      ← ghi chú: GPU VRAM, thời gian/iter, có lỗi không
```

---

### Bước 9 — Train đầy đủ + 3 seeds

> **Mục tiêu:** Kết quả có ý nghĩa thống kê (Confidence Interval).

```bash
# Seed 1
python -m ldctbench.scripts.train --config configs/edrrednet.yaml --seed 1339

# Seed 2  
python -m ldctbench.scripts.train --config configs/edrrednet.yaml --seed 2024

# Seed 3
python -m ldctbench.scripts.train --config configs/edrrednet.yaml --seed 42
```

**Thời gian ước tính:** ~6-12h/seed trên GPU (Kaggle T4/P100).

**📁 Lưu kết quả vào:**
```
results/training/
├── seed1339/
│   ├── best_model.pth          ← checkpoint tốt nhất (download từ Kaggle)
│   ├── train_log.csv           ← loss/val_metric theo iteration
│   └── config_used.yaml        ← bản config thực tế đã dùng
├── seed2024/
│   └── (tương tự)
└── seed42/
    └── (tương tự)
```

---

### Bước 10 — Đánh giá và so sánh kết quả

> **Mục tiêu:** Bảng kết quả công bằng với 8 baseline hiện có, trên 100 bệnh nhân (Chest + Abdomen).

**Chạy evaluation:**
```bash
python -m ldctbench.scripts.evaluate \
  --checkpoint runs/edrrednet/seed1339/best_model.pth \
  --config configs/edrrednet.yaml
```

**Metrics cần báo cáo (theo thứ tự ưu tiên):**

**Nhóm bắt buộc** (reviewer luôn hỏi):

| Metric | Mô tả | Tool |
|---|---|---|
| PSNR | Peak Signal-to-Noise Ratio | `ldctbench.evaluate.utils` |
| SSIM | Structural Similarity Index | `ldctbench.evaluate.utils` |
| RMSE | Root Mean Squared Error | `ldctbench.evaluate.utils` |
| **VIF** | Visual Information Fidelity — có ý nghĩa lâm sàng hơn PSNR | `ldctbench.evaluate.utils` |

**Nhóm rất nên có** (phân biệt bài tốt/bình thường):

| Metric | Mô tả | Tool |
|---|---|---|
| **Edge SSIM** | SSIM tính trên Sobel edge map — đo khả năng giữ biên | Custom |
| **CNR** | Contrast-to-Noise Ratio (vùng ROI theo anatomy) | Custom |
| **LDCT-IQA** | No-reference IQA đặc thù CT *(nếu module có sẵn)* | `ldctbench.evaluate.ldct_iqa` |

> ⚠️ Kiểm tra `ldctbench.evaluate.ldct_iqa` có tồn tại trong repo trước khi đưa vào kết quả chính thức.

**📁 Lưu kết quả vào:**
```
results/evaluation/
├── per_seed/
│   ├── metrics_seed1339.csv    ← PSNR/SSIM/RMSE/VIF/EdgeSSIM/CNR từng ảnh
│   ├── metrics_seed2024.csv
│   └── metrics_seed42.csv
├── summary_mean_std.csv        ← Mean ± Std của 3 seeds (dùng trực tiếp vào bảng bài báo)
└── comparison_table.csv        ← EDR-REDNet vs 8 baseline (copy từ ldctbench results)
```

---

### Bước 11 — Ablation Study

> **Mục tiêu:** Chứng minh từng thành phần đóng góp, bắt buộc để bài được review chấp nhận.

| Variant | FixedSobelLayer | EdgeBlock | Sobel Loss | Perceptual Loss | Ghi chú |
|---|---|---|---|---|---|
| A | ❌ | ❌ | ❌ | ❌ | RED-CNN baseline |
| B | ❌ | ✅ | ❌ | ❌ | Chỉ thêm EdgeBlock |
| C | ✅ | ✅ | ❌ | ❌ | + FixedSobelLayer trong network |
| D | ✅ | ✅ | ✅ | ❌ | **EDR-REDNet đầy đủ** (model chính) |
| E | ✅ | ✅ | ✅ | ✅ | + Perceptual VGG (ablation optional) |

> 💡 Nếu tài nguyên hạn chế: 4 variant A–D là đủ cho bài trong nước/JST-ICT.
> Nếu muốn bài mạnh hơn: nên có đủ 5 variant A–E.

**📁 Lưu kết quả vào:**
```
results/ablation/
├── variant_A_redcnn/
│   └── metrics.csv             ← dùng kết quả baseline đã có
├── variant_B_edgeblock/
│   └── metrics.csv
├── variant_C_edgeblock_sobel_input/
│   └── metrics.csv
├── variant_D_full_edrrednet/
│   └── metrics.csv             ← giống results/evaluation/
├── variant_E_with_perceptual/  ← optional
│   └── metrics.csv
└── ablation_summary.csv        ← bảng so sánh tất cả variant (dùng vào bài báo)
```

---

### Bước 12 — Visual Results + Figures

> **Mục tiêu:** Hình minh họa cho bài báo.

**Hình cần có:**
1. Sơ đồ kiến trúc EDR-REDNet (encoder → EdgeBlock → decoder)
2. Hình so sánh: LDCT / RED-CNN / EDR-REDNet / NDCT (reference)
3. Zoom vào vùng Chest (prefix C) — cấu trúc phổi, mạch máu
4. Difference map (ảnh − NDCT)
5. Sobel edge map comparison
6. Ablation visual (4 variants A/B/C/D)
7. Boxplot PSNR/SSIM theo 3 seeds

---

## Tóm Tắt Timeline (10–12 Tuần)

> 8 tuần có thể ra bản **workshop/draft**, nhưng để bài chắc hơn nên dự trù **10–12 tuần**.

> ✅ **Cập nhật phương án dataset:** Dùng **100 bệnh nhân (50 Chest + 50 Abdomen)** thay vì 150. Head data bị TCIA restricted access. 100 bệnh nhân vẫn đủ để train và đảm bảo CI với 3 seeds.

| Tuần | Bước | Nội dung |
|---|---|---|
| 1 | 1–2 | Xác nhận baseline RED-CNN, tạo thư mục `edrrednet` |
| 2 | 3 | Viết `network.py` (FixedSobelLayer + EdgeBlock), test forward pass |
| 3 | 4–5 | Viết `loss.py` V1 (Charbonnier + Sobel) + `Trainer.py` |
| 4 | 6–7 | Viết config, đăng ký model, sanity check debug (mbs=8, 2000 iter) |
| 5 | 8 | Pilot train (mbs=16, 20000 iter, subset 50%) — kiểm tra PSNR/SSIM |
| 6–7 | 9 | Train đầy đủ 3 seeds (92994 iter, full data — 100 BN) |
| 8 | 10 | Đánh giá metrics (PSNR/SSIM/RMSE/VIF/EdgeSSIM/CNR) |
| 9 | 11 | Ablation study 4–5 variant |
| 10–12 | 12 | Vẽ hình, viết bài, chỉnh sửa |

---

## Cấu Trúc Thư Mục `results/` (Số Liệu Bài Báo)

> 📌 **Tất cả số liệu dùng cho bài báo đều lưu tại:** `ldct-benchmark/results/`

```
results/
├── sanity_check/                   ← Bước 8
│   ├── loss_curve_500iter.csv
│   └── sanity_check_notes.md
│
├── training/                       ← Bước 9
│   ├── seed1339/
│   │   ├── best_model.pth
│   │   ├── train_log.csv
│   │   └── config_used.yaml
│   ├── seed2024/
│   └── seed42/
│
├── evaluation/                     ← Bước 10
│   ├── per_seed/
│   │   ├── metrics_seed1339.csv
│   │   ├── metrics_seed2024.csv
│   │   └── metrics_seed42.csv
│   ├── summary_mean_std.csv        ← ⭐ Bảng chính cho bài báo
│   └── comparison_table.csv        ← So sánh với 8 baseline
│
├── ablation/                       ← Bước 11
│   ├── variant_A_redcnn/
│   ├── variant_B_edgeblock/
│   ├── variant_C_edgeblock_sobel_input/
│   ├── variant_D_full_edrrednet/
│   ├── variant_E_with_perceptual/  ← optional
│   └── ablation_summary.csv        ← ⭐ Bảng ablation cho bài báo
│
└── figures/                        ← Bước 12
    ├── architecture_diagram.*
    ├── visual_comparison/
    ├── difference_maps/
    ├── edge_maps/
    └── boxplot_psnr_ssim.*
```

---

## Cấu Trúc File Sẽ Thêm Vào Repo

```
ldct-benchmark/
├── configs/
│   └── edrrednet.yaml              ← [MỚI] Config training
├── ldctbench/methods/
│   └── edrrednet/
│       ├── __init__.py             ← [MỚI]
│       ├── network.py              ← [MỚI] EDR-REDNet + EdgeDilatedResidualBlock
│       ├── loss.py                 ← [MỚI] SobelEdgeLoss + CombinedLoss
│       ├── Trainer.py              ← [MỚI] Training loop với combined loss
│       └── argparser.py            ← [MỚI] alpha, beta, gamma params
└── docs/markdown/
    └── EDRREDNet_KeHoachThucHien.md ← File này
```

---

## Tài Liệu Tham Khảo

1. **Gholizadeh-Ansari et al. (2020)** — Bài báo gốc ER-Net: dilated conv + edge detection + perceptual loss  
   DOI: [`10.1007/s10278-019-00274-4`](https://doi.org/10.1007/s10278-019-00274-4)

2. **Eulig et al. (2024)** — ldct-benchmark: framework benchmark chuẩn  
   DOI: [`10.1002/mp.17379`](https://doi.org/10.1002/mp.17379)

3. **Chen et al. (2017)** — RED-CNN gốc  
   DOI: [`10.1109/TMI.2017.2715284`](https://doi.org/10.1109/TMI.2017.2715284)

4. **Yu & Koltun (2015)** — Dilated convolutions  
   arXiv: [`1511.07122`](https://arxiv.org/abs/1511.07122)

5. **Johnson et al. (2016)** — Perceptual Loss  
   ECCV 2016
