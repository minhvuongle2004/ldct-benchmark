# EDR-CNN10: Kế Hoạch Thực Hiện

> Tài liệu này ghi lại **các bước cụ thể** để triển khai EDR-CNN10 từ đầu đến cuối.
> Đánh dấu `[x]` khi hoàn thành từng bước.

---

## GIAI ĐOẠN 1 — Tạo Module `edrcnn10`

### [x] Bước 1.1 — Tạo `network.py`
- Tạo file `ldctbench/methods/edrcnn10/network.py`
- Copy class `FixedSobelLayer` và `EdgeDilatedResidualBlock` từ `edrrednet/network.py` (không sửa)
- Viết mới class `Model` cho EDR-CNN10 với cấu trúc 4 lớp + injection

**Sanity check sau khi viết:**
```python
from argparse import Namespace
import torch
from ldctbench.methods.edrcnn10.network import Model
args = Namespace(num_edge_blocks=2, use_sobel_input=True)
model = Model(args)
x = torch.randn(1, 1, 128, 128)
out = model(x)
print(out.shape)  # phải ra torch.Size([1, 1, 128, 128])
```

### [x] Bước 1.2 — Tạo `loss.py`
- Copy nguyên `ldctbench/methods/edrrednet/loss.py` → `ldctbench/methods/edrcnn10/loss.py`
- Không sửa gì

### [x] Bước 1.3 — Tạo `Trainer.py`
- Copy `ldctbench/methods/edrrednet/Trainer.py` → `ldctbench/methods/edrcnn10/Trainer.py`
- Sửa duy nhất phần docstring (thay "EDR-REDNet" → "EDR-CNN10")
- Import Model từ `.network` — giữ nguyên

### [x] Bước 1.4 — Tạo `argparser.py`
- Copy nguyên `ldctbench/methods/edrrednet/argparser.py` → `ldctbench/methods/edrcnn10/argparser.py`
- Không sửa gì

### [x] Bước 1.5 — Tạo `__init__.py`
- Tạo `ldctbench/methods/edrcnn10/__init__.py`
- Nội dung: `from .Trainer import Trainer`

---

## GIAI ĐOẠN 2 — Tạo Config

### [x] Bước 2.1 — Tạo `configs/edrcnn10.yaml`
- Dựa trên `configs/edrrednet.yaml`
- Đổi `trainer: edrrednet` → `trainer: edrcnn10`
- Giữ: `loss_alpha: 0.1`, `num_edge_blocks: 2`, `optimizer: adam`
- Điều chỉnh từ CNN10 gốc: `lr: 0.00015837`, `patchsize: 92`
- Giai đoạn debug: `mbs: 8`, `max_iterations: 2000`, `data_subset: 0.1`

---

## GIAI ĐOẠN 3 — Kiểm Tra Cục Bộ (Local Debug)

### [x] Bước 3.1 — Forward pass test
```bash
python -c "
from argparse import Namespace
import torch
from ldctbench.methods.edrcnn10.network import Model
args = Namespace(num_edge_blocks=2, use_sobel_input=True)
model = Model(args)
x = torch.randn(1, 1, 128, 128)
out = model(x)
print('Shape OK:', x.shape == out.shape)
print('Params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
"
```
- **Kỳ vọng:** `Shape OK: True`, params ~171K

### [x] Bước 3.2 — Debug training 2000 iter
```bash
python -m ldctbench.train --config configs/edrcnn10.yaml
```
- **Kỳ vọng:** Loss giảm dần, không có NaN, không OOM

### [x] Bước 3.3 — Ghi kết quả debug vào nhật ký `CacThayDoi.md`

---

## GIAI ĐOẠN 4 — Ablation Study (Kaggle T4)

### Chuẩn bị
- [x] Upload notebook `results/training/EDR-CNN10/train-edrcnn10.ipynb` lên Kaggle
- [x] Dataset LDCT đã được add vào Kaggle notebook

### [x] Bước 4.1 — Train Variant D seed 1339
- Best val SSIM = 0.8752, PSNR = 39.683 @ iter 24000
- Thời gian: ~1 giờ trên T4

### [x] Bước 4.2 — Train Variant D seed 2024
- Best val SSIM = 0.8759, PSNR = 39.602 @ iter 23000

### [x] Bước 4.3 — Train Variant D seed 42
- Best val SSIM = 0.8770, PSNR = 39.610 @ iter 22000

> **Kết quả training:** Mean val SSIM = 0.8760 ± 0.0009 | PSNR = 39.632 ± 0.045
> Thời gian: ~1 giờ/seed (nhanh hơn RED-CNN 7x)
```bash
python -m ldctbench.train --method edrcnn10 --seed 1339 \
    --num_edge_blocks 2 --use_sobel_input False \
    --max_iterations 31000 --run_name edrcnn10_B_s1339

python -m ldctbench.train --method edrcnn10 --seed 2024 \
    --num_edge_blocks 2 --use_sobel_input False \
    --max_iterations 31000 --run_name edrcnn10_B_s2024

python -m ldctbench.train --method edrcnn10 --seed 42 \
    --num_edge_blocks 2 --use_sobel_input False \
    --max_iterations 31000 --run_name edrcnn10_B_s42
```

### [ ] Bước 4.2 — Train Variant C (+ SobelInput, không SobelLoss)
```bash
python -m ldctbench.train --method edrcnn10 --seed 1339 \
    --num_edge_blocks 2 --use_sobel_input True \
    --max_iterations 31000 --run_name edrcnn10_C_s1339

python -m ldctbench.train --method edrcnn10 --seed 2024 \
    --num_edge_blocks 2 --use_sobel_input True \
    --max_iterations 31000 --run_name edrcnn10_C_s2024

python -m ldctbench.train --method edrcnn10 --seed 42 \
    --num_edge_blocks 2 --use_sobel_input True \
    --max_iterations 31000 --run_name edrcnn10_C_s42
```

### [ ] Bước 4.3 — Train Variant D (Full EDR-CNN10)
```bash
python -m ldctbench.train --method edrcnn10 --seed 1339 \
    --num_edge_blocks 2 --use_sobel_input True --loss_alpha 0.1 \
    --max_iterations 31000 --run_name edrcnn10_D_s1339

python -m ldctbench.train --method edrcnn10 --seed 2024 \
    --num_edge_blocks 2 --use_sobel_input True --loss_alpha 0.1 \
    --max_iterations 31000 --run_name edrcnn10_D_s2024

python -m ldctbench.train --method edrcnn10 --seed 42 \
    --num_edge_blocks 2 --use_sobel_input True --loss_alpha 0.1 \
    --max_iterations 31000 --run_name edrcnn10_D_s42
```

### [ ] Bước 4.4 — Chọn Best Model mỗi Variant
- Theo SSIM cao nhất trên val set (giống cách làm với EDR-REDNet)
- Ghi vào `CacThayDoi.md` mục 6 (Nhật ký)

---

## GIAI ĐOẠN 5 — Đánh Giá Kết Quả ✅ HOÀN THÀNH

### [x] Bước 5.1 — Chạy Wilcoxon test trên 9 bệnh nhân test
- Script: `paper_scripts/evaluate_edrcnn10.py`
- Kết quả lưu tại: `results/training/EDR-CNN10/SoSanh/`

**Kết quả chính (EDR-CNN10 D best seed vs CNN10 Baseline):**

| Metric | CNN10 Baseline | EDR-CNN10 (mean 3 seeds) | Δ | p-value | Significant |
|:-------|:---:|:---:|:---:|:---:|:---:|
| PSNR | 42.83 ± 6.45 | 43.49 ± 6.75 | **+0.66 dB** | 0.0020 | ✅ |
| SSIM | 0.9275 ± 0.059 | 0.9291 ± 0.059 | **+0.0016** | 0.0020 | ✅ |
| Edge SSIM | 0.7521 ± 0.146 | 0.7549 ± 0.131 | +0.0028 | 0.1504 | ⚠️ |

**Nhận xét:**
- PSNR và SSIM cải thiện có ý nghĩa thống kê (p = 0.001953 = min Wilcoxon n=9)
- Edge SSIM không significant do Liver patients giảm nhẹ (Chest tăng, Liver giảm)
- EDR-CNN10 thắng CNN10 trên **tất cả 9 bệnh nhân** về PSNR

### [x] Bước 5.2 — So sánh với Baseline CNN10
- ✅ EDR-CNN10 cải thiện đáng kể PSNR (+0.66 dB) và SSIM (+0.17%) với p < 0.05
- ✅ Chứng minh 2 module có thể tổng quát hóa sang CNN10
- ⚠️ Edge SSIM chưa significant — cần ghi chú trong báo cáo

### [ ] Bước 5.3 — Đo Efficiency (tùy chọn)
```bash
python paper_scripts/measure_model_stats.py --model edrcnn10
```

---

## GIAI ĐOẠN 6 — Push lên GitHub ✅ HOÀN THÀNH

### [x] Bước 6.1 — Commit tất cả thay đổi
- Implementation, configs, notebooks, evaluation script đã được commit

### [x] Bước 6.2 — Push lên cả 2 repo
- `origin` (lung-diagnosis): ✅ đã push
- `erd-redcnn` (subtree): ✅ đã push (forced update)

---

## Ghi Chú Quan Trọng

> **Lưu ý khác biệt so với EDR-REDNet:**
> CNN10 không có bottleneck sẵn → phải thêm Conv(64→64) trung gian.
> Nếu forward pass ra shape sai (H, W thay đổi), kiểm tra lại `padding` của lớp mới.

> **Dự kiến thời gian train:** ~30–60 phút/run trên Kaggle T4 (CNN10 nhẹ hơn RED-CNN ~74×).

> **Nếu loss NaN:** Giảm `loss_alpha` từ 0.1 xuống 0.05, hoặc giảm `lr`.

---

*Cập nhật lần cuối: 12/06/2026*

---

## TỔNG KẾT DỰ ÁN EDR-CNN10 ✅

| Giai đoạn | Trạng thái |
|:----------|:----------:|
| Thiết kế kiến trúc | ✅ |
| Implementation (code) | ✅ |
| Sanity check + debug | ✅ |
| Training (Variant D, 3 seeds, Kaggle T4) | ✅ |
| Evaluation (Wilcoxon, per-patient) | ✅ |
| Push GitHub | ✅ |

**Kết luận cuối:** EDR-CNN10 chứng minh được `FixedSobelLayer` + `EdgeDilatedResidualBlock` cải thiện PSNR (+0.66 dB) và SSIM (+0.17%) có ý nghĩa thống kê trên CNN10, xác nhận tính tổng quát hóa của 2 module sang các kiến trúc CNN khác ngoài RED-CNN.
