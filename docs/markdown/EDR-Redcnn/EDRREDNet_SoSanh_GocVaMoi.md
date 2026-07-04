# So Sánh: `ldct-benchmark` Gốc vs. Phiên Bản Phát Triển EDR-REDNet

> **Repo gốc:** [eeulig/ldct-benchmark](https://github.com/eeulig/ldct-benchmark) — Eulig et al., *Medical Physics* 2024  
> **Phiên bản này:** `d:\cothuy\lung-diagnosis\ldct-benchmark` — Phát triển bởi Vương, dựa trên kế hoạch [EDRREDNet_KeHoachThucHien.md](EDRREDNet_KeHoachThucHien.md)  
> **Mục tiêu:** Đề xuất mô hình **EDR-REDNet** (Edge-Dilated Residual RED-CNN) cho bài báo khoa học

---

## 1. Tổng Quan So Sánh Nhanh

| Khía cạnh | Repo gốc (eeulig) | Phiên bản EDR-REDNet |
|:---|:---|:---|
| **Mục đích** | Benchmark công bằng 8 thuật toán LDCT | Nghiên cứu & đề xuất model mới (EDR-REDNet) |
| **Số model** | 8 baseline (Bilateral, CNN10, DU-GAN, QAE, RED-CNN, ResNet, TransCT, WGAN-VGG) | 8 baseline + **1 model mới (EDR-REDNet)** |
| **Dataset** | Mayo 2016 + LDCT-and-Projection-data | Như gốc + **tập trung Chest/Abdomen (100 BN)** |
| **Loss function** | Tùy từng model (MSE, adversarial, perceptual...) | Thêm **Charbonnier + SobelEdgeLoss** |
| **Metrics** | PSNR, SSIM, RMSE, VIF | Thêm **Edge SSIM, CNR, HU deviation** |
| **Web App** | Không có | **Có** — Streamlit app so sánh + ROI analysis |
| **Scripts phân tích** | Cơ bản | Thêm nhiều scripts thống kê và visualization |
| **Paper-ready outputs** | Không có | **Có** — Figure 1–7, Table 1–5 dạng PNG journal-style |

---

## 2. Cấu Trúc Thư Mục: Gốc vs. Phiên Bản Mới

### 2.1 `ldctbench/methods/` — Module Model

| Thư mục | Repo gốc | Phiên bản mới | Ghi chú |
|:---|:---:|:---:|:---|
| `bilateral/` | ✅ | ✅ | Giữ nguyên |
| `cnn10/` | ✅ | ✅ | Giữ nguyên |
| `dugan/` | ✅ | ✅ | Giữ nguyên |
| `qae/` | ✅ | ✅ | Giữ nguyên |
| `redcnn/` | ✅ | ✅ | Giữ nguyên (dùng làm Variant A) |
| `resnet/` | ✅ | ✅ | Giữ nguyên |
| `transct/` | ✅ | ✅ | Giữ nguyên |
| `wganvgg/` | ✅ | ✅ | Giữ nguyên |
| **`edrrednet/`** | ❌ | ✅ **[MỚI]** | **Module chính — model đề xuất** |

#### Chi tiết `ldctbench/methods/edrrednet/` [HOÀN TOÀN MỚI]

```
edrrednet/
├── __init__.py        — Export module
├── network.py         — Kiến trúc EDR-REDNet (FixedSobelLayer + EdgeDilatedResidualBlock)
├── loss.py            — SobelEdgeLoss + CombinedLoss (Charbonnier + α×Sobel)
├── Trainer.py         — Training loop tích hợp combined loss
└── argparser.py       — Tham số riêng: alpha, dilation rates
```

**Thành phần kiến trúc mới trong `network.py`:**
- `FixedSobelLayer` — Non-trainable layer trích xuất edge map 4 hướng (H, V, Diag45°, Diag135°)
- `EdgeDilatedResidualBlock` — Dilated conv (rate=2, rate=3) + residual connection tại bottleneck
- `EDRREDNet` (Model class) — RED-CNN backbone + EdgeBlock + global residual skip + additive edge fusion

**Loss mới trong `loss.py`:**
- `SobelEdgeLoss` — L1 loss trên Sobel gradient của pred vs target
- `CombinedLoss` — `L_total = L_Charbonnier + α × L_Sobel` (α = 0.1 mặc định)

---

### 2.2 `configs/` — File Cấu Hình Training

| File | Repo gốc | Phiên bản mới |
|:---|:---:|:---:|
| `bilateral.yaml` | ✅ | ✅ |
| `cnn10.yaml` / `cnn10-hpopt.yaml` | ✅ | ✅ |
| `dugan.yaml` | ✅ | ✅ |
| `qae.yaml` | ✅ | ✅ |
| `redcnn.yaml` | ✅ | ✅ |
| `resnet.yaml` | ✅ | ✅ |
| `transct.yaml` | ✅ | ✅ |
| `wganvgg.yaml` | ✅ | ✅ |
| **`edrrednet.yaml`** | ❌ | ✅ **[MỚI]** |

`edrrednet.yaml` cấu hình: `loss_alpha=0.1`, `lr=9.583e-05`, `mbs=16`, `max_iterations=92994`, `data_norm=meanstd`

---

### 2.3 Scripts Phân Tích [HOÀN TOÀN MỚI]

Repo gốc **không có** các scripts sau — tất cả đều được thêm mới:

| Script | Mục đích |
|:---|:---|
| `evaluate_statistical_test.py` | Tính PSNR/SSIM/Edge-SSIM trên 9 bệnh nhân test, Wilcoxon signed-rank test, xuất per-patient CSV |
| `measure_model_stats.py` | Đo MACs (GFLOPs), số parameters, inference time cho từng variant |
| `generate_paper_figures.py` | Tạo Figure 1 (kiến trúc) và Figure 2 (ablation design) dạng PNG journal-style |
| `generate_paper_tables.py` | Tạo Table 1–5 dạng PNG journal-style (booktabs style) |
| `generate_table2.py` | Tự động tổng hợp validation metrics từ 3 seeds → Table 2 CSV |
| `split_paper_figures.py` | Tách Figure 3/4/5 từ combined PNG thành 3 file riêng |
| `generate_stress_test.py` | Tạo data stress test với các mức noise khác nhau |
| `find_matching_slices.py` | Tìm lát cắt tương ứng giữa LDCT và NDCT |
| `train_tpu_standalone.py` | Script train độc lập cho TPU (Kaggle/Colab) |
| `test_my_model.py` | Quick test forward pass model |
| `app.py` | **Web app Streamlit** — toàn bộ dashboard phân tích |

---

### 2.4 `app.py` — Web App Streamlit [HOÀN TOÀN MỚI]

Repo gốc **không có** web app. Phiên bản này có Streamlit app với 3 tab:

| Tab | Chức năng |
|:---|:---|
| 🔍 **So sánh Mô hình** | Chạy inference thời gian thực, hiển thị PSNR/SSIM/Edge-SSIM cho từng variant |
| 📊 **Ablation Study** | Bảng so sánh tất cả variant, bar charts, boxplots |
| 📄 **Paper Figures** | Boxplots PSNR/SSIM/Edge-SSIM, Zoom comparison, Difference Map, Sobel Map, ROI analysis (CNR, HU deviation), Export PNG 300 DPI |

---

### 2.5 `results/` — Kết Quả [HOÀN TOÀN MỚI]

Repo gốc không có thư mục `results/` với nội dung training. Phiên bản này có:

```
results/
├── fig1_architecture.png           ← Figure 1: Kiến trúc EDR-REDNet (white bg, journal-style)
├── fig2_ablation_design.png        ← Figure 2: Ablation variants A–D (white bg)
├── fig3_qualitative_comparison.png ← Figure 3: So sánh ảnh xám LDCT/A/B/C/D/NDCT
├── fig4_difference_maps.png        ← Figure 4: Difference maps vs NDCT
├── fig5_sobel_edge_maps.png        ← Figure 5: Sobel edge map comparison
├── fig_paper_L150_sl77.png         ← Figure 3+4+5 dạng combined (patient L150, slice 77)
│
├── tables/
│   ├── table1_ablation_variants.png   ← Table 1: Component design
│   ├── table2_validation_seeds.png    ← Table 2: Val SSIM/PSNR theo seed
│   ├── table3_patient_ablation.png    ← Table 3: Per-patient kết quả
│   ├── table4_wilcoxon.png            ← Table 4: Wilcoxon p-values
│   └── table5_efficiency.png          ← Table 5: Params/MACs/Inference time
│
├── evaluation/
│   ├── per_patient_scores.csv         ← PSNR/SSIM/Edge-SSIM của 9 BN × 4 variants
│   ├── wilcoxon_pvalues.csv           ← p-values D vs A/B/C
│   ├── model_efficiency.csv           ← Params, MACs, inference time
│   └── table2_validation_seeds.csv    ← Validation metrics 3 seeds
│
├── training/
│   ├── VariantA/ (RED-CNN baseline)
│   ├── VariantB/ (+ EdgeBlock), VariantC/ (+ Sobel), VariantD/ (Full EDR-REDNet)
│   └── [mỗi variant có Seed1339, Seed2024, Seed42]
│
├── ablation/                          ← Kết quả ablation study
└── test_metrics.yaml                  ← Metrics tổng hợp
```

---

### 2.6 `docs/` — Tài Liệu

| File | Repo gốc | Phiên bản mới |
|:---|:---:|:---:|
| `docs/` (MkDocs documentation) | ✅ | ✅ Giữ nguyên |
| `docs/markdown/EDRREDNet_KeHoachThucHien.md` | ❌ | ✅ **[MỚI]** |
| `docs/markdown/EDRREDNet_KeHoachNangCap_TapChi.md` | ❌ | ✅ **[MỚI]** |
| `docs/markdown/EDRREDNet_TienDoNghienCuu.md` | ❌ | ✅ **[MỚI]** |
| `docs/markdown/EDRREDNet_SoSanh_GocVaMoi.md` | ❌ | ✅ **[File này]** |

---

### 2.7 `src/` — Scripts Tiện Ích [MỚI]

| File | Mô tả |
|:---|:---|
| `src/denoise_single_image.py` | Denoise một ảnh đơn lẻ, hỗ trợ tất cả variant |
| `src/denoise_folder_gif.py` | Denoise cả thư mục, tạo GIF so sánh động |

---

## 3. Thống Kê Thay Đổi

| Loại | Số lượng |
|:---|:---:|
| Files **giữ nguyên** từ repo gốc | ~40 files |
| Files **thêm mới hoàn toàn** | ~20 files |
| Thư mục **thêm mới** | `edrrednet/`, `results/`, `src/`, `docs/markdown/` |
| Model method mới | **1** (EDR-REDNet) |
| Config mới | **1** (edrrednet.yaml) |
| Scripts phân tích mới | **10+** |
| Figures journal-ready | **5** (Fig 1–5) |
| Tables journal-ready | **5** (Table 1–5) |

---

## 4. Kiến Trúc EDR-REDNet vs RED-CNN (Baseline)

```
RED-CNN (Baseline — Variant A):
  Input → Conv1→Conv2→Conv3→Conv4→Conv5 (Encoder)
        → TConv1→TConv2→TConv3→TConv4→TConv5 (Decoder)
        → Output (+ global residual skip)
  Loss: MSELoss

EDR-REDNet (Ours — Variant D):
  Input ──┬──────────────────────────────────────────┐
          ↓                                          │
   FixedSobelLayer (4 edge maps, non-trainable)      │ [MỚI]
          ↓ (additive fusion tại bottleneck)         │
   Conv1→Conv2→Conv3→Conv4→Conv5 (Encoder RED-CNN)  │
          ↓                                          │
   EdgeDilatedResidualBlock (d=2) [MỚI]              │
          ↓                                          │
   EdgeDilatedResidualBlock (d=3) [MỚI]              │
          ↓                                          │
   TConv1→TConv2→TConv3→TConv4→TConv5 (Decoder)    │
          ↓                                          │
   Output ←──────────────────────────────────────────┘
  Loss: L_Charbonnier + 0.1 × L_SobelEdge  [MỚI]
```

**Tham số:**
| | RED-CNN | EDR-REDNet |
|:---|:---:|:---:|
| Trainable params | 1.85 M | 2.18 M (+18%) |
| Non-trainable params | 0 | 384 (FixedSobelLayer) |
| MACs | 458.4 G | 538.8 G (+18%) |

---

## 5. Kết Quả Đạt Được (So Với Kế Hoạch)

### ✅ Đã Hoàn Thành

- [x] Xây dựng module `edrrednet` hoàn chỉnh (network, loss, trainer, argparser, config)
- [x] Train 4 ablation variants (A, B, C, D) × 3 seeds (1339, 2024, 42)
- [x] Đánh giá trên 9 bệnh nhân test — PSNR, SSIM, Edge-SSIM
- [x] Wilcoxon signed-rank test (D vs A/B/C) → p < 0.05 cho Edge-SSIM
- [x] Đo computational efficiency (MACs, Params, Inference time)
- [x] Figure 1: Kiến trúc EDR-REDNet (journal-style, white bg)
- [x] Figure 2: Ablation design diagram (journal-style)
- [x] Figure 3: Qualitative comparison (LDCT/A/B/C/D/NDCT)
- [x] Figure 4: Difference maps vs NDCT
- [x] Figure 5: Sobel edge map comparison
- [x] Table 1–5 dạng PNG journal-style (booktabs)
- [x] Web App Streamlit với ROI analysis (CNR, HU deviation)
- [x] Boxplots PSNR/SSIM/Edge-SSIM (qua app, tab Paper Figures)

### 🔄 Đang Tiến Hành / Còn Lại

- [ ] **Figure 6:** Boxplots dạng PNG 300 DPI (tải qua app → nút "Tải Boxplots")
- [ ] **Figure 7:** Screenshot ROI demo từ web app
- [ ] **Table 6 (Optional):** ROI metrics CNR và HU deviation
- [ ] **Table 7 (Optional):** Anatomy-wise Edge SSIM (Chest vs Abdomen)
- [ ] **Soạn thảo bản thảo (Manuscript):** Abstract, Introduction, Methodology, Results, Discussion

---

## 6. Kết Quả Chính (Variant D vs Baseline)

| Metric | Variant A (Baseline) | Variant D (Ours) | Cải thiện |
|:---|:---:|:---:|:---:|
| Mean SSIM | ~0.887 | 0.8890 ± 0.0144 | Tương đương |
| Mean PSNR | ~40.2 | 40.41 ± 1.31 | Tương đương |
| Edge SSIM | — | **Cao hơn đáng kể** | p = 0.0020 ✅ |
| Wilcoxon p (Edge SSIM) | — | **0.0020** | p < 0.05 ✅ |

> **Kết luận:** EDR-REDNet không làm giảm PSNR/SSIM so với baseline (p ≥ 0.05) nhưng cải thiện đáng kể khả năng bảo tồn biên (Edge SSIM, p = 0.0020 < 0.05). Đây là đóng góp chính của bài báo.

---

## 7. Tài Liệu Tham Khảo

| | |
|:---|:---|
| **Repo gốc** | [eeulig/ldct-benchmark](https://github.com/eeulig/ldct-benchmark) |
| **Bài báo framework** | Eulig et al., *Medical Physics* 2024, DOI: [10.1002/mp.17379](https://doi.org/10.1002/mp.17379) |
| **Bài báo cơ sở** | Gholizadeh-Ansari et al., *J. Digital Imaging* 2020, DOI: [10.1007/s10278-019-00274-4](https://doi.org/10.1007/s10278-019-00274-4) |
| **RED-CNN** | Chen et al., *IEEE TMI* 2017, DOI: [10.1109/TMI.2017.2715284](https://doi.org/10.1109/TMI.2017.2715284) |
| **Dilated Conv** | Yu & Koltun, arXiv 2015, [1511.07122](https://arxiv.org/abs/1511.07122) |

---

*Tổng hợp bởi: Antigravity AI Assistant*  
*Ngày tạo: 25/05/2026*
