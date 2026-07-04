# Kế Hoạch Nâng Cấp EDR-REDNet (Tiêu chuẩn Tạp chí Quốc tế)

**Tên đề tài dự kiến:** *EDR-REDNet improves edge-aware robustness and small-structure preservation while maintaining comparable global image quality to RED-CNN.*

---

## 1. MỤC TIÊU CHIẾN LƯỢC
Nâng tầm bài báo từ mức báo cáo kỹ thuật lên nghiên cứu khoa học chuyên sâu, tập trung chứng minh 2 giá trị cốt lõi:
1.  **Sự bền vững (Robustness):** Khả năng hoạt động ổn định ở liều tia cực thấp (Stress test).
2.  **Sự sắc nét biên (Edge-awareness):** Bảo tồn chi tiết nhỏ mà không làm hỏng chất lượng tổng thể.

---

## 2. LỘ TRÌNH THỰC HIỆN (ROADMAP)

### Giai đoạn 1: Hoàn tất Ablation Study (Tuần 1-2)
Mục tiêu: Chứng minh giá trị của từng thành phần trong mô hình.
*   **Variant A:** RED-CNN (Baseline).
*   **Variant B:** RED-CNN + EdgeBlock (Dilated residual).
*   **Variant C:** Variant B + FixedSobelLayer (Input features).
*   **Variant D:** **EDR-REDNet Full** (Mô hình chính).
*   **Variant E:** Variant D + Perceptual Loss (VGG).
*   *Hành động:* Train 3 seeds cho mỗi variant để lấy Mean ± Std.

### Giai đoạn 2: Tích hợp Chỉ số Nâng cao (Tuần 3)
Mục tiêu: Cung cấp bằng chứng định lượng không thể bác bỏ về "Biên".
*   **Edge Metrics:** Edge SSIM, Sobel cosine similarity, Gradient error.
*   **Clinical Metrics:** 
    *   **CNR (Contrast-to-Noise Ratio):** Tính trên vùng ROI (mạch máu, nốt phổi).
    *   **HU deviation:** Đo độ lệch Hounsfield trên các vùng mô đồng nhất.
*   *Hành động:* Cập nhật Web App hỗ trợ chọn vùng ROI để tính toán.

### Giai đoạn 3: Kiểm định Thống kê & Hiệu năng (Tuần 4)
Mục tiêu: Đảm bảo tính khoa học và thực tiễn.
*   **Statistical Test:** Chạy kiểm định **Wilcoxon signed-rank test** trên toàn bộ 100 bệnh nhân để báo cáo p-value.
*   **Efficiency:** Đo lường số lượng tham số (Params) và thời gian xử lý (Runtime) để chứng minh mô hình không quá nặng.
*   *Hành động:* Hoàn thiện script `measure_model_stats.py` và viết `statistical_test.py`.

### Giai đoạn 4: Trực quan hóa & Viết bài (Tuần 5-6)
Mục tiêu: Hình ảnh minh họa "vạn người mê".
*   **Visual comparison:** Zoom vùng mạch máu, biên phổi.
*   **Difference map & Sobel map:** Chứng minh sự khác biệt rõ rệt bằng hình ảnh.
*   **Boxplots:** Vẽ biểu đồ phân bố SSIM/PSNR cho 3 seeds.

---

## 3. TIMELINE CHI TIẾT

| Tuần | Công việc trọng tâm | Kết quả cần đạt (Deliverables) |
| :--- | :--- | :--- |
| **Tuần 1** | Train Ablation Variants B & C | Checkpoints của B, C. |
| **Tuần 2** | Train Ablation Variant E & Tổng hợp số liệu | Bảng kết quả Ablation (Mean ± Std). |
| **Tuần 3** | Code chỉ số CNR & HU deviation trên App | Web App hiển thị CNR vùng ROI. |
| **Tuần 4** | Chạy kiểm định thống kê & Đo Runtime | Bảng p-value và bảng Params/Runtime. |
| **Tuần 5** | Xuất hình ảnh Zoom, Difference Maps | Bộ Figures chất lượng cao cho bài báo. |
| **Tuần 6** | Hoàn thiện bản thảo (Manuscript) | File báo cáo tổng hợp cuối cùng. |

---
## 4. DANH SÁCH CÁC CHỈ SỐ CẦN BÁO CÁO (FINAL TABLE)
| Mô hình | PSNR ↑ | SSIM ↑ | VIF ↑ | **Edge SSIM ↑** | **CNR ↑** | **Params ↓** | **Runtime ↓** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RED-CNN | ... | ... | ... | ... | ... | ... | ... |
| **EDR-REDNet**| ... | ... | ... | **Cao nhất** | **Cao nhất** | < 1.1x | < 1.2x |

---
*Lập bởi: Antigravity AI Assistant*  
*Ngày lập: 12/05/2026*
