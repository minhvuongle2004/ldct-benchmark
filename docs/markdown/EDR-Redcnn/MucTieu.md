# TỔNG HỢP MỤC TIÊU VÀ KẾT QUẢ ĐẠT ĐƯỢC
*(Dựa trên tài liệu YeuCauHeThong.docx)*

Tài liệu này theo dõi sát sao tất cả các hạng mục cần có để hoàn thiện bài báo khoa học. Các mục đã chạy thành công sẽ được điền sẵn kết quả thực nghiệm.

---

## MỤC A — Tái Tạo Kết Quả Benchmark Gốc
> **Mục tiêu:** Chứng minh benchmark gốc hoạt động đúng và kết quả của các model image-domain có thể tái tạo được (đóng vai trò Baseline).

### ✅ A.1 - Bảng 1 (SSIM / PSNR / VIF)
- **Trạng thái:** HOÀN THÀNH.
- **Dữ liệu test:** Đã chạy thành công trên tập giải phẫu vùng bụng (Abdomen) của 4 bệnh nhân, gồm hơn 1.200 lát cắt.
- **Kết quả:**

| Mô hình | SSIM | PSNR (dB) | VIF |
| :--- | :--- | :--- | :--- |
| **cnn10** | 0.500 | 12.920 | 0.069 |
| **redcnn** | **0.502** | 12.929 | **0.075** |
| **wganvgg** | 0.488 | 12.891 | 0.063 |
| **resnet** | **0.502** | **12.935** | 0.074 |
| **qae** | 0.493 | 12.914 | 0.064 |
| **dugan** | 0.491 | 12.894 | 0.065 |
| **transct** | 0.498 | 12.918 | 0.067 |

*(Ghi chú: Bilateral Filter bị loại bỏ do thiếu thư viện C++ tương thích trên Windows).*

### ✅ A.2 - Bảng 2 (Confidence Interval)
- **Trạng thái:** HOÀN THÀNH (Đã chạy trên 1 seed cố định là `seed=1332` đạt yêu cầu tối thiểu).

### ❌ A.3 - Bảng 3 (So sánh với paper gốc)
- **Trạng thái:** CHƯA HOÀN THÀNH.
- **Công việc tiếp theo:** Anh cần copy các chỉ số SSIM/PSNR ở trên, dán vào file Word và đối chiếu với bảng kết quả trong file PDF paper gốc (arXiv: 2401.04661) để tính chênh lệch.

---

## MỤC B — Efficiency Benchmark & Perceptual Metric
> **Mục tiêu:** Chứng minh 2 lỗ hổng của bài báo gốc: Thiếu đánh giá hiệu năng (efficiency) và thiếu perceptual metric (LDCTIQA).

### ✅ B.1 - Bảng Efficiency (Thời gian & Bộ nhớ)
- **Trạng thái:** HOÀN THÀNH. 
- **Thiết bị chạy:** NVIDIA GeForce RTX 3050 Laptop GPU.
- **Kết quả:**

| Mô hình | Số tham số (Triệu) | Thời gian chạy (ms/ảnh) | Tốn Peak VRAM (MB) |
| :--- | :--- | :--- | :--- |
| **cnn10** | 0.025 | **10.30** ms | 129.10 MB |
| **wganvgg** | 0.056 | 13.07 ms | **65.32 MB** |
| **transct** | 79.348 | 15.52 ms | 608.00 MB |
| **qae** | 0.050 | 32.67 ms | 205.09 MB |
| **redcnn** | 1.849 | 198.13 ms | 661.11 MB |
| **dugan** | 1.849 | 201.40 ms | 661.11 MB |
| **resnet** | 1.843 | 334.95 ms | 648.13 MB |

### ❌ B.2 - Bảng LDCTIQA (Score no-reference perceptual quality)
- **Trạng thái:** CHƯA HOÀN THÀNH. Lỗi kỹ thuật ở mã nguồn gốc `test.py` do tham số chưa được gán giá trị đúng cách. Cần sửa code và chạy lại riêng cho metric này.

### ❌ B.3 - Biểu đồ Quality vs Speed (Scatter plot)
- **Trạng thái:** CHƯA HOÀN THÀNH. Đã có đủ số liệu từ phần B.1, chỉ chờ sử dụng Python `matplotlib` vẽ trục X là Runtime, Y là SSIM.

### ❌ B.4 - Checkpoint model đã train lại từ đầu
- **Trạng thái:** CHƯA HOÀN THÀNH. Yêu cầu phải train lại 1 mạng từ đầu (như cnn10) để sinh ra `training log` (loss curve) chứng minh mình tự code/train chứ không copy pretrained.

---

## MỤC C — Dual-Domain Model (Đóng Góp Chính Của Bài Báo)
> **Mục tiêu:** Xây dựng mô hình mới tận dụng cả Image Domain và Projection Domain (Sinogram). Đây là giá trị khác biệt (Novelty).
- **Trạng thái chung:** ❌ HOÀN TOÀN CHƯA BẮT ĐẦU. (Theo thảo luận ban đầu, mục này đang được gác lại để ưu tiên A, B, D).

---

## MỤC D — Web Demo
> **Mục tiêu:** Tạo web app trực quan cho phép người dùng đăng tải ảnh DICOM và xem mô hình hoạt động.

### ❌ D.1 - Web App chạy được bằng Gradio/Streamlit
- **Trạng thái:** CHƯA HOÀN THÀNH. Các file checkpoint pretrained của 7 mô hình đã có sẵn trên máy, cần bắt tay vào viết file `app.py`.

### ❌ D.2 - URL Deploy Public (Hugging Face Spaces)
- **Trạng thái:** CHƯA HOÀN THÀNH.
