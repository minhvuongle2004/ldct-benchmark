# Dự án: Đánh giá Lâm sàng EDR-REDNet bằng Sybil (Lung Cancer Prediction)

Tài liệu này lưu trữ toàn bộ các luận điểm khoa học, những việc đã hoàn thành và danh sách kiểm việc (Checklist) cho quy trình kết nối mô hình EDR-REDNet với mô hình dự đoán ung thư phổi Sybil.

---

## 1. Mục tiêu Nghiên cứu (Lý do tích hợp Sybil)
- **Thực trạng:** Bài báo chỉ đo lường điểm số ảnh (PSNR, SSIM) thường bị từ chối ở các tạp chí y khoa top đầu vì không chứng minh được "ảnh đẹp hơn thì bác sĩ có chẩn đoán tốt hơn không".
- **Giải pháp:** Sử dụng mô hình Sybil (của MIT, dự đoán rủi ro ung thư phổi 1-6 năm) làm **Downstream Clinical Task**. 
- **Cách chứng minh:** Chứng minh rằng mô hình Sybil chạy trên ảnh đã khử nhiễu (Denoised LDCT) cho ra kết quả dự đoán ung thư chính xác (bám sát ảnh gốc) hơn là chạy trên ảnh nhiễu (LDCT).

---

## 2. Những hạng mục ĐÃ HOÀN THÀNH

### 2.1. Đồng bộ Môi trường (Environment Unification)
- **Vấn đề:** Lỗi xung đột phiên bản (Sybil dùng PyTorch 1.13, pydicom 2.3 | EDR-REDNet dùng PyTorch 2.0+, pydicom 3.0.1).
- **Giải quyết:**
  - Nới lỏng ràng buộc trong `Sybil/setup.cfg` (đổi `==` thành `>=`).
  - Ép Sybil chạy tương thích ngược trên môi trường PyTorch 2.x của dự án hiện tại.
  - Tạo `unified_requirements.txt` và script `verify_env.py` (Đã kiểm tra import đồng thời 2 thư viện thành công).

### 2.2. Dữ liệu thử nghiệm lâm sàng (LIDC-IDRI)
- **Hành động:** Đã viết script `paper_scripts/download_lidc.py` truy cập TCIA Public API không cần tài khoản.
- **Kết quả:** Đã tải thành công 5 bệnh nhân (Patient 0001 - 0005) từ LIDC-IDRI lưu tại `data_sybil/LIDC-IDRI/`. (VD: Patient 0001 có 133 lát cắt DICOM).
- **Ý nghĩa Dữ liệu:** LIDC-IDRI là bộ dữ liệu kinh điển nhất về ung thư phổi, chứa tọa độ chính xác của các khối u (Nodules) do 4 bác sĩ X-quang khoanh vùng. Việc dùng bộ dữ liệu này làm ảnh gốc (Ground Truth) giúp đảm bảo tính minh bạch và độ tin cậy tuyệt đối khi đánh giá với Sybil. Mặc dù bộ dữ liệu xịn nhất là NLST (National Lung Screening Trial - do Sybil được train trên NLST), nhưng LIDC-IDRI hoàn toàn đủ tiêu chuẩn để làm bài test lâm sàng sơ bộ.

---

## 3. Lý luận Khoa học: Tiêm nhiễu Y khoa (Physics-based Noise)
Để giải quyết bài toán LIDC-IDRI chỉ là ảnh chụp liều chuẩn (NDCT), ta phải tiêm nhiễu để tạo ảnh liều thấp (LDCT). Tuyệt đối **KHÔNG** dùng nhiễu ngẫu nhiên vô tội vạ (Gaussian Noise) trên mảng điểm ảnh (Image Domain) vì sai lệch hoàn toàn về mặt vật lý.

**3.1. Bản chất của Nhiễu CT thực tế**
Ảnh CT liều thấp (LDCT) bị nhiễu do hiện tượng **Photon Starvation (Đói Photon)**. Khi dòng điện (mAs) của máy quét CT giảm xuống, số lượng tia X đâm xuyên qua cơ thể cực kỳ ít. Tại các vùng mỡ, xương dày, cảm biến không nhận đủ tín hiệu dẫn đến phân bố nhiễu không đồng đều (Spatially Variant) và tạo các sọc vằn (Streak artifacts).

**3.2. Tiêu chuẩn Y khoa: "Nhiễu bao nhiêu là hợp lý?"**
Chúng ta không thể vặn nhiễu quá tay (chỉ còn 5% tia X - ảnh nát bét không thể phục hồi), hoặc quá ít (còn 80% tia X - ảnh vẫn nét không cần AI). 
**Chuẩn mực lâm sàng (Gold Standard):** Trong Tầm soát ung thư phổi, liều chụp tiêu chuẩn được vặn nhỏ xuống còn **25% (Quarter-dose)** so với thông thường (từ ~100mAs xuống còn ~25mAs). Đây là mức độ hoàn hảo nhất để thử thách EDR-REDNet.

**3.3. Thuật toán mô phỏng nhiễu chuẩn trong Code:**
1. Áp dụng biến đổi Radon (Radon Transform) lên ảnh NDCT để chuyển về miền sóng tia X nguyên thủy (Sinogram Domain).
2. Thiết lập `Dose_Rate = 0.25` (Giả lập máy CT chạy ở mức 25% công suất).
3. Tiêm nhiễu **Poisson** vào Sinogram: $I_{noisy} = Poisson(I_0 \times 0.25)$.
4. Dùng biến đổi FBP (Filtered Back Projection) tái tạo lại ảnh để tạo ra các vệt nhiễu sọc hệt như máy CT ngoài đời thực.

---

## 4. CHECKLIST KẾ HOẠCH THỰC THI CHÍNH (SẮP LÀM)

Dưới đây là Checklist chi tiết để em và anh bám sát tiến độ code:

- `[ ]` **Giai đoạn 1: Code Thuật toán Tiêm Nhiễu (Noise Simulation)**
  - `[ ]` Viết module `simulate_ldct.py` biến đổi Radon -> Tiêm Poisson Noise -> FBP.
  - `[ ]` Chạy thử trên 1 bệnh nhân LIDC-IDRI để sinh ra thư mục DICOM nhiễu (`LDCT`).

- `[ ]` **Giai đoạn 2: Khử nhiễu tự động hàng loạt (Batch Denoising)**
  - `[ ]` Viết hàm đọc liên tục 133 file DICOM LDCT vừa sinh ra.
  - `[ ]` Nạp file trọng số (Checkpoint) tốt nhất của EDR-REDNet.
  - `[ ]` Trích xuất mảng pixel -> Gọt nhiễu -> Ghi đè lại mảng pixel sạch vào file gốc để tạo file DICOM mới (`Denoised`). Đảm bảo bảo toàn tuyệt đối Header (PixelSpacing, SliceThickness).

- `[ ]` **Giai đoạn 3: Viết Pipeline Nối Sybil**
  - `[ ]` Viết `evaluate_sybil.py` khởi tạo class `Sybil()`.
  - `[ ]` Chạy Sybil trên 3 thư mục: `NDCT (Gốc)`, `LDCT (Nhiễu)`, `Denoised (Sạch)`.
  - `[ ]` Trích xuất Risk Score.

- `[ ]` **Giai đoạn 4: Tổng hợp Dữ liệu & Đánh giá**
  - `[ ]` Lưu kết quả của 5 bệnh nhân ra bảng Excel/CSV.
  - `[ ]` Vẽ biểu đồ (Bar chart) so sánh độ lệch (Error) của điểm LDCT và điểm Denoised so với điểm NDCT gốc.
