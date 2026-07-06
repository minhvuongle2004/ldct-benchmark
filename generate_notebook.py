import nbformat as nbf
import io

nb = nbf.v4.new_notebook()

text_1 = """# 🚀 Huấn luyện EDR-REDNet trên Kaggle (Bản sửa lỗi 100%)
Notebook này đã được Fix toàn bộ các lỗi liên quan đến Kaggle (Đường dẫn, GPU, Thư viện, Config).

**Lưu ý trước khi chạy:**
1. Đảm bảo anh đã đẩy code mới nhất lên Github.
2. Bật **GPU T4x2** ở cột bên phải (Session options -> Accelerator). TRÁNH DÙNG P100 vì PyTorch bản mới đã ngừng hỗ trợ con chip cũ này.
3. Đã Add thư mục Data `AAPM-Mayo Clinic` vào Kaggle."""

code_1 = """# 1. Cài đặt các thư viện phụ trợ
!pip install wandb pydicom pyyaml
# Đã xóa lệnh cài PyTorch để dùng bản CUDA gốc của Kaggle."""

code_2 = """import os

# 2. Tải Code từ GitHub về Kaggle
GIT_REPO_URL = "https://github.com/minhvuongle2004/ldct-benchmark.git"
WORKING_DIR = "/kaggle/working/ldct-benchmark"

if not os.path.exists(WORKING_DIR):
    print("Đang clone source code từ GitHub...")
    !git clone {GIT_REPO_URL}
    print("Clone hoàn tất!")
else:
    print("Code đã tồn tại, đang pull cập nhật mới nhất...")
    os.chdir(WORKING_DIR)
    !git pull

os.chdir(WORKING_DIR)
print("Thư mục hiện tại:", os.getcwd())"""

code_3 = """# 3. Tự động quét đường dẫn Data siêu chuẩn
import os
import glob

# Tự động tìm thư mục LDCT-and-Projection-data ở bất cứ đâu trong /kaggle/input/
found_paths = glob.glob('/kaggle/input/**/LDCT-and-Projection-data', recursive=True)

if len(found_paths) == 0:
    print("❌ KHÔNG TÌM THẤY DỮ LIỆU! Anh hãy kiểm tra lại xem đã Add Data vào notebook chưa nhé!")
else:
    # Lấy thư mục cha của LDCT-and-Projection-data
    DATA_DATASET_PATH = found_paths[0].replace('/LDCT-and-Projection-data', '')
    os.environ['LDCTBENCH_DATAFOLDER'] = DATA_DATASET_PATH
    print("✅ Đã tìm thấy và Set đường dẫn Data thành công:")
    print(os.environ['LDCTBENCH_DATAFOLDER'])"""

code_4 = """# 4. Sửa lỗi Config phản chủ
# Xóa bỏ dòng `datafolder: data` trong file yaml để nó chịu nhận đường dẫn Kaggle
!sed -i 's/datafolder: data/datafolder: ""/g' /kaggle/working/ldct-benchmark/configs/edrrednet.yaml
print("Đã Fix lỗi Config!")"""

code_4b = """# 5. CHỈ ĐỊNH SEED MUỐN TRAIN (ĐỂ CHẠY SONG SONG NHIỀU KAGGLE)
# Sửa giá trị này thành 42 hoặc 2024 tùy ý anh nhé!
TRAIN_SEED = 42
print(f"✅ Đã chốt Seed: {TRAIN_SEED}")"""

code_5 = """# 6. BẮT ĐẦU TRAIN EDR-REDNet VỚI SEED ĐÃ CHỌN 🚀
%cd /kaggle/working/ldct-benchmark
# Tự động sửa file yaml theo cái biến TRAIN_SEED ở cell trên
!sed -i "s/seed: .*/seed: {TRAIN_SEED}/g" configs/edrrednet.yaml
# Bắt đầu train
!python -m ldctbench.scripts.train --config configs/edrrednet.yaml --dryrun"""

code_6 = """# 7. LƯU FILE TRỌNG SỐ VÀ KẾT QUẢ (.pt, .csv)
# Dùng latest-run để chỉ lấy kết quả của lần chạy gần nhất, thêm || true để không báo lỗi nếu lỡ dừng sớm
!cp /kaggle/working/ldct-benchmark/wandb/latest-run/files/*_best_SSIM.pt /kaggle/working/seed{TRAIN_SEED}_best_SSIM.pt 2>/dev/null || true
!cp /kaggle/working/ldct-benchmark/wandb/latest-run/files/Metrics.csv /kaggle/working/seed{TRAIN_SEED}_Metrics.csv 2>/dev/null || true
!cp /kaggle/working/ldct-benchmark/wandb/latest-run/files/Losses.csv /kaggle/working/seed{TRAIN_SEED}_Losses.csv 2>/dev/null || true
!ls -lh /kaggle/working/"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_code_cell(code_4b),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_code_cell(code_6)
]

with io.open('Train_EDR_REDNet_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Đã tạo xong Notebook Kaggle bản hoàn hảo!')

