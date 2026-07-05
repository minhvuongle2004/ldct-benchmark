import nbformat as nbf
import io

nb = nbf.v4.new_notebook()

text_1 = """# 🚀 Huấn luyện EDR-REDNet trên Kaggle (Bản sửa lỗi 100%)
Notebook này đã được Fix toàn bộ các lỗi liên quan đến Kaggle (Đường dẫn, GPU, Thư viện, Config).

**Lưu ý trước khi chạy:**
1. Đảm bảo anh đã đẩy code mới nhất lên Github.
2. Bật GPU P100 ở cột bên phải (Session options -> Accelerator).
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

code_5 = """# 5. BẮT ĐẦU TRAIN EDR-REDNet 🚀
%cd /kaggle/working/ldct-benchmark
# Dùng python -m để sửa lỗi ModuleNotFoundError, --dryrun để bỏ qua đăng nhập Wandb
!python -m ldctbench.scripts.train --config configs/edrrednet.yaml --dryrun"""

code_6 = """# 6. ÉP KAGGLE HIỂN THỊ FILE TRỌNG SỐ (.pt)
# Khi chạy "Save & Run All", Kaggle thường tự động ẩn/xóa các file trong thư mục ẩn hoặc bị gitignore.
# Lệnh này sẽ lôi cổ file trọng số ra thư mục gốc để đảm bảo 100% anh tải về được!
!cp /kaggle/working/ldct-benchmark/wandb/*/files/*.pt /kaggle/working/
!ls -lh /kaggle/working/*.pt"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_code_cell(code_6)
]

with io.open('Train_EDR_REDNet_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Đã tạo xong Notebook Kaggle bản hoàn hảo!')

