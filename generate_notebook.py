import nbformat as nbf
import io

nb = nbf.v4.new_notebook()

text_1 = """# 🚀 Huấn luyện EDR-REDNet trên Kaggle (Bộ dữ liệu 100 Bệnh nhân)
Notebook này được thiết kế tự động để anh chạy thẳng trên Kaggle qua đường Git Clone. 

**Lưu ý trước khi chạy:**
1. Đảm bảo anh đã đẩy code mới nhất lên Github của anh. (Sửa lại link Git clone ở Cell bên dưới cho đúng).
2. Anh nén thư mục `AAPM-Mayo Clinic` thành file `data.zip` và upload lên Kaggle Dataset (ví dụ đặt tên là `ldct-100-patients`).
3. Add cái dataset Data đó vào Notebook này."""

code_1 = """# 1. Cài đặt các thư viện cần thiết
!pip install wandb pydicom pyyaml
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"""

code_2 = """import os

# 2. Tải Code từ GitHub về Kaggle
# CHÚ Ý: Đổi link Github dưới đây thành link Repo của anh
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

code_3 = """# 3. Cấu hình đường dẫn Data
# CHÚ Ý: Thay đổi tên thư mục 'ldct-100-patients' thành tên Dataset data của anh
DATA_DATASET_PATH = '/kaggle/input/ldct-100-patients'

# Truyền đường dẫn data vào biến môi trường để code tự nhận
os.environ['LDCTBENCH_DATAFOLDER'] = DATA_DATASET_PATH
print(f"Đã set đường dẫn Data: {os.environ['LDCTBENCH_DATAFOLDER']}")"""

code_4 = """# 4. Đăng nhập Wandb (để theo dõi biểu đồ Loss/SSIM)
import wandb
wandb.login() # Kaggle sẽ hiện ô nhập API Key của anh"""

code_5 = """# 5. BẮT ĐẦU TRAIN EDR-REDNet 🚀
# Lưu ý: Checkpoint sẽ được lưu trong folder 'results/training/'
!python ldctbench/scripts/train.py --config configs/edrrednet.yaml"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_code_cell(code_5)
]

with io.open('Train_EDR_REDNet_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Updated Train_EDR_REDNet_Kaggle.ipynb with Git Clone support')
