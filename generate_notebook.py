import nbformat as nbf
import io

nb = nbf.v4.new_notebook()

text_1 = """# 🚀 Huấn luyện EDR-REDNet trên Kaggle
**Lưu ý cực kỳ quan trọng:**
1. Hãy bật **GPU T4x2** ở mục `Session options -> Accelerator`. TUYỆT ĐỐI KHÔNG DÙNG P100 vì PyTorch bản mới trên Kaggle đã ngừng hỗ trợ con chip cũ này, sẽ bị báo lỗi `Invalid device id`.
2. Đảm bảo đã Add thư mục Data vào Kaggle."""

code_1 = """import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_START_METHOD"] = "thread"

!git clone https://github.com/minhvuongle2004/ldct-benchmark.git
%cd /kaggle/working/ldct-benchmark
!pip install -e . -q
!pip install wandb pydicom pyyaml

print("✅ Setup done!")"""

code_2 = """import os

print("=== /kaggle/input/ contents ===")
for item in os.listdir("/kaggle/input"):
    print(f"  {item}/")

data_path = None
print("\\n=== Tìm LDCT-and-Projection-data ===")
for dataset_slug in os.listdir("/kaggle/input"):
    base = f"/kaggle/input/{dataset_slug}"
    for root, dirs, files in os.walk(base):
        if "LDCT-and-Projection-data" in dirs:
            data_path = root
            sub = os.path.join(root, "LDCT-and-Projection-data")
            patients = sorted(os.listdir(sub))
            print(f"✅ Datafolder: {data_path}")
            print(f"   Số bệnh nhân: {len(patients)}")
            print(f"   5 đầu: {patients[:5]}")
            break
    if data_path:
        break

if data_path is None:
    print("❌ Không tìm thấy LDCT-and-Projection-data!")"""

code_3 = """import os, yaml

info_path = "ldctbench/data/info.yml"
with open(info_path) as f:
    info = yaml.safe_load(f)

ldct_dir = os.path.join(data_path, "LDCT-and-Projection-data")
available = set(os.listdir(ldct_dir))

print(f"Bệnh nhân có sẵn: {len(available)}")
print("\\n=== Kiểm tra chi tiết từng bệnh nhân ===")

missing_patients = []
wrong_slices = []

for split in ["train_set", "val_set", "test_set"]:
    for entry in info.get(split, []):
        pid = entry["id"]
        if pid not in available:
            missing_patients.append((split, pid))
            continue

        input_rel = entry["input"].replace("./LDCT-and-Projection-data/", "")
        input_full = os.path.join(ldct_dir, input_rel)

        if not os.path.exists(input_full):
            wrong_slices.append((split, pid, 0, entry["n_slices"], "FOLDER MISSING"))
            continue

        actual_files = [f for f in os.listdir(input_full) if f.endswith(".dcm")]
        actual_n = len(actual_files)
        expected_n = entry["n_slices"]

        if actual_n == 0:
            wrong_slices.append((split, pid, actual_n, expected_n, "EMPTY FOLDER"))
        elif actual_n != expected_n:
            wrong_slices.append((split, pid, actual_n, expected_n, "SLICE MISMATCH"))

print(f"\\n❌ Bệnh nhân thiếu hoàn toàn: {len(missing_patients)}")
for split, pid in missing_patients[:10]:
    print(f"   [{split}] {pid}")

print(f"\\n⚠️ Bệnh nhân có folder nhưng DCM thiếu: {len(wrong_slices)}")
for split, pid, actual, expected, reason in wrong_slices[:20]:
    print(f"   [{split}] {pid}: {actual}/{expected} slices — {reason}")

if not missing_patients and not wrong_slices:
    print("\\n✅ Tất cả dữ liệu đều OK!")"""

code_4 = """import os, yaml, shutil

info_path = "ldctbench/data/info.yml"
with open(info_path) as f:
    info = yaml.safe_load(f)

ldct_dir = os.path.join(data_path, "LDCT-and-Projection-data")
available = set(os.listdir(ldct_dir))

new_info = {k: v for k, v in info.items() if k not in ["train_set", "val_set", "test_set"]}

for split in ["train_set", "val_set", "test_set"]:
    original = info.get(split, [])
    filtered = []
    for entry in original:
        pid = entry["id"]
        if pid not in available:
            continue

        input_rel = entry["input"].replace("./LDCT-and-Projection-data/", "")
        input_full = os.path.join(ldct_dir, input_rel)

        if not os.path.exists(input_full):
            continue

        actual_files = sorted([f for f in os.listdir(input_full) if f.endswith(".dcm")])
        if len(actual_files) == 0:
            continue

        entry = dict(entry)
        entry["n_slices"] = len(actual_files)
        filtered.append(entry)

    new_info[split] = filtered
    print(f"{split}: {len(original)} → {len(filtered)} bệnh nhân")

shutil.copy(info_path, info_path + ".bak")
with open(info_path, "w") as f:
    yaml.dump(new_info, f, default_flow_style=False, allow_unicode=True)

print(f"\\n✅ Đã lưu info.yml mới")"""

code_5 = """import os, re

# Detect format DCM file thực tế (local='00000001.dcm' vs TCIA='1-001.dcm')
ldct_dir = os.path.join(data_path, "LDCT-and-Projection-data")
sample_dcm = None

for patient in sorted(os.listdir(ldct_dir)):
    patient_path = os.path.join(ldct_dir, patient)
    for root, dirs, files in os.walk(patient_path):
        dcm_files = sorted([f for f in files if f.endswith(".dcm")])
        if dcm_files:
            sample_dcm = dcm_files[0]
            break
    if sample_dcm:
        break

print(f"Sample DCM filename: {sample_dcm}")

ldct_mayo_path = "ldctbench/data/LDCTMayo.py"
with open(ldct_mayo_path, "r") as f:
    content = f.read()

if sample_dcm and not sample_dcm.startswith("0"):
    match = re.match(r'^(.*?)(\d+)\.dcm$', sample_dcm)
    if match:
        prefix = match.group(1)          # '1-'
        digits = len(match.group(2))     # 3
        old_line = '        return "{}.dcm".format(str(idx).zfill(8))'
        new_line = f'        return "{prefix}{{}}.dcm".format(str(idx).zfill({digits}))'
        if old_line in content:
            content = content.replace(old_line, new_line)
            old_line2 = '        return "{}.dcm".format(str(idx).zfill(8))'
            content = content.replace(old_line2, new_line)
            with open(ldct_mayo_path, "w") as f:
                f.write(content)
            print(f"✅ Patched LDCTMayo.py → format '{prefix}{{:0{digits}d}}.dcm'")
        else:
            print("⚠️ Format cũ không tìm thấy để patch.")
else:
    print(f"✅ Format đã đúng (00000XXX.dcm), không cần patch")"""

code_6 = """import glob
import os
import yaml
os.environ["WANDB_MODE"] = "offline"

assert data_path is not None, "❌ Chạy Cell 2 trước!"
print(f"✅ Datafolder: {data_path}")

# ==========================================
# CHỈ ĐỊNH SEED MUỐN TRAIN
# Sửa giá trị này thành 42 hoặc 2024
# ==========================================
TRAIN_SEED = 42

with open("configs/edrrednet.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Cập nhật các thông số động cho Kaggle
config["seed"] = TRAIN_SEED
config["datafolder"] = data_path
config["num_workers"] = 2  # Trả về 2 để tránh bị lỗi tràn RAM/treo Dataloader
config["devices"] = [0]    # Chỉ dùng 1 Card T4 để tránh lỗi DataParallel bị treo sau vài Epoch

checkpoints = glob.glob(f"/kaggle/input/**/seed{TRAIN_SEED}_best_*.pt", recursive=True)
if checkpoints:
    ckpt_path = checkpoints[0]
    config["resume"] = ckpt_path
    
    # Tự động tăng max_iterations lên để chạy tiếp
    config["max_iterations"] = int(config.get("max_iterations", 40000) * 2.325)
    print(f"\\n✅ TÌM THẤY CHECKPOINT CŨ: {ckpt_path}")
    print(f"   -> Sẽ Resume và train tiếp đến {config['max_iterations']} iterations.")
else:
    print("\\nℹ️ KHÔNG TÌM THẤY CHECKPOINT CŨ.")
    print(f"   -> Sẽ Train từ đầu đến {config.get('max_iterations', 40000)} iterations.")

with open("configs/edrrednet.yaml", "w", encoding="utf-8") as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"\\nConfig saved. Bắt đầu training seed {TRAIN_SEED}...")
!python -m ldctbench.scripts.train --config configs/edrrednet.yaml --dryrun"""

code_7 = """import glob, shutil, os

output_dir = "/kaggle/working"
seed = TRAIN_SEED

# Checkpoint lưu tại wandb.run.dir = wandb/offline-run-XXXXX/files/
checkpoints = glob.glob("wandb/offline-run-*/files/*_best_*.pt")
print(f"Checkpoints found: {checkpoints}")
for ckpt in checkpoints:
    dest = os.path.join(output_dir, f"seed{seed}_{os.path.basename(ckpt)}")
    shutil.copy(ckpt, dest)
    print(f"✅ Saved: {dest}")

# Loss/metrics CSV
csv_files = glob.glob("wandb/offline-run-*/files/*.csv")
for csv in csv_files:
    dest = os.path.join(output_dir, f"seed{seed}_{os.path.basename(csv)}")
    shutil.copy(csv, dest)
    print(f"✅ Log: {dest}")

print("\\n✅ Done! Click 'Save Version' để lưu output.")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_code_cell(code_7)
]

with io.open('Train_EDR_REDNet_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Đã tạo xong Notebook Kaggle bản hoàn hảo!')
