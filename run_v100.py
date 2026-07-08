import os, sys, yaml, re, shutil, subprocess

# 1. Update config
print("=== Cập nhật cấu hình EDR-REDNet cho V100 ===")
config_path = "configs/edrrednet.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["datafolder"] = "data/AAPM-Mayo Clinic"
config["num_workers"] = 8
config["devices"] = [0]
config["seed"] = 42 # Sửa seed tại đây nếu muốn

with open(config_path, "w", encoding="utf-8") as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

# 2. Patch data info
print("=== Lọc dữ liệu bệnh nhân thực tế ===")
info_path = "ldctbench/data/info.yml"
with open(info_path) as f:
    info = yaml.safe_load(f)

ldct_dir = os.path.join(config["datafolder"], "LDCT-and-Projection-data")
if not os.path.exists(ldct_dir):
    print(f"❌ Không tìm thấy {ldct_dir}. Bạn đã tải và giải nén đúng chưa?")
    sys.exit(1)

available = set(os.listdir(ldct_dir))
new_info = {k: v for k, v in info.items() if k not in ["train_set", "val_set", "test_set"]}

for split in ["train_set", "val_set", "test_set"]:
    filtered = []
    for entry in info.get(split, []):
        if entry["id"] not in available:
            continue
        input_full = os.path.join(ldct_dir, entry["input"].replace("./LDCT-and-Projection-data/", ""))
        if not os.path.exists(input_full):
            continue
        actual_files = [f for f in os.listdir(input_full) if f.endswith(".dcm")]
        if not actual_files:
            continue
        entry = dict(entry)
        entry["n_slices"] = len(actual_files)
        filtered.append(entry)
    new_info[split] = filtered

shutil.copy(info_path, info_path + ".bak")
with open(info_path, "w") as f:
    yaml.dump(new_info, f, default_flow_style=False, allow_unicode=True)

# 3. Patch DCM filenames
print("=== Sửa định dạng tên file DCM (LDCTMayo.py) ===")
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

if sample_dcm and not sample_dcm.startswith("0"):
    match = re.match(r'^(.*?)(\d+)\.dcm$', sample_dcm)
    if match:
        prefix, digits = match.group(1), len(match.group(2))
        ldct_mayo_path = "ldctbench/data/LDCTMayo.py"
        with open(ldct_mayo_path, "r") as f: content = f.read()
        old_line = '        return "{}.dcm".format(str(idx).zfill(8))'
        new_line = f'        return "{prefix}{{}}.dcm".format(str(idx).zfill({digits}))'
        if old_line in content:
            content = content.replace(old_line, new_line)
            with open(ldct_mayo_path, "w") as f: f.write(content)
            print(f"✅ Đã vá lỗi tên file ({sample_dcm})")

# 4. Chạy Train
print("\n🚀 Bắt đầu Train...")
subprocess.run(["python", "-m", "ldctbench.scripts.train", "--config", config_path])
