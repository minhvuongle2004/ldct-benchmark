import os
import sys
import shutil
import torch

print("=== SO SANH RED-CNN vs EDR-REDNet tren Test Set (9 benh nhan) ===")

# ==========================================
# 1. FIX PYTORCH 2.6+ (weights_only error)
# ==========================================
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load
import ldctbench.evaluate.utils
ldctbench.evaluate.utils.torch.load = safe_load

import ldctbench.utils
import yaml
def safe_load_yaml(path: str):
    with open(path, encoding='utf-8') as file:
        return yaml.load(file, Loader=yaml.FullLoader)
ldctbench.evaluate.utils.load_yaml = safe_load_yaml
ldctbench.utils.load_yaml = safe_load_yaml

# ==========================================
# 2. SETUP CHECKPOINT EDR-REDNet (VariantD seed2024)
# ==========================================
CHECKPOINT_PATH = r"results\training\EDR-RedCnn\VariantD\seed2024\seed2024_best_SSIM.pt"
FAKE_RUN_DIR    = r"wandb\edr_redcnn_seed2024\files"
DATA_FOLDER     = r"D:\cothuy\ldct-benchmark\AAPM-Mayo Clinic\LDCT-and-Projection-data"

if not os.path.exists(CHECKPOINT_PATH):
    print(f"KHONG TIM THAY checkpoint: {CHECKPOINT_PATH}")
    sys.exit(1)

os.makedirs(FAKE_RUN_DIR, exist_ok=True)
shutil.copy(r"configs\edrrednet.yaml", os.path.join(FAKE_RUN_DIR, "args.yaml"))
shutil.copy(CHECKPOINT_PATH,           os.path.join(FAKE_RUN_DIR, "best_SSIM.pt"))
print(f"[OK] Checkpoint EDR-REDNet ready: {CHECKPOINT_PATH}")

# ==========================================
# 3. CHAY TEST: RED-CNN (hub) vs EDR-REDNet (seed2024)
# ==========================================
print("\nDang chay inference tren 9 benh nhan test (CPU, co the mat 15-30 phut)...")
sys.argv = [
    "test_my_model.py",
    "--methods", "redcnn", "edr_redcnn_seed2024",
    "--metrics", "SSIM", "PSNR", "VIF",
    "--datafolder", DATA_FOLDER,
    "--print_table",
]

try:
    from ldctbench.scripts.test import main
    main()
except Exception as e:
    import traceback
    print(f"\nLoi: {e}")
    traceback.print_exc()
