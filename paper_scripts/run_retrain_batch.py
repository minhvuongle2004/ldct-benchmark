"""
run_retrain_batch.py
======================
Script tự động chạy toàn bộ các phiên bản huấn luyện (Batch Training Runner) cho Docker GPU.

Tự động thực hiện:
1. RED-CNN Baseline (10 seeds)
2. HQ-REDCNN (Central Crop) (10 seeds)
3. Ablation Variants (A1, A2, A3) (3 seeds mỗi variant)

Cách sử dụng:
  # Chạy tất cả:
  python paper_scripts/run_retrain_batch.py --mode all

  # Chỉ chạy 10 seed RED-CNN:
  python paper_scripts/run_retrain_batch.py --mode redcnn

  # Chỉ chạy 10 seed HQ-REDCNN:
  python paper_scripts/run_retrain_batch.py --mode hq_redcnn

  # Chỉ chạy Ablation:
  python paper_scripts/run_retrain_batch.py --mode ablation
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

# Auto-detect PyTorch python environment (prefer conda if available)
PYTHON_BIN = sys.executable
if os.path.exists("/opt/conda/bin/python"):
    PYTHON_BIN = "/opt/conda/bin/python"

# 10 seeds chuẩn mực cho nghiên cứu
SEEDS = [42, 1339, 2024, 101, 202, 303, 404, 505, 606, 707]
ABLATION_SEEDS = [42, 1339, 2024]


def run_command(cmd, log_file=None):
    print(f"\n🚀 Executing: {cmd}")
    start_time = time.time()
    
    # Set environment variables to force non-interactive offline logging
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = "offline"
    env["WANDB_SILENT"] = "true"
    
    # Allow process to output directly to terminal standard output
    proc = subprocess.Popen(cmd, shell=True, env=env)
    proc.wait()
        
    elapsed = time.time() - start_time
    if proc.returncode == 0:
        print(f"✅ Finished in {elapsed/60:.2f} mins (Exit code 0)")
    else:
        print(f"❌ Failed with exit code {proc.returncode} after {elapsed/60:.2f} mins")
    return proc.returncode


def train_redcnn():
    print("\n============================================================")
    print("🔥 STARTING RED-CNN BASELINE (10 SEEDS)")
    print("============================================================")
    for seed in SEEDS:
        print(f"\n--- Training RED-CNN Seed {seed} ---")
        cmd = f'{PYTHON_BIN} -m ldctbench.scripts.train --config configs/redcnn.yaml --seed {seed}'
        log = f"logs/training/redcnn_seed_{seed}.log"
        run_command(cmd, log_file=log)


# 7 seeds còn lại cho HQ-REDCNN
HQ_REDCNN_7SEEDS = [101, 202, 303, 404, 505, 606, 707]


def train_hq_redcnn():
    print("\n============================================================")
    print("🔥 STARTING HQ-REDCNN CENTRAL CROP (7 REMAINING SEEDS)")
    print("============================================================")
    for seed in HQ_REDCNN_7SEEDS:
        print(f"\n--- Training HQ-REDCNN Seed {seed} ---")
        cmd = f'{PYTHON_BIN} -m ldctbench.scripts.train --config configs/edrrednet.yaml --seed {seed}'
        log = f"logs/training/hq_redcnn_seed_{seed}.log"
        run_command(cmd, log_file=log)


def train_ablation():
    print("\n============================================================")
    print("🔥 STARTING ABLATION STUDIES (A1, A2, A3)")
    print("============================================================")
    
    # A1: RED-CNN + Charbonnier Loss (No Sobel loss, No Sobel input, No dilated blocks)
    for seed in ABLATION_SEEDS:
        print(f"\n--- Training Ablation A1 (Charbonnier only) Seed {seed} ---")
        cmd = f'{PYTHON_BIN} -m ldctbench.scripts.train --config configs/edrrednet.yaml --seed {seed} --loss_alpha 0.0 --use_sobel_input False --num_edge_blocks 0 --wandbtag ablation_a1'
        log = f"logs/training/ablation_a1_seed_{seed}.log"
        run_command(cmd, log_file=log)

    # A2: A1 + Sobel Loss (Charbonnier + Sobel loss, No Sobel input, No dilated blocks)
    for seed in ABLATION_SEEDS:
        print(f"\n--- Training Ablation A2 (Charb+Sobel loss) Seed {seed} ---")
        cmd = f'{PYTHON_BIN} -m ldctbench.scripts.train --config configs/edrrednet.yaml --seed {seed} --loss_alpha 0.1 --use_sobel_input False --num_edge_blocks 0 --wandbtag ablation_a2'
        log = f"logs/training/ablation_a2_seed_{seed}.log"
        run_command(cmd, log_file=log)

    # A3: A2 + Edge Injection (Sobel input, No dilated blocks)
    for seed in ABLATION_SEEDS:
        print(f"\n--- Training Ablation A3 (Edge injection, no dilated blocks) Seed {seed} ---")
        cmd = f'{PYTHON_BIN} -m ldctbench.scripts.train --config configs/edrrednet.yaml --seed {seed} --loss_alpha 0.1 --use_sobel_input True --num_edge_blocks 0 --wandbtag ablation_a3'
        log = f"logs/training/ablation_a3_seed_{seed}.log"
        run_command(cmd, log_file=log)


def main():
    parser = argparse.ArgumentParser(description="Batch Train Runner for Docker GPU")
    parser.add_argument("--mode", choices=["all", "redcnn", "hq_redcnn", "ablation"], default="all")
    args = parser.parse_args()

    os.makedirs("logs/training", exist_ok=True)
    
    if args.mode in ["all", "redcnn"]:
        train_redcnn()
    if args.mode in ["all", "hq_redcnn"]:
        train_hq_redcnn()
    if args.mode in ["all", "ablation"]:
        train_ablation()

    print("\n🎉 ALL TRAINING BATCHES COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
