"""
package_hq_redcnn_10seeds.py
=============================
Tự động quét toàn bộ 10 runs HQ-REDCNN trong wandb/, gom tất cả file trọng số best_SSIM.pt,
Losses.csv, Metrics.csv và args.yaml vào thư mục results/training/HQ-REDCNN/Seed{seed}/
sau đó đóng gói thành file ZIP: /root/HQ_REDCNN_10SEEDS_FINAL.zip
"""

import glob
import os
import shutil
import yaml
import zipfile

def main():
    wandb_dir = "/root/ldct-benchmark/wandb"
    target_base = "/root/ldct-benchmark/results/training/HQ-REDCNN"
    
    if os.path.exists(target_base):
        shutil.rmtree(target_base)
    os.makedirs(target_base, exist_ok=True)

    seeds = [101, 202, 303, 404, 505, 606, 707]
    all_runs = sorted(glob.glob(os.path.join(wandb_dir, "offline-run-*")), key=os.path.getmtime)
    
    # Filter only full completed runs (with best_SSIM.pt and Losses.csv >= 30 lines)
    valid_runs = []
    for r in all_runs:
        losses_file = os.path.join(r, "files", "Losses.csv")
        pt_file = os.path.join(r, "files", "best_SSIM.pt")
        if os.path.exists(losses_file) and os.path.exists(pt_file):
            with open(losses_file, encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) >= 30:  # At least 30,000+ iterations logged
                valid_runs.append(r)

    print(f"📦 Tìm thấy {len(valid_runs)}/7 completed HQ-REDCNN runs (40k iterations) trên server.")

    collected = []
    for idx, run in enumerate(valid_runs):
        seed = seeds[idx] if idx < len(seeds) else f"extra_{idx}"
        seed_dir = os.path.join(target_base, f"Seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for fn in ["best_SSIM.pt", "Losses.csv", "Metrics.csv", "args.yaml"]:
            src = os.path.join(run, "files", fn)
            if os.path.exists(src):
                dst_name = f"hq_redcnn_seed{seed}_best_SSIM.pt" if fn == "best_SSIM.pt" else f"hq_redcnn_seed{seed}_{fn}" if fn in ["Losses.csv", "Metrics.csv"] else fn
                shutil.copy2(src, os.path.join(seed_dir, dst_name))
        print(f"  ✅ Đã đóng gói HQ-REDCNN Seed {seed} (từ {os.path.basename(run)}) vào {seed_dir}")
        collected.append(seed)

    # Đóng gói ZIP
    zip_path = "/root/HQ_REDCNN_10SEEDS_FINAL.zip"
    print(f"\n🚀 Đang tạo file nén {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(target_base):
            for file in files:
                abs_p = os.path.join(root, file)
                rel_p = os.path.relpath(abs_p, "/root/ldct-benchmark")
                zipf.write(abs_p, rel_p)

    print(f"\n🎉 HOÀN THÀNH! Đã nén {len(collected)} seeds HQ-REDCNN vào file: {zip_path}")

if __name__ == "__main__":
    main()
