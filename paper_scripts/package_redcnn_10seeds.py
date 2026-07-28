"""
package_redcnn_10seeds.py
==========================
Tự động quét toàn bộ 10 runs trong wandb/, gom tất cả file trọng số best_SSIM.pt,
Losses.csv, Metrics.csv và args.yaml vào thư mục results/training/RED-CNN/Seed{seed}/
sau đó đóng gói thành file ZIP gọn nhẹ: /root/REDCNN_10SEEDS_FINAL.zip
"""

import glob
import os
import shutil
import yaml
import zipfile

def main():
    wandb_dir = "/root/ldct-benchmark/wandb"
    target_base = "/root/ldct-benchmark/results/training/RED-CNN"
    
    if os.path.exists(target_base):
        shutil.rmtree(target_base)
    os.makedirs(target_base, exist_ok=True)

    seeds = [42, 1339, 2024, 101, 202, 303, 404, 505, 606, 707]
    runs = sorted(glob.glob(os.path.join(wandb_dir, "offline-run-*")), key=os.path.getmtime)
    print(f"📦 Tìm thấy {len(runs)} offline wandb runs trên server.")

    # Drop early debug runs if more than 10
    if len(runs) > 10:
        runs = runs[-10:]

    collected = []
    for idx, run in enumerate(runs):
        seed = seeds[idx] if idx < len(seeds) else f"extra_{idx}"
        seed_dir = os.path.join(target_base, f"Seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for fn in ["best_SSIM.pt", "Losses.csv", "Metrics.csv", "args.yaml"]:
            src = os.path.join(run, "files", fn)
            if os.path.exists(src):
                dst_name = f"redcnn_seed{seed}_best_SSIM.pt" if fn == "best_SSIM.pt" else f"redcnn_seed{seed}_{fn}" if fn in ["Losses.csv", "Metrics.csv"] else fn
                shutil.copy2(src, os.path.join(seed_dir, dst_name))
        print(f"  ✅ Đã đóng gói Seed {seed} (từ {os.path.basename(run)}) vào {seed_dir}")
        collected.append(seed)

    # Đóng gói ZIP gọn nhẹ
    zip_path = "/root/REDCNN_10SEEDS_FINAL.zip"
    print(f"\n🚀 Đang tạo file nén {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(target_base):
            for file in files:
                abs_p = os.path.join(root, file)
                rel_p = os.path.relpath(abs_p, "/root/ldct-benchmark")
                zipf.write(abs_p, rel_p)

    print(f"\n🎉 HOÀN THÀNH! Đã nén {len(collected)} seeds vào file: {zip_path}")

if __name__ == "__main__":
    main()
