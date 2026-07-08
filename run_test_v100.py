import os, sys, torch
import pandas as pd
from tqdm import tqdm
from ldctbench.data import TestData
from ldctbench.evaluate import compute_metric
from ldctbench.methods.edrrednet.network import Model

def main():
    print("=== BẮT ĐẦU CHẤM ĐIỂM (TEST) TRÊN V100 ===")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dev = torch.device("cuda:0")
    
    # 1. Load data
    print("Đang tải dữ liệu Test...")
    data = TestData("data/AAPM-Mayo Clinic", "meanstd")
    
    # 2. Load model
    print("Đang tải mô hình best_SSIM.pt (Seed 42)...")
    net = Model(args=None).to(dev)
    ckpt_path = "wandb/offline-run-20260708_083042-14q2ac5o/files/best_SSIM.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Không tìm thấy file {ckpt_path}!")
        sys.exit(1)
        
    checkpoint = torch.load(ckpt_path, map_location=dev)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Handle DataParallel prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    net.load_state_dict(new_state_dict)
    net.eval()
    
    # 3. Evaluate
    metrics = []
    print("Đang tiến hành chấm điểm từng ảnh...")
    with torch.no_grad():
        for i, patient in enumerate(data):
            patient_name = patient["info"]["id"]
            exam_type = patient["f_hd"][0][0].split("/")[1][0]
            
            for slice_idx in tqdm(range(patient["info"]["n_slices"]), desc=f"Bệnh nhân {patient_name}"):
                x = torch.unsqueeze(torch.unsqueeze(patient["x"][slice_idx], 0), 0).to(dev)
                y = torch.unsqueeze(torch.unsqueeze(patient["y"][slice_idx], 0), 0).to(dev)
                
                y_hat = net(x)
                res = compute_metric(
                    y, y_hat, metrics=["SSIM", "PSNR", "RMSE"],
                    denormalize_fn=data.denormalize, exam_type=exam_type
                )
                
                for m, vals in res.items():
                    metrics.append({
                        "patient": patient_name,
                        "slice": slice_idx,
                        "metric": m,
                        "value": vals[0]
                    })
    
    # 4. Export to CSV
    df = pd.DataFrame(metrics)
    # Tính trung bình theo từng metric
    summary = df.groupby('metric')['value'].mean().reset_index()
    print("\n=== ĐIỂM TRUNG BÌNH TOÀN TẬP TEST ===")
    print(summary.to_string(index=False))
    
    df.to_csv("results_seed42.csv", index=False)
    print(f"\n✅ XONG! Đã xuất toàn bộ kết quả chi tiết ra file: results_seed42.csv")

if __name__ == "__main__":
    main()
