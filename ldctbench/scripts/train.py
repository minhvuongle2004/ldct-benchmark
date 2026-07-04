"""
Script huấn luyện (training) mô hình khử nhiễu cho ảnh CT liều thấp (LDCT)
Sử dụng Weights & Biases (wandb) để theo dõi quá trình training
"""

import glob  # Thư viện tìm kiếm file theo pattern
import importlib  # Thư viện import module động (import tên module từ string)
import os  # Thư viện xử lý hệ thống file và biến môi trường
import shutil  # Thư viện sao chép/di chuyển file
import time  # Thư viện xử lý thời gian
import warnings  # Thư viện hiển thị cảnh báo

import matplotlib  # Thư viện vẽ đồ thị
import numpy as np  # Thư viện tính toán số học
import torch  # Thư viện deep learning PyTorch
import wandb  # Weights & Biases - công cụ theo dõi experiment ML

import ldctbench.utils.auxiliaries as aux  # Các hàm tiện ích của dự án
from ldctbench.utils.argparser import make_parser, use_config  # Parser cho arguments

# ============================================================================
# CẤU HÌNH MATPLOTLIB
# ============================================================================
# Sử dụng backend "Agg" (non-GUI) để vẽ đồ thị trên server không có màn hình
matplotlib.use("Agg")

# ============================================================================
# CẤU HÌNH BIẾN MÔI TRƯỜNG CHO WANDB
# ============================================================================
# Sử dụng "thread" để spawn subprocess trên cluster node
# Tránh lỗi khi chạy trên hệ thống phân tán/cluster
os.environ["WANDB_START_METHOD"] = "thread"

# Không upload các file .pt (model weights) lên wandb để tiết kiệm dung lượng
# File .pt thường rất nặng (hàng trăm MB đến GB)
os.environ["WANDB_IGNORE_GLOBS"] = "*.pt"


# ============================================================================
# HÀM CHÍNH: HUẤN LUYỆN MÔ HÌNH
# ============================================================================
def train(args):
    """
    Hàm thực hiện quá trình huấn luyện mô hình
    
    Args:
        args: Object chứa tất cả các tham số cấu hình (từ command line hoặc config file)
    """
    
    # ========================================================================
    # BƯỚC 1: THIẾT LẬP RANDOM SEED (Để tái tạo kết quả)
    # ========================================================================
    # Nếu không có seed được chỉ định, tạo seed ngẫu nhiên
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)  # Random từ 1-9999
    
    # Set seed cho PyTorch (đảm bảo khởi tạo weights giống nhau mỗi lần chạy)
    torch.manual_seed(args.seed)
    
    # Set seed cho NumPy (đảm bảo random operations giống nhau)
    np.random.seed(args.seed)

    # ========================================================================
    # BƯỚC 2: THIẾT LẬP GPU/CPU
    # ========================================================================
    # Xử lý 3 trường hợp: nhiều GPU, 1 GPU, hoặc CPU
    
    # Trường hợp 1: Sử dụng NHIỀU GPU (Multi-GPU training)
    if isinstance(args.devices, list) and len(args.devices) > 1:
        # Ví dụ: devices=[0,1,2] -> "CUDA_VISIBLE_DEVICES=0,1,2"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.devices])
        # Đánh số lại GPU từ 0 (0,1,2 thay vì giá trị gốc)
        args.devices = list(range(len(args.devices)))
    
    # Trường hợp 2: Sử dụng 1 GPU
    elif isinstance(args.devices, list) and len(args.devices) == 1:
        # Ví dụ: devices=[2] -> "CUDA_VISIBLE_DEVICES=2"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices[0])
        args.devices = 0  # Đánh số lại thành 0
    
    # Trường hợp 3: Giá trị đơn (int), không phải list
    else:
        # Ví dụ: devices=1 -> "CUDA_VISIBLE_DEVICES=1"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
        args.devices = 0  # Đánh số lại thành 0
    
    # Tạo device object (cuda hoặc cpu)
    device = torch.device("cuda" if args.cuda else "cpu")

    # ========================================================================
    # BƯỚC 3: KHỞI TẠO WEIGHTS & BIASES (WANDB) - Công cụ theo dõi training
    # ========================================================================
    # Vòng lặp vô hạn để retry nếu kết nối mạng bị gián đoạn
    while True:
        try:
            # Kiểm tra xem có tag (nhãn) cho experiment không
            if hasattr(args, "wandbtag") and args.wandbtag:
                # Khởi tạo wandb với project name và tags
                # Tags giúp phân loại các experiment (ví dụ: "baseline", "v2", "final")
                wandb.init(project="ldct-benchmark", config=args, tags=[args.wandbtag])
            else:
                # Khởi tạo wandb không có tags
                wandb.init(project="ldct-benchmark", config=args)
            
            # Nếu khởi tạo thành công, thoát vòng lặp
            break
            
        except Exception as e:
            # Nếu có lỗi (thường do mạng), in lỗi và retry sau 10 giây
            print(f"{e} ... retrying...")
            time.sleep(10)

    # ========================================================================
    # BƯỚC 4: ĐẶT TÊN CHO RUN VÀ LƯU CẤU HÌNH
    # ========================================================================
    # Lấy tên run từ đường dẫn thư mục wandb
    # Ví dụ: "/path/to/wandb/run-20231201_120000-abc123" -> "20231201_120000-abc123"
    wandb.run.name = wandb.run.dir.split(os.sep)[-2].split("run-")[-1]
    
    # Lưu toàn bộ cấu hình (args) vào file trong thư mục wandb
    # Giúp tái tạo lại experiment sau này
    aux.dump_config(args, wandb.run.dir)

    # ========================================================================
    # BƯỚC 5: LOAD TRAINER CLASS (Dynamic Import)
    # ========================================================================
    # Import module trainer dựa trên tên được chỉ định trong args
    # Ví dụ: args.trainer="redcnn" -> import ldctbench.methods.redcnn.Trainer
    # LƯU Ý: Toàn bộ logic load dữ liệu (DataLoader) và training loop nằm bên trong class Trainer này
    try:
        trainer_module = importlib.import_module(
            "ldctbench.methods.{}.Trainer".format(args.trainer)
        )
    except ModuleNotFoundError:
        # Nếu không tìm thấy module, raise error với thông báo rõ ràng
        raise ValueError(
            "Trainer {0} not known and module methods.{0}.Trainer not found".format(
                args.trainer
            )
        )
    
    # Lấy class "Trainer" từ module vừa import
    # getattr(object, "attribute_name") tương đương object.attribute_name
    trainer_class = getattr(trainer_module, "Trainer")
    
    # Khởi tạo instance của Trainer class
    # Truyền vào args (cấu hình) và device (GPU/CPU)
    trainer = trainer_class(args, device)

    # ========================================================================
    # BƯỚC 6: BẮT ĐẦU HUẤN LUYỆN
    # ========================================================================
    print("Start training...")
    # Gọi phương thức fit() để bắt đầu quá trình training
    # Phương thức này được định nghĩa trong từng Trainer class cụ thể
    trainer.fit()


# ============================================================================
# HÀM MAIN: ĐIỂM BẮT ĐẦU CỦA CHƯƠNG TRÌNH
# ============================================================================
def main():
    """
    Hàm main để parse arguments và gọi hàm train
    """
    
    # ========================================================================
    # BƯỚC 1: PARSE ARGUMENTS TỪ COMMAND LINE
    # ========================================================================
    # Tạo argument parser (xử lý các tham số từ dòng lệnh)
    parser = make_parser()
    
    # Parse các arguments người dùng nhập vào
    # Ví dụ: python train.py --trainer redcnn --epochs 100
    args = parser.parse_args()
    
    # Load cấu hình từ config file (nếu có)
    # Ghi đè hoặc bổ sung vào args từ command line
    args = use_config(args)
    
    # ========================================================================
    # BƯỚC 2: KIỂM TRA VÀ THIẾT LẬP DATAFOLDER
    # ========================================================================
    # Datafolder là đường dẫn đến thư mục chứa dữ liệu training
    
    # Kiểm tra xem args có chứa datafolder không hoặc datafolder có rỗng không
    if not hasattr(args, "datafolder") or not args.datafolder:
        # Nếu không có trong args, thử lấy từ biến môi trường
        if "LDCTBENCH_DATAFOLDER" in os.environ:
            # Lấy giá trị từ environment variable
            args.datafolder = os.environ["LDCTBENCH_DATAFOLDER"]
            
            # Hiển thị cảnh báo cho biết đang dùng datafolder từ env var
            warnings.warn(
                f"No datafolder in args. Will use the one provided via environment variable LDCTBENCH_DATAFOLDER: {args.datafolder}"
            )
        else:
            # Nếu không có ở cả 2 nơi, raise error với hướng dẫn chi tiết
            raise ValueError(
                "No datafolder provided! Add via\n"
                " \t- Config file: add key: datafolder\n"
                "\t- Arguments: add argument --datafolder\n"
                "\t- Environment variable: export LDCTBENCH_DATAFOLDER=..."
            )

    # ========================================================================
    # BƯỚC 3: XỬ LÝ CHẾ ĐỘ DRYRUN (Test mode)
    # ========================================================================
    # Dryrun mode: Chạy thử không ghi log lên wandb server
    # Hữu ích khi test code, không muốn tạo experiment thật
    if args.dryrun:
        os.environ["WANDB_MODE"] = "dryrun"

    # ========================================================================
    # BƯỚC 4: GỌI HÀM TRAIN
    # ========================================================================
    train(args)


# ============================================================================
# ĐIỂM BẮT ĐẦU CHƯƠNG TRÌNH
# ============================================================================
# Chỉ chạy main() khi file được execute trực tiếp
# Không chạy khi file được import như một module
if __name__ == "__main__":
    main()