from argparse import Namespace

import torch
import torch.nn as nn

from ldctbench.methods.base import BaseTrainer
from ldctbench.utils.training_utils import setup_optimizer

from .network import Model


class Trainer(BaseTrainer):
    """Trainer dùng cho mô hình RED-CNN[^1].

    [^1]: H. Chen et al., “Low-dose CT with a residual encoder-decoder convolutional neural network,” IEEE Transactions on Medical Imaging, vol. 36, no. 12, pp. 2524–2535, Dec. 2017.

    """

    def __init__(self, args: Namespace, device: torch.device):
        """Hàm khởi tạo Trainer.

        Parameters
        ----------
        args : Namespace
            Arguments to configure the trainer.
        device : torch.device
            Torch device to use for training.
        """
        # Gọi BaseTrainer để thiết lập các phần chung (dataloader, log, thiết bị, vòng lặp train cơ bản, ...)
        super().__init__(args, device)

        # Hàm "chấm điểm" cho mô hình: Mean Squared Error (MSE)
        # - So sánh từng pixel giữa ảnh model dự đoán và ảnh CT chuẩn (ground truth)
        # - MSE càng nhỏ thì model khử nhiễu càng tốt
        self.criterion = nn.MSELoss()

        # Khởi tạo mô hình RED-CNN (định nghĩa trong network.py) và đưa lên thiết bị (CPU hoặc GPU)
        self.model = Model(args).to(self.dev)

        # Nếu cấu hình yêu cầu dùng nhiều GPU (danh sách device id),
        # bọc mô hình trong DataParallel để tự động chia batch lên các GPU
        if isinstance(self.args.devices, list):
            self.model = nn.DataParallel(self.model, device_ids=self.args.devices)

        # Tạo optimizer (thường là Adam) để cập nhật trọng số của model trong quá trình training.
        # Các tham số như learning rate, beta, weight_decay... được lấy từ args / file config (vd: configs/redcnn.yaml).
        self.optimizer = setup_optimizer(args, self.model.parameters())
