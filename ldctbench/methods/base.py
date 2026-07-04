import os

import numpy as np
import torch
import wandb
from torch.autograd import Variable
from tqdm import tqdm

import ldctbench.utils as utils
from ldctbench.data.LDCTMayo import LDCTMayo


class BaseTrainer(object):
    def __init__(self, args, device):
        self.args = args
        self.dev = device

        # Thiết lập dataset train/val cho bài toán LDCT (low-dose CT)
        self.data = {phase: LDCTMayo(phase, self.args) for phase in ["train", "val"]}
        # Tạo dataloader cho từng phase (batch, shuffle, num_workers...)
        self.dataloader = utils.setup_dataloader(self.args, self.data)

        # Các thuộc tính này sẽ được lớp con (vd: RED-CNN Trainer) gán cụ thể:
        # - criterion: hàm tính loss (độ sai)
        # - model: mô hình khử nhiễu (RED-CNN, v.v.)
        # - optimizer: thuật toán cập nhật trọng số (Adam, SGD...)
        self.criterion = None
        self.model = None
        self.optimizer = None

        # Bộ ghi loss và metric (SSIM, PSNR, RMSE) cho cả train và val
        self.losses = utils.metrics.Losses(self.dataloader)
        self.metrics = utils.metrics.Metrics(
            self.dataloader,
            metrics=["SSIM", "PSNR", "RMSE"],
            denormalize_fn=self.data["train"].denormalize,
        )

        self.savedir = wandb.run.dir
        self.iteration = 0

    def train_step(self, batch):
        """Một bước train trên 1 batch dữ liệu.

        Đây là nơi:
        - Cho ảnh đi qua model để khử nhiễu.
        - Tính loss so với ảnh ground truth.
        - Backprop và cập nhật trọng số (học).
        """
        inputs, targets = batch["x"], batch["y"]

        # 1) Cho ảnh nhiễu đi qua model RED-CNN (gọi forward trong network.py)
        outputs = self.model(inputs)
        # 2) Tính độ sai (loss) giữa ảnh dự đoán và ảnh CT chuẩn (high-dose)
        loss = self.criterion(outputs, targets)

        # 3) Xóa gradient cũ
        self.optimizer.zero_grad()
        # 4) Backprop: tính gradient cho từng trọng số trong model
        loss.backward()
        # 5) Cập nhật trọng số theo gradient (optimizer: Adam/SGD...)
        self.optimizer.step()

        # Cập nhật số bước đã train và lưu loss lại để thống kê
        self.iteration += 1
        self.losses.push(loss, "train")

    @torch.no_grad()
    def val_step(self, batch_idx, batch):
        """Một bước validate trên 1 batch dữ liệu.

        Giống train_step nhưng:
        - KHÔNG backprop
        - KHÔNG cập nhật trọng số
        - Có thêm tính metric (SSIM, PSNR, RMSE).
        """
        inputs, targets = batch["x"], batch["y"]

        # Cho ảnh đi qua model (khử nhiễu)
        outputs = self.model(inputs)
        # Tính loss trên tập val (để so sánh với train)
        loss = self.criterion(outputs, targets)

        # Ghi lại loss và metric để cuối epoch tổng hợp
        self.losses.push(loss, "val")
        self.metrics.push(targets, outputs)
        if batch_idx < self.args.valsamples:
            self.log_wandb_images(
                {"low dose": inputs, "prediction": outputs, "high dose": targets}
            )

    def train(self):
        """Chạy train qua toàn bộ dataloader["train"] một lần."""
        self.model.train()
        for batch in tqdm(self.dataloader["train"], desc="Train: "):
            batch = {
                k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()
            }
            self.train_step(batch)
        # Sau khi train xong một vòng, tóm tắt loss train (trung bình, v.v.)
        self.losses.summarize("train")

    def validate(self):
        """Chạy validate trên dataloader["val"] và tính metric."""
        self.images = {}
        self.model.eval()
        for batch_idx, batch in enumerate(
            tqdm(self.dataloader["val"], desc="Validate: ")
        ):
            batch = {
                k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()
            }
            self.val_step(batch_idx, batch)
        self.losses.summarize("val")
        self.metrics.summarize()

    def save_checkpoint(self, to_optimize="SSIM", minimize=False):
        if to_optimize in self.metrics.names:
            values = self.metrics.metrics[to_optimize]
        elif to_optimize in self.losses.names:
            values = self.losses.losses["val"][to_optimize]
        else:
            raise ValueError(
                f"to_optimize must be logged by metrics or losses but got {to_optimize} instead"
            )

        find_opt = np.argmin if minimize else np.argmax

        if find_opt(values) == len(values) - 1:
            # Last value was the best one so far
            print(
                f"Store network at iteration {self.iteration} with {to_optimize}: {values[-1]}"
            )
            checkpoint_path = os.path.join(self.savedir, f"best_{to_optimize}.pt")
            state_dict = (
                self.model.module.state_dict()
                if isinstance(self.args.devices, list)
                else self.model.state_dict()
            )
            torch.save(
                {
                    "args": self.args,
                    "iteration": self.iteration,
                    "model_state_dict": state_dict,
                },
                checkpoint_path,
            )

    def log(self):
        """Lưu checkpoint + log loss/metric + ảnh lên wandb."""
        self.save_checkpoint()
        # Losses
        self.losses.log(self.savedir, self.iteration, self.args.iterations_before_val)
        self.losses.plot(self.savedir)
        # Metrics
        self.metrics.log(self.savedir, self.iteration, self.args.iterations_before_val)
        self.metrics.plot(self.savedir)
        # Validation samples
        wandb.log(self.images, step=self.iteration)

    def log_wandb_images(self, images):
        for tag, img in images.items():
            img = wandb.Image(img.data.cpu()[0], caption=tag)
            if tag not in self.images:
                self.images[tag] = [img]
            else:
                self.images[tag].append(img)

    def fit(self):
        """Vòng lặp huấn luyện tổng thể.

        Lặp lại:
        - train() trên tập train
        - validate() trên tập val
        - log kết quả
        cho tới khi đạt max_iterations.
        """
        # Resume from checkpoint nếu được chỉ định
        if hasattr(self.args, "resume") and self.args.resume and os.path.exists(self.args.resume):
            print(f"Resuming from checkpoint: {self.args.resume}")
            checkpoint = torch.load(self.args.resume, map_location=self.dev, weights_only=False)
            state_dict = checkpoint["model_state_dict"]
            if isinstance(self.args.devices, list):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            self.iteration = checkpoint["iteration"]
            print(f"✅ Resumed at iteration {self.iteration}, continuing to {self.args.max_iterations}")

        delta_seed = self.iteration // self.args.iterations_before_val
        while self.iteration < self.args.max_iterations:
            torch.manual_seed(self.args.seed + delta_seed)
            np.random.seed(self.args.seed + delta_seed)

            # Train and validate
            self.train()
            self.validate()

            # Log
            self.log()
            self.losses.reset()
            self.metrics.reset()

            delta_seed += 1
