import os
import random
from argparse import Namespace
from typing import Dict, List, Optional, Callable

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ldctbench.utils import load_yaml


class LDCTMayo(Dataset):
    """Dataset class for the LDCT dataset (Training/Val)"""

    def __init__(self, mode: str, args: Namespace):
        # Set seeds
        self.seed = args.seed
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.path = args.datafolder
        if not hasattr(args, "eval_patchsize"):
            args.eval_patchsize = 128
        self.patchsize = (
            args.eval_patchsize if mode == "val" else args.patchsize
        )
        self.data_subset = args.data_subset
        self.data_norm = args.data_norm
        self.info = load_yaml(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "info.yml")
        )

        # Cache for sorted file paths
        self.patient_cache = {}

        # Get all slices
        self.samples = [
            {**patient_dict, "slice_idx": s}
            for patient_dict in self.info[mode + "_set"]
            for s in range(patient_dict["n_slices"])
        ]
        random.shuffle(self.samples)

        if self.data_subset < 1.0:
            self.samples = self.samples[: int(len(self.samples) * self.data_subset)]

        # WeightedRandomSampler yêu cầu thuộc tính weights
        # Dùng uniform weights (tất cả bằng nhau) = random sampling thông thường
        self.weights = torch.ones(len(self.samples))

    def _get_sorted_files(self, folder_rel_path):
        if folder_rel_path in self.patient_cache:
            return self.patient_cache[folder_rel_path]

        folder_abs_path = os.path.join(self.path, folder_rel_path[2:])
        if not os.path.exists(folder_abs_path):
            return []

        files = [f for f in os.listdir(folder_abs_path) if f.endswith(".dcm")]
        slice_info = []
        for f in files:
            p = os.path.join(folder_abs_path, f)
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True)
                loc = float(getattr(ds, "SliceLocation", 0))
                slice_info.append((loc, p))
            except:
                slice_info.append((0, p))
        
        slice_info.sort(key=lambda x: x[0])
        sorted_paths = [x[1] for x in slice_info]
        self.patient_cache[folder_rel_path] = sorted_paths
        return sorted_paths

    def _normalize(self, X):
        if self.data_norm == "meanstd":
            return (X - self.info["mean"]) / self.info["std"]
        elif self.data_norm == "minmax":
            return (X - float(self.info["min"])) / (float(self.info["max"]) - float(self.info["min"]))
        return X

    def denormalize(self, X):
        if self.data_norm == "meanstd":
            return X * self.info["std"] + self.info["mean"]
        elif self.data_norm == "minmax":
            return X * (self.info["max"] - self.info["min"]) + self.info["min"]
        return X

    def _random_crop(self, images):
        if self.patchsize and images[0].shape[0] > self.patchsize:
            x = np.random.randint(images[0].shape[0] - self.patchsize)
            y = np.random.randint(images[0].shape[1] - self.patchsize)
            return [im[x : x + self.patchsize, y : y + self.patchsize] for im in images]
        return images

    @staticmethod
    def to_torch(X):
        return torch.unsqueeze(torch.from_numpy(X), 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        in_files = self._get_sorted_files(sample["input"])
        tg_files = self._get_sorted_files(sample["target"])
        
        si = sample["slice_idx"]
        if si >= len(in_files) or si >= len(tg_files): si = 0
            
        x = pydicom.dcmread(in_files[si]).pixel_array.astype("float32")
        y = pydicom.dcmread(tg_files[si]).pixel_array.astype("float32")
        x, y = self._random_crop([x, y])
        return {"x": self.to_torch(self._normalize(x)), "y": self.to_torch(self._normalize(y))}


class TestData(Dataset):
    """Lazy Loading Test Dataset to save RAM"""
    def __init__(self, datafolder, data_norm):
        self.path = datafolder
        self.data_norm = data_norm
        self.info = load_yaml(os.path.join(os.path.dirname(os.path.realpath(__file__)), "info.yml"))
        
        self.samples = []
        for patient_dict in self.info["test_set"]:
            # Chỉ lưu đường dẫn, không load ảnh vào RAM
            in_folder = os.path.join(self.path, patient_dict["input"][2:])
            tg_folder = os.path.join(self.path, patient_dict["target"][2:])
            
            in_files = self._get_sorted_paths(in_folder)
            tg_files = self._get_sorted_paths(tg_folder)
            
            n = min(len(in_files), len(tg_files))
            if n == 0:
                continue  # Bỏ qua bệnh nhân không có file .dcm cục bộ
            self.samples.append({
                "info": patient_dict,
                "in_files": in_files,
                "tg_files": tg_files,
                "n": n
            })

    def _get_sorted_paths(self, folder):
        if not os.path.exists(folder): return []
        files = [f for f in os.listdir(folder) if f.endswith(".dcm")]
        info = []
        for f in files:
            p = os.path.join(folder, f)
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True)
                info.append((float(getattr(ds, "SliceLocation", 0)), p))
            except: info.append((0, p))
        info.sort(key=lambda x: x[0])
        return [x[1] for x in info]

    def _normalize(self, X):
        if self.data_norm == "meanstd":
            return (X - self.info["mean"]) / self.info["std"]
        elif self.data_norm == "minmax":
            return (X - float(self.info["min"])) / (float(self.info["max"]) - float(self.info["min"]))
        return X

    def denormalize(self, X):
        if self.data_norm == "meanstd":
            return X * self.info["std"] + self.info["mean"]
        elif self.data_norm == "minmax":
            return X * (self.info["max"] - self.info["min"]) + self.info["min"]
        return X

    def _convert_hu(self, X, to_hu=True):
        return X - 1024.0 if to_hu else X + 1024.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load full patient data lazily when requested"""
        s = self.samples[idx]
        px, py = [], []
        fld, fhd = [], []
        
        for i in range(s["n"]):
            x = pydicom.dcmread(s["in_files"][i]).pixel_array.astype("float32")
            y = pydicom.dcmread(s["tg_files"][i]).pixel_array.astype("float32")
            px.append(x)
            py.append(y)
            fld.append((os.path.dirname(s["in_files"][i]), os.path.basename(s["in_files"][i])))
            fhd.append((os.path.dirname(s["tg_files"][i]), os.path.basename(s["tg_files"][i])))
            
        return {
            "info": s["info"],
            "x": torch.from_numpy(self._normalize(np.stack(px))),
            "y": torch.from_numpy(self._normalize(np.stack(py))),
            "f_ld": fld, "f_hd": fhd
        }
