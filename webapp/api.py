"""
EDR-REDNet Visualizer — FastAPI Backend
Rebuilt from Streamlit for better performance.
"""

import os
import sys
import io
import base64
import shutil
import argparse
import tempfile
from typing import Optional, List

import numpy as np
import torch
import pydicom
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from skimage import filters, metrics

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# ── Path Setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ── PyTorch safe_load patch ───────────────────────────────────────────────────
_orig_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    if "map_location" not in kwargs:
        kwargs["map_location"] = "cpu"
    return _orig_load(*args, **kwargs)
torch.load = _safe_load

def _safe_load_yaml(path: str):
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

import ldctbench.evaluate.utils
import ldctbench.utils
ldctbench.evaluate.utils.torch.load = _safe_load
ldctbench.evaluate.utils.load_yaml = _safe_load_yaml
ldctbench.utils.load_yaml = _safe_load_yaml

from ldctbench.data import TestData
from ldctbench.evaluate import setup_trained_model
from ldctbench.hub import load_model as hub_load_model

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_dataset: Optional[TestData] = None
_networks: Optional[dict] = None

def _setup_wandb_dir(run_name: str, ckpt_path: str, cfg_path: str, overrides: dict = None):
    d = os.path.join(ROOT, "wandb", run_name, "files")
    os.makedirs(d, exist_ok=True)
    if os.path.exists(ckpt_path) and os.path.exists(cfg_path):
        if overrides:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg.update(overrides)
            with open(os.path.join(d, "args.yaml"), "w", encoding="utf-8") as f:
                yaml.dump(cfg, f)
        else:
            shutil.copy(cfg_path, os.path.join(d, "args.yaml"))
        shutil.copy(ckpt_path, os.path.join(d, "best_SSIM.pt"))
        return True
    return False


def _get_dataset() -> TestData:
    global _dataset
    if _dataset is None:
        data_dir = os.path.join(ROOT, "AAPM-Mayo Clinic")
        _dataset = TestData(data_dir, "meanstd")
    return _dataset


def _get_networks() -> dict:
    global _networks
    if _networks is None:
        cfg = os.path.join(ROOT, "configs", "edrrednet.yaml")
        ckpt = os.path.join(ROOT, "best_SSIM.pt")
        _setup_wandb_dir("edr_redcnn_latest", ckpt, cfg)
        _setup_wandb_dir("edr_variant_b",
            os.path.join(ROOT, "results", "training", "EDR-RedCnn", "VariantB", "Seed1339", "variantB_seed1339_best_SSIM.pt"),
            cfg, overrides={"use_sobel_input": False})
        _setup_wandb_dir("edr_variant_c",
            os.path.join(ROOT, "results", "training", "EDR-RedCnn", "VariantC", "Seed1339", "variantC_seed1339_best_SSIM.pt"),
            cfg, overrides={"use_sobel_input": True})

        nets = {}
        nets["redcnn"] = hub_load_model("redcnn", eval=True).to(DEVICE)
        nets["variant_a"] = nets["redcnn"]

        net_d = setup_trained_model(run_name="edr_redcnn_latest", device=DEVICE,
            network_name="Model", state_dict="best_SSIM", eval=True)
        nets["edr_redcnn"] = net_d
        nets["variant_d"] = net_d

        for name, run_name in [("variant_b", "edr_variant_b"), ("variant_c", "edr_variant_c")]:
            try:
                nets[name] = setup_trained_model(run_name=run_name, device=DEVICE,
                    network_name="Model", state_dict="best_SSIM", eval=True)
            except Exception:
                nets[name] = None

        _networks = nets
    return _networks


# ── Image Utilities ───────────────────────────────────────────────────────────
def _to_hu(tensor, ds: TestData) -> np.ndarray:
    arr = ds.denormalize(tensor.cpu().squeeze()).numpy()
    return ds._convert_hu(arr, to_hu=True)


def _window(img: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    return (np.clip(img, hu_min, hu_max) - hu_min) / (hu_max - hu_min)


def _ndarray_to_b64(img_windowed: np.ndarray, cmap: str = "gray") -> str:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(img_windowed, cmap=cmap, interpolation="lanczos")
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _get_edge_map(img: np.ndarray) -> np.ndarray:
    return np.hypot(ndimage.sobel(img, axis=0), ndimage.sobel(img, axis=1))


def _calc_metrics(pred: np.ndarray, target: np.ndarray,
                  roi_target: tuple = None, roi_bg: tuple = None) -> dict:
    vmin, vmax = -1024.0, 3000.0
    p_n = (np.clip(pred, vmin, vmax) - vmin) / (vmax - vmin)
    t_n = (np.clip(target, vmin, vmax) - vmin) / (vmax - vmin)
    ssim_v = float(metrics.structural_similarity(t_n, p_n, data_range=1.0))
    psnr_v = float(metrics.peak_signal_noise_ratio(t_n, p_n, data_range=1.0))
    ep = _get_edge_map(p_n); et = _get_edge_map(t_n)
    ep = (ep - ep.min()) / (ep.max() - ep.min() + 1e-8)
    et = (et - et.min()) / (et.max() - et.min() + 1e-8)
    edge_ssim = float(metrics.structural_similarity(et, ep, data_range=1.0))

    result = {"SSIM": ssim_v, "PSNR": psnr_v, "Edge_SSIM": edge_ssim}

    try:
        from sewar.full_ref import vifp
        result["VIF"] = float(vifp(t_n, p_n))
    except Exception:
        result["VIF"] = None

    if roi_target and roi_bg:
        tx, ty, ts = roi_target
        bx, by, bs = roi_bg
        pred_roi_t = pred[ty:ty+ts, tx:tx+ts]
        pred_roi_b = pred[by:by+bs, bx:bx+bs]
        mean_t = float(np.mean(pred_roi_t))
        mean_b = float(np.mean(pred_roi_b))
        std_b  = float(np.std(pred_roi_b)) + 1e-8
        result["CNR"] = float(abs(mean_t - mean_b) / std_b)
        result["HU_Dev"] = float(std_b)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="EDR-REDNet Visualizer API")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── /api/info ─────────────────────────────────────────────────────────────────
@app.get("/api/info")
def api_info():
    return {
        "device": str(DEVICE),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# ── /api/patients ─────────────────────────────────────────────────────────────
@app.get("/api/patients")
def api_patients():
    ds = _get_dataset()
    return [{"idx": i, "id": s["info"]["id"], "n_slices": s["n"]}
            for i, s in enumerate(ds.samples)]


# ── /api/slice ────────────────────────────────────────────────────────────────
@app.get("/api/slice")
def api_slice(
    pat_idx: int = Query(0),
    slice_idx: int = Query(0),
    hu_min: float = Query(-160),
    hu_max: float = Query(245),
    show_diff: bool = Query(False),
    show_edge: bool = Query(False),
):
    ds = _get_dataset()
    nets = _get_networks()

    s = ds.samples[pat_idx]
    x_np = pydicom.dcmread(s["in_files"][slice_idx]).pixel_array.astype("float32")
    y_np = pydicom.dcmread(s["tg_files"][slice_idx]).pixel_array.astype("float32")
    x_norm = ds._normalize(x_np)
    y_norm = ds._normalize(y_np)
    x_t = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    y_t = torch.from_numpy(y_norm)

    with torch.no_grad():
        pred_redcnn = nets["redcnn"](x_t)
        pred_edr    = nets["edr_redcnn"](x_t)

    img_ld     = _to_hu(x_t, ds)
    img_redcnn = _to_hu(pred_redcnn, ds)
    img_edr    = _to_hu(pred_edr, ds)
    img_ndct   = _to_hu(y_t, ds)

    result = {
        "images": {
            "ldct":   _ndarray_to_b64(_window(img_ld,     hu_min, hu_max)),
            "redcnn": _ndarray_to_b64(_window(img_redcnn, hu_min, hu_max)),
            "edr":    _ndarray_to_b64(_window(img_edr,    hu_min, hu_max)),
            "ndct":   _ndarray_to_b64(_window(img_ndct,   hu_min, hu_max)),
        },
        "metrics": {
            "redcnn": _calc_metrics(img_redcnn, img_ndct),
            "edr":    _calc_metrics(img_edr,    img_ndct),
        }
    }

    if show_diff:
        result["diff"] = {
            "ldct":   _ndarray_to_b64(np.clip((img_ld     - img_ndct + 200) / 400, 0, 1), "seismic"),
            "redcnn": _ndarray_to_b64(np.clip((img_redcnn - img_ndct + 200) / 400, 0, 1), "seismic"),
            "edr":    _ndarray_to_b64(np.clip((img_edr    - img_ndct + 200) / 400, 0, 1), "seismic"),
        }

    if show_edge:
        def edge_b64(img):
            e = filters.sobel(img)
            e = (e - e.min()) / (e.max() - e.min() + 1e-8)
            return _ndarray_to_b64(e)
        result["edge"] = {
            "ldct":   edge_b64(img_ld),
            "redcnn": edge_b64(img_redcnn),
            "edr":    edge_b64(img_edr),
            "ndct":   edge_b64(img_ndct),
        }

    return result


# ── /api/upload ───────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def api_upload(
    ldct_file: UploadFile = File(...),
    ndct_file: UploadFile = File(None),
    hu_min: float = Query(-160),
    hu_max: float = Query(245),
):
    ds = _get_dataset()
    nets = _get_networks()

    async def read_dcm(upload: UploadFile) -> np.ndarray:
        content = await upload.read()
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        dcm = pydicom.dcmread(tmp_path)
        os.unlink(tmp_path)
        slope = float(getattr(dcm, "RescaleSlope", 1))
        intercept = float(getattr(dcm, "RescaleIntercept", 0))
        hu = dcm.pixel_array.astype("float32") * slope + intercept
        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            hu = np.max(hu) - hu
        return hu

    x_hu = await read_dcm(ldct_file)
    if x_hu.shape != (512, 512):
        import cv2
        x_hu = cv2.resize(x_hu, (512, 512))

    x_raw = x_hu + 1024.0
    x_norm = ds._normalize(x_raw)
    x_t = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_redcnn = nets["redcnn"](x_t)
        pred_edr    = nets["edr_redcnn"](x_t)

    img_ld     = _to_hu(x_t, ds)
    img_redcnn = _to_hu(pred_redcnn, ds)
    img_edr    = _to_hu(pred_edr, ds)

    result = {
        "images": {
            "ldct":   _ndarray_to_b64(_window(img_ld,     hu_min, hu_max)),
            "redcnn": _ndarray_to_b64(_window(img_redcnn, hu_min, hu_max)),
            "edr":    _ndarray_to_b64(_window(img_edr,    hu_min, hu_max)),
        },
        "metrics": {}
    }

    if ndct_file:
        y_hu = await read_dcm(ndct_file)
        if y_hu.shape != (512, 512):
            import cv2
            y_hu = cv2.resize(y_hu, (512, 512))
        img_ndct = y_hu
        result["images"]["ndct"] = _ndarray_to_b64(_window(img_ndct, hu_min, hu_max))
        result["metrics"] = {
            "redcnn": _calc_metrics(img_redcnn, img_ndct),
            "edr":    _calc_metrics(img_edr,    img_ndct),
        }

    return result


# ── /api/ablation ─────────────────────────────────────────────────────────────
@app.get("/api/ablation")
def api_ablation(
    pat_idx: int = Query(0),
    slice_idx: int = Query(0),
    hu_min: float = Query(-160),
    hu_max: float = Query(245),
    tx: int = Query(200), ty: int = Query(250), ts: int = Query(20),
    bx: int = Query(100), by: int = Query(250), bs: int = Query(30),
):
    ds = _get_dataset()
    nets = _get_networks()

    s = ds.samples[pat_idx]
    x_np = pydicom.dcmread(s["in_files"][slice_idx]).pixel_array.astype("float32")
    y_np = pydicom.dcmread(s["tg_files"][slice_idx]).pixel_array.astype("float32")
    x_t = torch.from_numpy(ds._normalize(x_np)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    y_t = torch.from_numpy(ds._normalize(y_np))

    img_ndct = _to_hu(y_t, ds)

    VARIANTS = {
        "A — RED-CNN (Baseline)": "variant_a",
        "B — +EdgeBlock":         "variant_b",
        "C — +Sobel Input":       "variant_c",
        "D — Full EDR-REDNet":    "variant_d",
    }
    COLORS = {"A — RED-CNN (Baseline)": "#888888", "B — +EdgeBlock": "#4e9af1",
              "C — +Sobel Input": "#f1a74e", "D — Full EDR-REDNet": "#2ecc71"}

    images, metrics_out = {}, {}
    with torch.no_grad():
        images["LDCT (Input)"] = _ndarray_to_b64(_window(_to_hu(x_t, ds), hu_min, hu_max))
        images["NDCT (Ref)"]   = _ndarray_to_b64(_window(img_ndct, hu_min, hu_max))
        for vname, key in VARIANTS.items():
            net = nets.get(key)
            if net is not None:
                img_v = _to_hu(net(x_t), ds)
                images[vname] = _ndarray_to_b64(_window(img_v, hu_min, hu_max))
                metrics_out[vname] = _calc_metrics(img_v, img_ndct,
                    roi_target=(tx, ty, ts), roi_bg=(bx, by, bs))
            else:
                images[vname] = None
                metrics_out[vname] = None

    # ROI preview (draw boxes on NDCT)
    roi_img = _window(img_ndct, hu_min, hu_max)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
    ax.imshow(roi_img, cmap="gray")
    ax.add_patch(mpatches.Rectangle((tx, ty), ts, ts, lw=2, ec="red",   fc="none"))
    ax.add_patch(mpatches.Rectangle((bx, by), bs, bs, lw=2, ec="cyan",  fc="none"))
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    roi_preview = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    return {"images": images, "metrics": metrics_out,
            "colors": COLORS, "roi_preview": roi_preview}


# ── /api/zoom ─────────────────────────────────────────────────────────────────
@app.get("/api/zoom")
def api_zoom(
    pat_idx: int = Query(0),
    slice_idx: int = Query(0),
    x: int = Query(180), y: int = Query(220),
    w: int = Query(120), h: int = Query(120),
    hu_min: float = Query(-160), hu_max: float = Query(245),
):
    ds = _get_dataset()
    nets = _get_networks()

    s = ds.samples[pat_idx]
    x_np = pydicom.dcmread(s["in_files"][slice_idx]).pixel_array.astype("float32")
    y_np = pydicom.dcmread(s["tg_files"][slice_idx]).pixel_array.astype("float32")
    x_t = torch.from_numpy(ds._normalize(x_np)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    y_t = torch.from_numpy(ds._normalize(y_np))

    VARIANTS = {"LDCT": None, "A — Baseline": "variant_a",
                "B — +EdgeBlock": "variant_b", "C — +Sobel": "variant_c",
                "D — EDR-REDNet": "variant_d", "NDCT (GT)": "ndct"}

    full_imgs, zoom_imgs, diff_imgs, sobel_imgs = {}, {}, {}, {}
    img_ndct = _to_hu(y_t, ds)
    x1, y1 = x, y
    x2, y2 = min(x + w, 512), min(y + h, 512)

    with torch.no_grad():
        all_imgs = {"LDCT": _to_hu(x_t, ds)}
        for key in ["variant_a", "variant_b", "variant_c", "variant_d"]:
            net = nets.get(key)
            short = {"variant_a": "A — Baseline", "variant_b": "B — +EdgeBlock",
                     "variant_c": "C — +Sobel",   "variant_d": "D — EDR-REDNet"}[key]
            if net is not None:
                all_imgs[short] = _to_hu(net(x_t), ds)
        all_imgs["NDCT (GT)"] = img_ndct

    for lbl, img in all_imgs.items():
        win = _window(img, hu_min, hu_max)
        # Full image with yellow box
        fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
        ax.imshow(win, cmap="gray")
        ax.add_patch(mpatches.Rectangle((x1, y1), x2-x1, y2-y1, lw=2, ec="yellow", fc="none"))
        ax.set_title(lbl, fontsize=7); ax.axis("off")
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig); buf.seek(0)
        full_imgs[lbl] = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

        # Zoom
        crop = img[y1:y2, x1:x2]
        zoom_imgs[lbl] = _ndarray_to_b64(_window(crop, hu_min, hu_max))

        # Diff
        if lbl not in ["LDCT", "NDCT (GT)"]:
            ndct_crop = img_ndct[y1:y2, x1:x2]
            diff = crop - ndct_crop
            diff_imgs[lbl] = _ndarray_to_b64(np.clip((diff + 200) / 400, 0, 1), "seismic")

        # Sobel
        edge = filters.sobel(crop if lbl not in ["LDCT", "NDCT (GT)"] else crop)
        edge_n = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
        sobel_imgs[lbl] = _ndarray_to_b64(edge_n, "hot")

    return {"full": full_imgs, "zoom": zoom_imgs,
            "diff": diff_imgs, "sobel": sobel_imgs}


# ── /api/boxplot ──────────────────────────────────────────────────────────────
@app.get("/api/boxplot")
def api_boxplot():
    import pandas as pd
    csv_path = os.path.join(ROOT, "results", "evaluation", "per_patient_scores.csv")
    pval_path = os.path.join(ROOT, "results", "evaluation", "wilcoxon_pvalues.csv")

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="per_patient_scores.csv not found")

    df = pd.read_csv(csv_path)
    VARIANT_COLS = {
        "A — RED-CNN": ("A_PSNR", "A_SSIM", "A_Edge_SSIM"),
        "B — +EdgeBlock": ("B_PSNR", "B_SSIM", "B_Edge_SSIM"),
        "C — +Sobel": ("C_PSNR", "C_SSIM", "C_Edge_SSIM"),
        "D — EDR-REDNet": ("D_PSNR", "D_SSIM", "D_Edge_SSIM"),
    }

    boxplot_data = {}
    for vname, cols in VARIANT_COLS.items():
        boxplot_data[vname] = {}
        for metric, col in zip(["PSNR", "SSIM", "Edge_SSIM"], cols):
            if col in df.columns:
                vals = df[col].dropna().tolist()
                boxplot_data[vname][metric] = vals

    pval_data = None
    if os.path.exists(pval_path):
        pval_data = pd.read_csv(pval_path).to_dict(orient="records")

    return {"data": boxplot_data, "pvalues": pval_data}
