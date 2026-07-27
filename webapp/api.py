"""
EDR-REDNet Visualizer — FastAPI Backend
Rebuilt from Streamlit for better performance.
"""

import os
import sys
import io
import json
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
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse

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

import argparse
from ldctbench.data import TestData
from ldctbench.hub import load_model as hub_load_model

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_dataset: Optional[TestData] = None
_networks: Optional[dict] = None

def _load_edr_model(ckpt_path: str, use_sobel: bool = True) -> torch.nn.Module:
    """Load EDR-REDNet directly from .pt file without wandb setup."""
    from ldctbench.methods.edrrednet.network import Model
    if not os.path.exists(ckpt_path):
        return None
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Strip module. prefix
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k[7:] if k.startswith("module.") else k] = v
    mock_args = argparse.Namespace(use_sobel_input=use_sobel, num_edge_blocks=2)
    net = Model(args=mock_args).to(DEVICE)
    net.load_state_dict(new_sd)
    net.eval()
    return net


def _get_dataset() -> TestData:
    global _dataset
    if _dataset is None:
        data_dir = os.path.join(ROOT, "AAPM-Mayo Clinic")
        _dataset = TestData(data_dir, "meanstd")
        
        # Inject 20 NLST patients
        nlst_pids = ['133786','110253','109589','107058','108352','126576','124323','104683','120790','129511',
                     '100586','101259','102332','102607','102361','101459','101662','100885','100334','100004']
        nlst_root = os.path.join(ROOT, "National Lung Screening Trial (NLST)", "NLST")
        if os.path.exists(nlst_root):
            for pid in nlst_pids:
                pat_dir = os.path.join(nlst_root, pid)
                if not os.path.exists(pat_dir): continue
                
                # Find deepest dir with .dcm
                dcm_count = 0
                series_dir = ''
                for r, d, f in os.walk(pat_dir):
                    dcms = [x for x in f if x.endswith('.dcm')]
                    if len(dcms) > dcm_count:
                        dcm_count = len(dcms)
                        series_dir = r
                
                if dcm_count > 0:
                    dcms = sorted([os.path.join(series_dir, x) for x in os.listdir(series_dir) if x.endswith('.dcm')])
                    _dataset.samples.append({
                        "info": {"id": f"[NLST] {pid}"},
                        "in_files": dcms,
                        "tg_files": dcms,  # Mock target with input so metric logic doesn't crash
                        "n": len(dcms)
                    })
    return _dataset


def _get_networks() -> dict:
    global _networks
    if _networks is None:
        nets = {}
        # RED-CNN pretrained baseline
        nets["redcnn"] = hub_load_model("redcnn", eval=True).to(DEVICE)
        nets["variant_a"] = nets["redcnn"]

        # EDR-REDNet Variant D — use Seed42 by default
        ckpt_candidates = [
            os.path.join(ROOT, "results", "training", "EDR-REDCNN", "Seed42",   "best_SSIM_seed42.pt"),
            os.path.join(ROOT, "results", "training", "EDR-REDCNN", "Seed1339", "best_SSIM_seed1339.pt"),
            os.path.join(ROOT, "results", "training", "EDR-REDCNN", "Seed2024", "best_SSIM_seed2024.pt"),
            os.path.join(ROOT, "wandb", "edr_redcnn_seed42",  "files", "best_SSIM.pt"),
            os.path.join(ROOT, "wandb", "edr_redcnn_latest",  "files", "best_SSIM.pt"),
            os.path.join(ROOT, "wandb", "edr_redcnn",         "files", "best_SSIM.pt"),
        ]
        ckpt_d = next((p for p in ckpt_candidates if os.path.exists(p)), None)
        net_d = _load_edr_model(ckpt_d, use_sobel=True) if ckpt_d else None
        if net_d is None:
            print("[WARNING] EDR-REDNet model not found! Check results/training/EDR-REDCNN/")
        nets["edr_redcnn"] = net_d
        nets["variant_d"] = net_d

        # Variant B (+EdgeBlock, no Sobel)
        ckpt_b = next((p for p in [
            os.path.join(ROOT, "results", "training", "EDR-RedCnn", "VariantB", "Seed1339", "variantB_seed1339_best_SSIM.pt"),
            os.path.join(ROOT, "wandb", "edr_variant_b", "files", "best_SSIM.pt"),
        ] if os.path.exists(p)), None)
        nets["variant_b"] = _load_edr_model(ckpt_b, use_sobel=False) if ckpt_b else None

        # Variant C (+Sobel, no EdgeBlock)
        ckpt_c = next((p for p in [
            os.path.join(ROOT, "results", "training", "EDR-RedCnn", "VariantC", "Seed1339", "variantC_seed1339_best_SSIM.pt"),
            os.path.join(ROOT, "wandb", "edr_variant_c", "files", "best_SSIM.pt"),
        ] if os.path.exists(p)), None)
        nets["variant_c"] = _load_edr_model(ckpt_c, use_sobel=True) if ckpt_c else None

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


# ── /api/upload_folder ──────────────────────────────────────────────────────────
import uuid

@app.post("/api/upload_folder")
async def api_upload_folder(files: List[UploadFile] = File(...)):
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    session_dir = os.path.join(ROOT, "temp_uploads", session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    saved_files = []
    for f in files:
        if f.filename.lower().endswith(".dcm"):
            path = os.path.join(session_dir, os.path.basename(f.filename))
            with open(path, "wb") as out:
                content = await f.read()
                out.write(content)
            saved_files.append(path)
            
    if not saved_files:
        raise HTTPException(status_code=400, detail="Không tìm thấy file .dcm nào.")
        
    return {"session_id": session_id, "total_slices": len(saved_files)}

# ── /api/upload_slice ─────────────────────────────────────────────────────────
@app.get("/api/upload_slice")
def api_upload_slice(
    session_id: str,
    slice_idx: int = Query(0),
    hu_min: float = Query(-160),
    hu_max: float = Query(245)
):
    session_dir = os.path.join(ROOT, "temp_uploads", session_id)
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Upload session not found")
        
    dcm_files = sorted([os.path.join(session_dir, f) for f in os.listdir(session_dir) if f.endswith(".dcm")])
    if slice_idx >= len(dcm_files):
        slice_idx = 0
        
    ds = _get_dataset()
    nets = _get_networks()
    
    dcm = pydicom.dcmread(dcm_files[slice_idx])
    slope = float(getattr(dcm, "RescaleSlope", 1))
    intercept = float(getattr(dcm, "RescaleIntercept", 0))
    hu = dcm.pixel_array.astype("float32") * slope + intercept
    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        hu = np.max(hu) - hu
        
    if hu.shape != (512, 512):
        import cv2
        hu = cv2.resize(hu, (512, 512))
        
    x_raw = hu + 1024.0
    x_norm = ds._normalize(x_raw)
    x_t = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_redcnn = nets["redcnn"](x_t)
        pred_edr    = nets["edr_redcnn"](x_t)
        
    img_ld     = _to_hu(x_t, ds)
    img_redcnn = _to_hu(pred_redcnn, ds)
    img_edr    = _to_hu(pred_edr, ds)
    
    return {
        "images": {
            "ldct":   _ndarray_to_b64(_window(img_ld,     hu_min, hu_max)),
            "redcnn": _ndarray_to_b64(_window(img_redcnn, hu_min, hu_max)),
            "edr":    _ndarray_to_b64(_window(img_edr,    hu_min, hu_max)),
            "ndct":   None  # No NDCT available in upload mode
        },
        "metrics": {
            "redcnn": None, "edr": None
        }
    }


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
        "A — RED-CNN": "variant_a",
        "B — EB-REDCNN": "variant_b",
        "C — SI-REDCNN": "variant_c",
        "D — EDR-REDNet": "variant_d",
    }
    COLORS = {"A — RED-CNN": "#888888", "B — EB-REDCNN": "#4e9af1",
              "C — SI-REDCNN": "#f1a74e", "D — EDR-REDNet": "#2ecc71"}

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

    VARIANTS = {"LDCT": None, "A — RED-CNN": "variant_a",
                "B — EB-REDCNN": "variant_b", "C — SI-REDCNN": "variant_c",
                "D — EDR-REDNet": "variant_d", "NDCT (GT)": "ndct"}

    full_imgs, zoom_imgs, diff_imgs, sobel_imgs = {}, {}, {}, {}
    img_ndct = _to_hu(y_t, ds)
    x1, y1 = x, y
    x2, y2 = min(x + w, 512), min(y + h, 512)

    with torch.no_grad():
        all_imgs = {"LDCT": _to_hu(x_t, ds)}
        for key in ["variant_a", "variant_b", "variant_c", "variant_d"]:
            net = nets.get(key)
            short = {"variant_a": "A — RED-CNN", "variant_b": "B — EB-REDCNN",
                     "variant_c": "C — SI-REDCNN",   "variant_d": "D — EDR-REDNet"}[key]
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
        "B — EB-REDCNN": ("B_PSNR", "B_SSIM", "B_Edge_SSIM"),
        "C — SI-REDCNN": ("C_PSNR", "C_SSIM", "C_Edge_SSIM"),
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


# ── /api/predict_sybil ────────────────────────────────────────────────────────
@app.get("/api/predict_sybil")
def api_predict_sybil(
    pat_idx: int = Query(0),
    session_id: str = Query(None)
):
    """Run denoising + Sybil prediction with real-time progress via SSE."""
    ds = _get_dataset()
    nets = _get_networks()

    if session_id:
        ldct_dir = os.path.join(ROOT, "temp_uploads", session_id)
        if not os.path.exists(ldct_dir):
            return {"error": "Session not found"}
        patient_id = f"Upload_{session_id}"
        total_slices = len([f for f in os.listdir(ldct_dir) if f.endswith(".dcm")])
    else:
        s = ds.samples[pat_idx]
        ldct_dir = os.path.dirname(s["in_files"][0])
        patient_id = s["info"]["id"]
        total_slices = len(s["in_files"])

    # Temp dirs for denoised DICOMs
    temp_base = os.path.join(ROOT, "temp_sybil")
    redcnn_dir = os.path.join(temp_base, f"{patient_id}_redcnn")
    edr_dir = os.path.join(temp_base, f"{patient_id}_edr")

    def event_stream():
        from dicom_utils import denoise_dicom_folder_stream, run_sybil_prediction

        # ── Stage 1: Denoise with RED-CNN (0% → 33%) ──
        yield _sse({"stage": "redcnn_start", "message": f"Đang khử nhiễu RED-CNN: 0/{total_slices} slices...", "percent": 0})
        for current, total, fname in denoise_dicom_folder_stream(ldct_dir, redcnn_dir, nets["redcnn"], DEVICE):
            pct = round((current / total) * 33, 1)
            yield _sse({"stage": "redcnn", "current": current, "total": total, "percent": pct,
                        "message": f"Đang khử nhiễu RED-CNN: {current}/{total} slices..."})

        # ── Stage 2: Denoise with EDR-REDNet (33% → 66%) ──
        yield _sse({"stage": "edr_start", "message": f"Đang khử nhiễu EDR-REDNet: 0/{total_slices} slices...", "percent": 33})
        for current, total, fname in denoise_dicom_folder_stream(ldct_dir, edr_dir, nets["edr_redcnn"], DEVICE):
            pct = round(33 + (current / total) * 33, 1)
            yield _sse({"stage": "edr", "current": current, "total": total, "percent": pct,
                        "message": f"Đang khử nhiễu EDR-REDNet: {current}/{total} slices..."})

        # ── Stage 3: Sybil on LDCT (66% → 77%) ──
        yield _sse({"stage": "sybil_ldct", "message": "Sybil đang dự đoán trên LDCT gốc...", "percent": 66})
        try:
            scores_ldct = run_sybil_prediction(ldct_dir)
        except Exception as e:
            scores_ldct = [f"Error: {str(e)}"]

        # ── Stage 4: Sybil on RED-CNN (77% → 88%) ──
        yield _sse({"stage": "sybil_redcnn", "message": "Sybil đang dự đoán trên RED-CNN...", "percent": 77})
        try:
            scores_redcnn = run_sybil_prediction(redcnn_dir)
        except Exception as e:
            scores_redcnn = [f"Error: {str(e)}"]

        # ── Stage 5: Sybil on EDR-REDNet (88% → 100%) ──
        yield _sse({"stage": "sybil_edr", "message": "Sybil đang dự đoán trên EDR-REDNet...", "percent": 88})
        try:
            scores_edr = run_sybil_prediction(edr_dir)
        except Exception as e:
            scores_edr = [f"Error: {str(e)}"]

        # ── Done ──
        import datetime
        results = {
            "patient_id": patient_id,
            "total_slices": total_slices,
            "scores_ldct": scores_ldct,
            "scores_redcnn": scores_redcnn,
            "scores_edr": scores_edr,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to history
        try:
            history_file = os.path.join(ROOT, "sybil_history.json")
            history = []
            if os.path.exists(history_file):
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            history.insert(0, results)
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save history: {e}")

        yield _sse({
            "stage": "done",
            "percent": 100,
            "message": "Hoàn tất!",
            "results": results
        })

        # Clean up temp dirs
        try:
            shutil.rmtree(redcnn_dir, ignore_errors=True)
            shutil.rmtree(edr_dir, ignore_errors=True)
        except Exception:
            pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(data: dict) -> str:
    """Format a dict as an SSE event string."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

# ── /api/sybil_history ────────────────────────────────────────────────────────
@app.get("/api/sybil_history")
def api_sybil_history():
    history_file = os.path.join(ROOT, "sybil_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []
