# inference_utils.py
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def load_model(model_class, checkpoint_path, device):
    """
    model_class: class (e.g. CSRNet)
    checkpoint_path: path to .pth
    device: torch.device
    """
    model = model_class().to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    # support checkpoints saved as state_dict or full dict
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # tolerate small mismatches
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def _to_tensor(img_bgr):
    """
    Convert OpenCV BGR uint8 image to torch.FloatTensor (1, C, H, W) normalized to [0,1].
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_f = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0)  # 1,C,H,W
    return tensor

def predict_density(model, img_bgr, device, out_size=None, scale=1.0):
    """
    Run CSRNet and return {"density": np.array(HxW), "pred_count": float}
    - img_bgr: OpenCV image (H,W,3) uint8
    - device: torch.device
    - out_size: (w,h) or (h,w) accepted by cv2.resize (if provided will resize density)
    - scale: multiplicative calibration factor for count
    """
    if img_bgr is None:
        return {"density": np.zeros((1,1), dtype=np.float32), "pred_count": 0.0}

    # convert and move to device
    tensor = _to_tensor(img_bgr).to(device)

    with torch.no_grad():
        out = model(tensor)  # model expects (1,C,H,W)
        # upsample to input image size
        H, W = img_bgr.shape[:2]
        den = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        den_np = den.squeeze().cpu().numpy()
        # ensure non-negative
        den_np = np.clip(den_np, 0, None).astype(np.float32)

    if out_size is not None:
        # out_size may be (w,h) from user; cv2.resize expects (w,h)
        try:
            den_resized = cv2.resize(den_np, out_size, interpolation=cv2.INTER_LINEAR)
        except Exception:
            # try swapped
            den_resized = cv2.resize(den_np, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)
        den_np = den_resized

    pred = float(den_np.sum() * float(scale))
    return {"density": den_np, "pred_count": pred}


def make_overlay(img_bgr, density, clip_pct=99.5, alpha=0.5):
    """
    Blend heatmap on top of BGR image.
    - density: 2D numpy array same H,W as img_bgr
    - clip_pct: percentile for clipping max (to avoid a single peak dominating colors)
    - alpha: blending factor for heatmap
    """
    h, w = img_bgr.shape[:2]
    if density is None or density.size == 0:
        return img_bgr.copy()

    # Normalize density to 0..1 using percentile clipping
    vmax = np.percentile(density, clip_pct) if density.size else 1.0
    vmax = float(vmax) if vmax > 0 else 1.0
    den_norm = np.clip(density / vmax, 0.0, 1.0)
    den_uint8 = (den_norm * 255).astype(np.uint8)

    heat = cv2.applyColorMap(den_uint8, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)

    overlay = cv2.addWeighted(heat, alpha, img_bgr, 1.0 - alpha, 0)
    return overlay


def find_blob_peaks(density, peak_percentile=99.4, min_sep=10, blob_sigma=4):
    """
    Simple blob/peak finder:
    - threshold = percentile of density
    - connected components -> centroids
    - enforce min_sep by greedy filtering
    Returns list of (x,y) coordinates in image pixel coordinates.
    """
    if density is None or density.size == 0:
        return []

    # optional smoothing
    den_s = cv2.GaussianBlur(density, (0, 0), sigmaX=blob_sigma, sigmaY=blob_sigma)

    thresh = np.percentile(den_s, peak_percentile)
    if thresh <= 0:
        return []

    bw = (den_s >= thresh).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(bw)

    centers = []
    for lbl in range(1, num_labels):
        ys, xs = np.where(labels == lbl)
        if len(xs) == 0:
            continue
        cx = int(np.mean(xs)); cy = int(np.mean(ys))
        centers.append((cx, cy))

    # enforce min separation (greedy)
    centers_sorted = sorted(centers, key=lambda p: -density[p[1], p[0]] if 0 <= p[1] < density.shape[0] and 0 <= p[0] < density.shape[1] else 0)
    selected = []
    for c in centers_sorted:
        too_close = False
        for s in selected:
            if (c[0]-s[0])**2 + (c[1]-s[1])**2 < (min_sep**2):
                too_close = True
                break
        if not too_close:
            selected.append(c)
    return selected


def draw_peaks(img_bgr, peaks, color=(0,255,0), radius=4):
    out = img_bgr.copy()
    for (x, y) in peaks:
        cv2.circle(out, (int(x), int(y)), radius, color, -1)
    return out


def put_text(img_bgr, text, org=(10, 34), scale=1.0, color=(0,0,255), thickness=2):
    out = img_bgr.copy()
    cv2.putText(out, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, float(scale), color, thickness, cv2.LINE_AA)
    return out
