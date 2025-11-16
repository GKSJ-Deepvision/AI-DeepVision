#worked
#Takes one image as input and annotates and generated density map and overlays the image and density map and prints the count to terminal and shows the image  and saves the file in the image folder itself
#viz_one_shanghaitech_fixed.py
import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import gaussian_filter

def mat_for_image(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(img_path)), "ground-truth", f"GT_{base}.mat"),
        os.path.join(os.path.dirname(img_path), "..", "ground-truth", f"GT_{base}.mat"),
        os.path.join(os.path.dirname(img_path), "ground-truth", f"GT_{base}.mat"),
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.isfile(p):
            return p
    return None


def unwrap_object_array(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.flatten()[0]
    return x


def find_coords_in_mat(mat):
    # Try direct known keys
    for key in ('annPoints', 'points', 'gt', 'GT'):
        if key in mat:
            cand = unwrap_object_array(mat[key])
            if isinstance(cand, np.ndarray) and cand.ndim == 2 and cand.shape[1] == 2:
                return cand.astype(np.float32)

    # image_info structured array
    if 'image_info' in mat:
        info = unwrap_object_array(mat['image_info'])
        if hasattr(info, 'dtype') and info.dtype.names:
            if 'location' in info.dtype.names:
                loc = unwrap_object_array(info['location'])
                if isinstance(loc, np.ndarray) and loc.ndim == 2:
                    return loc.astype(np.float32)

        if isinstance(info, np.ndarray):
            for el in info.flatten():
                el_un = unwrap_object_array(el)
                if hasattr(el_un, 'dtype') and el_un.dtype.names and 'location' in el_un.dtype.names:
                    loc = unwrap_object_array(el_un['location'])
                    if isinstance(loc, np.ndarray) and loc.ndim == 2:
                        return loc.astype(np.float32)

    # fallback
    for k in mat.keys():
        if not k.startswith("__"):
            v = unwrap_object_array(mat[k])
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 2:
                return v.astype(np.float32)

    return None


def points_to_density(points, shape, sigma=15):
    H, W = shape
    den = np.zeros((H, W), dtype=np.float32)
    if points is None:
        return den

    for (x, y) in points:
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= iy < H and 0 <= ix < W:
            den[iy, ix] += 1.0

    den = gaussian_filter(den, sigma=sigma, mode='constant')
    return den


def visualize_and_save(image_path, out_path=None, sigma=12):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Cannot read image: " + image_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    mat_path = mat_for_image(image_path)
    points = None
    if mat_path:
        mat = sio.loadmat(mat_path, simplify_cells=False)
        points = find_coords_in_mat(mat)

    if points is None or points.size == 0:
        print("No coordinates found in the mat file.")
        plt.figure(figsize=(8,6))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.show()
        return

    den = points_to_density(points, (H, W), sigma=sigma)
    s = den.sum()

    # Print in your exact format
    print(f"Annotated count (from mat): {points.shape[0]}    Density sum (approx): {s}")

    # ---- Visualization ----
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(img_rgb)
    axs[0].scatter(points[:, 0], points[:, 1], c="red", s=6)
    axs[0].axis("off")
    axs[0].set_title("Image + Points")

    vmax = den.max() if den.max() > 0 else 1.0
    im = axs[1].imshow(den, cmap="jet", vmin=0, vmax=vmax)
    axs[1].axis("off")
    axs[1].set_title("Density Map")
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    heat = plt.get_cmap("jet")(den / vmax)[:, :, :3]
    overlay = 0.5 * (img_rgb / 255.0) + 0.5 * heat

    axs[2].imshow(overlay)
    axs[2].axis("off")
    axs[2].set_title(f"Overlay (sum={s:.1f})")

    plt.tight_layout()
    plt.show()

    # Save file
    if out_path is None:
        out_path = os.path.splitext(image_path)[0] + "_density_vis.png"

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved visualization to:", out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python viz_one_shanghaitech_fixed.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    visualize_and_save(image_path)
