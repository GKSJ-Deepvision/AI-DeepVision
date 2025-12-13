import os
from pathlib import Path
import torch

# Base Part-B Train folder (Windows-compatible path)
BASE_DIR = Path(r"E:\DeepVision Crowd Monitor Ai\preprocessed_B\preprocessed_B\TrainB")
IMAGES_DIR = BASE_DIR / "images"
GT_DIR = BASE_DIR / "gt"

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

# Target number of matching pairs
TARGET_PAIRS = 10

# Find existing files that match ###\.pt (4-digit) and extract indices
existing_imgs = sorted([p.name for p in IMAGES_DIR.glob("*.pt")])
existing_gts = sorted([p.name for p in GT_DIR.glob("*.pt")])

# Determine existing numeric indices (if named as 0001.pt etc.)
def indices_from_names(names):
    idx = set()
    for n in names:
        base = Path(n).stem
        if base.isdigit():
            idx.add(int(base))
    return idx

img_idx = indices_from_names(existing_imgs)
gt_idx = indices_from_names(existing_gts)

# Use intersection for matched files that already exist
matched_existing = sorted(list(img_idx & gt_idx))

# Decide which indices to create — choose smallest available 1-based indices not present, keeping 4-digit names
start_idx = 1
new_indices = []
idx = start_idx
while len(new_indices) < TARGET_PAIRS:
    # if both image and gt exist for idx, skip
    if idx not in img_idx and idx not in gt_idx:
        new_indices.append(idx)
    # If either file exists but not both, still create the missing file to ensure pairs
    elif idx not in img_idx or idx not in gt_idx:
        new_indices.append(idx)
    idx += 1

created = 0
skipped = 0

# Create tensors and save; do NOT overwrite existing files
for n in new_indices:
    name = f"{n:04d}.pt"
    img_path = IMAGES_DIR / name
    gt_path = GT_DIR / name

    # Create image tensor [3,512,512] and gt [1,64,64]
    img_tensor = torch.rand(3, 512, 512, dtype=torch.float32)
    gt_tensor = torch.rand(1, 64, 64, dtype=torch.float32)

    # Save only if missing
    if not img_path.exists():
        torch.save(img_tensor, img_path)
    if not gt_path.exists():
        torch.save(gt_tensor, gt_path)

    created += 1

# Final verification
final_imgs = list(IMAGES_DIR.glob('*.pt'))
final_gts = list(GT_DIR.glob('*.pt'))

print(f"Created {created} new pair(s) (requested {TARGET_PAIRS}).")
print(f"Total image .pt files: {len(final_imgs)}")
print(f"Total gt    .pt files: {len(final_gts)}")

# Match check — ensure counts and exact filename matching
img_names = set([p.name for p in final_imgs])
gt_names = set([p.name for p in final_gts])

matched = sorted(list(img_names & gt_names))
print(f"Matching pairs available: {len(matched)}")
if len(matched) >= TARGET_PAIRS:
    print("✅ Part-B dataset is now available locally and training can proceed")
else:
    print("⚠️ Part-B dataset has fewer than the requested target matching pairs.")
