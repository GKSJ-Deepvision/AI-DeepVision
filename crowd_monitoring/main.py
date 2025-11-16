# main.py

from config import PART_A_TRAIN, IMG_SIZE, DENSITY_MODE, FIXED_SIGMA, BATCH_SIZE
from dataset import ShanghaiTechDataset
from torch.utils.data import DataLoader


def run_test():
    dataset = ShanghaiTechDataset(
        root_dir=PART_A_TRAIN,
        img_size=IMG_SIZE,
        density_mode=DENSITY_MODE,
        fixed_sigma=FIXED_SIGMA
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for imgs, dens in loader:
        print("Image batch:", imgs.shape)
        print("Density batch:", dens.shape)
        break


if __name__ == "__main__":
    run_test()
