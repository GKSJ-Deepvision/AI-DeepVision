import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

# CSRNET MODEL
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        vgg = models.vgg16(pretrained=True)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )

        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

# DATASET (NEW PREPROCESSED DATA)
MAX_SAMPLES = 100   # reduce if system is slow

class ShanghaiPreproDataset(Dataset):
    def __init__(self, prepro_dir):
        self.prepro_dir = prepro_dir
        self.files = sorted([
            f for f in os.listdir(prepro_dir) if f.endswith(".pt")
        ])[:MAX_SAMPLES]

        print(f"Using {len(self.files)} samples from {prepro_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.prepro_dir, self.files[idx])
        data = torch.load(path, map_location="cpu")

        img = data["image"].float()
        gt  = data["gt"].float()

        return img, gt
   

# TRAINING LOOP

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # UPDATE IF YOUR PATH IS DIFFERENT
    prepro_dir = r"archive (2)/ShanghaiTech/preprocessed/part_B/train_data"

    dataset = ShanghaiPreproDataset(prepro_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,     # WINDOWS SAFE
        pin_memory=False
    )

    model = CSRNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for img, gt in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            img = img.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            pred = model(img)

            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Batch Loss 1 = {avg_loss:.6f}")

    # SAVE MODEL AS REQUESTED
    torch.save(model.state_dict(), "model_4.pth")
    print("\n Training completed. Model saved as model_4.pth")

# RUN
if __name__ == "__main__":
    train()