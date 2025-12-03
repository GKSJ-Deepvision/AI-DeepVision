import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os


#                   CSRNET MODEL
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        # ---- FRONTEND (VGG-16 BN) ----
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33])

        # ---- BACKEND (Dilated CNN) ----
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,  3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )

        # ---- OUTPUT LAYER ----
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

#      DATASET FOR LOADING PREPROCESSED .pt FILES
class CrowdPreprocessedDataset(Dataset):
    def __init__(self, prepro_path, limit=None):
        self.files = sorted([f for f in os.listdir(prepro_path) if f.endswith(".pt")])

        # OPTIONAL: limit dataset size for weak PCs
        if limit:
            self.files = self.files[:limit]

        self.prepro_path = prepro_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.prepro_path, self.files[idx]))
        img = data["image"].float()
        gt  = data["gt"].float()
        return img, gt


#                 TRAINING PREPARATION
# path to your preprocessed folder
prepro_dir = r"archive (2)/ShanghaiTech/part_B/train_data/images/images_prepro"

# dataset (limit=200 for safety)
dataset = CrowdPreprocessedDataset(prepro_dir, limit=200)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# model
model = CSRNet().to(device)

# loss + optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

batch_losses = []
epoch_losses = []


#                     TRAINING LOOP
epochs = 5

for epoch in range(epochs):
    epoch_loss = 0
    print(f"\n---- Epoch {epoch+1}/{epochs} ----")

    for img, gt in dataloader:
        img = img.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        pred = model(img)
        loss = criterion(pred, gt)

        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        epoch_loss += loss.item()

        print(f"Batch Loss: {loss.item():.4f}")

    epoch_loss /= len(dataloader)
    epoch_losses.append(epoch_loss)

    print(f">>> Epoch {epoch+1} Loss = {epoch_loss:.4f}")


#                      SAVE MODEL
torch.save(model.state_dict(), "csrnet_weights.pth")
print("\nModel saved as csrnet_weights.pth")
