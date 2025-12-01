import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os  # Import the os module

from src.dataset import CrowdDataset
from src.csrnet import CSRNet

# Set device to CUDA (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model():

    # -------------------------------
    # 1. Load Dataset
    # -------------------------------
    train_dataset = CrowdDataset(
        img_dir="data/ShanghaiTech/part_A/train_data/images",
        dmap_dir="data/ShanghaiTech/part_A/train_data/ground-truth"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,  # Use 4 workers for GPU
        pin_memory=True # Set to True for GPU
    )

    # -------------------------------
    # 2. Load Model
    # -------------------------------
    
    # --- CHANGES HERE ---
    model = CSRNet().cuda()
    criterion = nn.MSELoss().cuda()
    # --------------------

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # -------------------------------
    # 3. Training Loop
    # -------------------------------
    epochs = 50

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = "checkpoints" 
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Saving checkpoints to: {checkpoint_dir}")


    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for imgs, dmaps in pbar:
            
            # --- CHANGES HERE ---
            imgs = imgs.cuda()
            dmaps = dmaps.cuda()
            # --------------------

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, dmaps)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch+1}, Loss = {epoch_loss / len(train_loader)}")

        # -------------------------------
        # 4. Save checkpoint each epoch
        # -------------------------------
        # Saving inside the new 'checkpoints' directory
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"csrnet_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    train_model()
    