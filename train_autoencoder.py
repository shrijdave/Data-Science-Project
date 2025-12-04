import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

# -----------------------------
# SETTINGS
# -----------------------------
DATASET_ROOT = "dataset"
SAVE_PATH = "ae_model.pth"

NUM_IMAGES = 1000          # train on 500–1000 images
EPOCHS = 15
BATCH_SIZE = 16
BOTTLENECK = 128           # same bottleneck as backend

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def load_random_images():
    """Load 1000 random image paths from dataset/"""
    all_imgs = [f for f in os.listdir(DATASET_ROOT) if f.endswith(".jpg")]
    random.shuffle(all_imgs)
    chosen = all_imgs[:NUM_IMAGES]
    return [os.path.join(DATASET_ROOT, img) for img in chosen]

# -----------------------------
# AUTOENCODER MODEL
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, bottleneck=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*128*3, 512),
            nn.ReLU(),
            nn.Linear(512, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 512),
            nn.ReLU(),
            nn.Linear(512, 128*128*3),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(-1, 3, 128, 128)
        return out

# -----------------------------
# LOAD IMAGES
# -----------------------------
print("Loading images...")
image_paths = load_random_images()
print(f"Training on {len(image_paths)} random images.")

images = []
for path in tqdm(image_paths):
    try:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        images.append(img)
    except:
        pass

dataset = torch.stack(images)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = Autoencoder(bottleneck=BOTTLENECK).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("Training...")

for epoch in range(EPOCHS):
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        recon = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg:.6f}")

# -----------------------------
# SAVE
# -----------------------------
torch.save(model.state_dict(), SAVE_PATH)
print(f"Training complete! Saved autoencoder to: {SAVE_PATH}")
