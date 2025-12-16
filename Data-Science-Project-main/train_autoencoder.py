import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm



DATASET_ROOT = "dataset"
SAVE_PATH = "ae_model.pth"

NUM_IMAGES = 10000        
EPOCHS = 50
BATCH_SIZE = 32
BOTTLENECK = 128         

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# IMAGE TRANSFORM

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# LOAD RANDOM IMAGES

def load_random_images():
    imgs = [f for f in os.listdir(DATASET_ROOT) if f.endswith(".jpg")]
    random.shuffle(imgs)
    imgs = imgs[:NUM_IMAGES]
    return [os.path.join(DATASET_ROOT, f) for f in imgs]

print("Loading images...")
paths = load_random_images()

images = []
for p in tqdm(paths):
    try:
        img = Image.open(p).convert("RGB")
        img = transform(img)
        images.append(img)
    except:
        pass

dataset = torch.stack(images)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# CONVOLUTIONAL AUTOENCODER

class Autoencoder(nn.Module):
    def __init__(self, bottleneck=128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64×64
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32×32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16×16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, bottleneck)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# TRAIN

model = Autoencoder(BOTTLENECK).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

print("Training autoencoder...")

for epoch in range(EPOCHS):
    total = 0
    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        recon, _ = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        optimizer.step()

        total += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total/len(loader):.6f}")

# SAVE

torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved model to {SAVE_PATH}")


