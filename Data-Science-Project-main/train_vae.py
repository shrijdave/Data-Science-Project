import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

DATASET_ROOT = "Data-Science-Project-main/dataset"
SAVE_PATH = "vae_model.pth"

NUM_IMAGES = 10000
EPOCHS = 50
BATCH_SIZE = 32
LATENT_DIM = 128

BETA = 1.0     # KL weight (try 0.1–1.0)
LR = 5e-4

SAMPLES_TO_SAVE = 64
SAMPLES_GRID_PATH = "vae_samples.png"
SAMPLES_DIR = "vae_samples"

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
    transforms.ToTensor(),  # [0,1]
])

# -----------------------------
# LOAD RANDOM IMAGES
# -----------------------------
def load_random_images():
    imgs = [f for f in os.listdir(DATASET_ROOT) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
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

if len(images) == 0:
    raise RuntimeError("No images loaded. Check DATASET_ROOT and file extensions.")

dataset = torch.stack(images)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# -----------------------------
# CONVOLUTIONAL VAE
# -----------------------------
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder: 128x128 -> 16x16 feature map
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16x16
            nn.ReLU(inplace=True),
        )
        self.enc_flat = nn.Flatten()
        self.enc_fc = nn.Linear(128 * 16 * 16, 512)

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 512)
        self.dec_fc2 = nn.Linear(512, 128 * 16 * 16)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 128x128
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        h = self.enc_flat(h)
        h = torch.relu(self.enc_fc(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.dec_fc1(z))
        h = torch.relu(self.dec_fc2(h))
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

@torch.no_grad()
def sample_images(model, n=64, out_grid_path="vae_samples.png", out_dir="vae_samples"):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    z = torch.randn(n, LATENT_DIM, device=device)
    samples = model.decode(z).clamp(0, 1)

    # Save a grid
    save_image(samples, out_grid_path, nrow=int(n ** 0.5))
    print(f"Saved sample grid -> {out_grid_path}")

    # Save individual images too
    for i in range(n):
        save_image(samples[i], os.path.join(out_dir, f"{i+1:04d}.png"))
    print(f"Saved {n} individual samples -> {out_dir}/")

def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = torch.mean((recon - x) ** 2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl

# -----------------------------
# TRAIN
# -----------------------------
model = ConvVAE(LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Training VAE...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_rec = 0.0
    total_kl = 0.0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        recon, mu, logvar, _ = model(batch)
        loss, rec, kl = vae_loss(recon, batch, mu, logvar, beta=BETA)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rec += rec.item()
        total_kl += kl.item()

    n_batches = len(loader)
    print(
        f"Epoch {epoch+1}/{EPOCHS}  "
        f"Loss: {total_loss/n_batches:.6f}  Recon: {total_rec/n_batches:.6f}  KL: {total_kl/n_batches:.6f}"
    )

    # optional: sample every 10 epochs
    if (epoch + 1) % 10 == 0:
        sample_images(model, n=SAMPLES_TO_SAVE, out_grid_path=SAMPLES_GRID_PATH, out_dir=SAMPLES_DIR)

# -----------------------------
# FINAL SAMPLE + SAVE MODEL
# -----------------------------
sample_images(model, n=SAMPLES_TO_SAVE, out_grid_path=SAMPLES_GRID_PATH, out_dir=SAMPLES_DIR)

torch.save(
    {
        "state_dict": model.state_dict(),
        "latent_dim": LATENT_DIM,
        "img_size": 128,
        "beta": BETA,
    },
    SAVE_PATH,
)
print(f"Saved model to {SAVE_PATH}")
