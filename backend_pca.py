import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import base64, io, math
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as TF
from torchvision.utils import make_grid


# Flask setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

IMG_SIZE = 256
AE_IMG_SIZE = 128
VAE_IMG_SIZE = 128


# -----------------------------
# Image helpers
# -----------------------------
def b64_to_pil(b64_data):
    return Image.open(
        io.BytesIO(base64.b64decode(b64_data))
    ).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

def img_to_np(img):
    return np.asarray(img).astype("float32") / 255.0

def np_to_pil(arr):
    arr = (np.clip(arr, 0, 1) * 255).astype("uint8")
    return Image.fromarray(arr)

def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# -----------------------------
# PCA API
# -----------------------------
@app.route("/pca", methods=["POST"])
def run_pca():
    data = request.json
    img_b64 = data["image"]
    K = int(data["components"])

    img = b64_to_pil(img_b64)
    arr = img_to_np(img)

    reconstructed = np.zeros_like(arr)

    for c in range(3):
        channel = arr[:, :, c]
        pca = PCA(n_components=min(K, IMG_SIZE))
        reduced = pca.fit_transform(channel)
        restored = pca.inverse_transform(reduced)
        reconstructed[:, :, c] = restored

    mse = float(mean_squared_error(arr.reshape(-1), reconstructed.reshape(-1)))
    rec_img = np_to_pil(reconstructed)

    return jsonify({
        "reconstruction": pil_to_b64(rec_img),
        "error": mse
    })


# -----------------------------
# AUTOENCODER
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, bottleneck=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


device = torch.device("cpu")

ae_model = Autoencoder(128).to(device)
ae_model.load_state_dict(torch.load("ae_model.pth", map_location=device))
ae_model.eval()


def decode_image(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert("RGB")
    img = img.resize((AE_IMG_SIZE, AE_IMG_SIZE))
    return TF.ToTensor()(img).unsqueeze(0)

def encode_image(tensor_img):
    arr = tensor_img.squeeze().permute(1,2,0).detach().numpy()
    arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/ae", methods=["POST"])
def run_autoencoder():
    data = request.json
    bottleneck = int(data["bottleneck"])
    img = decode_image(data["image"]).to(device)

    with torch.no_grad():
        _, z = ae_model(img)
        z[:, bottleneck:] = 0
        output = ae_model.decoder(z)

    mse = torch.mean((img - output) ** 2).item()

    return jsonify({
        "image": encode_image(output),
        "error": round(mse, 6)
    })


# -----------------------------
# VAE (GENERATION ONLY)
# -----------------------------
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # Encoder (EXACT MATCH)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc_flat = nn.Flatten()
        self.enc_fc = nn.Linear(128 * 16 * 16, 512)

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder (EXACT MATCH)
        self.dec_fc1 = nn.Linear(latent_dim, 512)
        self.dec_fc2 = nn.Linear(512, 128 * 16 * 16)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def decode(self, z):
        h = torch.relu(self.dec_fc1(z))
        h = torch.relu(self.dec_fc2(h))
        return self.dec(h)



vae_ckpt = torch.load("vae_model.pth", map_location=device)
vae_model = ConvVAE(latent_dim=vae_ckpt["latent_dim"]).to(device)
vae_model.load_state_dict(vae_ckpt["state_dict"])
vae_model.eval()


@app.route("/vae_generate", methods=["POST"])
def vae_generate():
    n = int(request.json.get("n", 64))

    with torch.no_grad():
        z = torch.randn(n, vae_ckpt["latent_dim"])
        samples = vae_model.decode(z)
        grid = make_grid(samples, nrow=int(math.sqrt(n)))

    grid = grid.permute(1,2,0).numpy()
    grid = (grid * 255).astype(np.uint8)
    img = Image.fromarray(grid)

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return jsonify({
        "grid": base64.b64encode(buf.getvalue()).decode()
    })


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    print("Backend ready at http://127.0.0.1:5000")
    app.run(debug=True)

