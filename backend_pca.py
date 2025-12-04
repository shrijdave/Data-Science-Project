import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import base64, io
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as TF

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

IMG_SIZE = 256  # resize for PCA


# -------------------------
# Helpers
# -------------------------
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


# -------------------------
# PCA API
# -------------------------
@app.route("/pca", methods=["POST"])
def run_pca():
    data = request.json
    img_b64 = data["image"]
    K = int(data["components"])

    # Load and resize
    img = b64_to_pil(img_b64)
    arr = img_to_np(img)  # shape: (128, 128, 3)

    reconstructed = np.zeros_like(arr)

    # Apply PCA per channel
    for c in range(3):
        channel = arr[:, :, c]

        # Fit PCA on the transpose (so samples = 128)
        pca = PCA(n_components=min(K, 256))
        transformed = pca.fit_transform(channel)
        restored = pca.inverse_transform(transformed)

        reconstructed[:, :, c] = restored

    # MSE
    mse = float(mean_squared_error(arr.reshape(-1), reconstructed.reshape(-1)))

    # Convert back to image
    rec_img = np_to_pil(reconstructed)
    rec_b64 = pil_to_b64(rec_img)

    return jsonify({
        "reconstruction": rec_b64,
        "error": mse,
        "components_used": min(K, 256)
    })
    
# -------------------------------
# Autoencoder Model
# -------------------------------
class Autoencoder(nn.Module):
    def __init__(self, bottleneck=64):
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
        return out, z
    
    


# -------------------------------
# Load pretrained Autoencoder
# -------------------------------
device = "cpu"

ae_model = Autoencoder(bottleneck=128).to(device)
ae_model.load_state_dict(torch.load("ae_model.pth", map_location=device))
ae_model.eval()


# -------------------------------
# Helpers
# -------------------------------
def decode_image(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((128, 128))
    return TF.ToTensor()(img).unsqueeze(0)

def encode_image(tensor_img):
    arr = tensor_img.squeeze().permute(1,2,0).detach().numpy() * 255
    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# -------------------------------
# AUTOENCODER API ENDPOINT
# -------------------------------
@app.route("/ae", methods=["POST"])
def run_autoencoder():
    global ae_model

    data = request.json
    bottleneck = int(data["bottleneck"])
    base64_img = data["image"]

    img = decode_image(base64_img).to(device)

    # Run through pretrained autoencoder
    output, z = ae_model(img)

    mse = torch.mean((img - output) ** 2).item()

    return jsonify({
        "image": encode_image(output),
        "error": round(mse, 6)
    })




# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    print("PCA backend ready at http://127.0.0.1:5000")
    app.run(debug=True)
