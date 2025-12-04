Dataset: https://www.kaggle.com/datasets/soumendraprasad/fifa-2022-all-players-image-dataset/data

If not already installed: pip install flask pillow numpy scikit-learn tqdm,  pip install flask-cors  , pip install torch torchvision torchaudio

In launch.JSON file change "file": to correct dashboard.html location

Replace the "python3" with "python" to run properly if you need to. Run in the terminal in order

1) python3 image_extractor.py  (extracts all 25k images to be under 1 folder)

2) python3 train_autoencoder.py  (it may take a minute to train) This will create a ae_model.pth (trained file)

3) python3 backend_pca.py 

In a new terminal:

4) python3 -m http.server 8080

Open Website:

http://localhost:8080/dashboard.html
