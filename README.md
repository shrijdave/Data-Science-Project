Replace python3 with python if you need to. Run in the terminal in order

1) python3 train_autoencoder.py  (it may take a minute to train) This will create a ae_model.pth (trained file)

2) python3 backend_pca.py 

In a new terminal:

3) python3 -m http.server 8080

Open Website:

http://localhost:8080/dashboard.html
