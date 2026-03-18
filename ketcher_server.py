"""
Local Ketcher server — serves Ketcher static files on localhost:8080
Run with: python ketcher_server.py
"""
import os
import urllib.request
import zipfile
from flask import Flask, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

KETCHER_DIR = os.path.join(os.path.dirname(__file__), "ketcher_static")

def download_ketcher():
    if os.path.exists(os.path.join(KETCHER_DIR, "index.html")):
        return
    print("Downloading Ketcher...")
    os.makedirs(KETCHER_DIR, exist_ok=True)
    url = "https://github.com/epam/ketcher/releases/download/v2.26.0/ketcher-standalone-v2.26.0.zip"
    zip_path = os.path.join(KETCHER_DIR, "ketcher.zip")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(KETCHER_DIR)
    os.remove(zip_path)
    print("Ketcher ready.")

@app.route("/")
@app.route("/<path:filename>")
def serve(filename="index.html"):
    return send_from_directory(KETCHER_DIR, filename)

if __name__ == "__main__":
    download_ketcher()
    app.run(port=8080, debug=False)