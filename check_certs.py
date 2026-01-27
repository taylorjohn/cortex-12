import json
import os

cert_dir = "certs/phase3"
for filename in os.listdir(cert_dir):
    if filename.endswith("_cert.json"):
        path = os.path.join(cert_dir, filename)
        with open(path) as f:
            cert = json.load(f)
        dims = cert["embedding_dims"]
        print(f"{filename}: dims = {dims}")