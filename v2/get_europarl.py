import urllib.request
import gzip
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = "https://object.pouta.csc.fi/OPUS-Europarl/v8/mono/fr.txt.gz"
gz_path = "europarl_fr.txt.gz"
txt_path = "europarl_fr.txt"

print("Téléchargement du corpus Europarl FR (80 Mo)...")
try:
    urllib.request.urlretrieve(url, gz_path)
    print("Décompression...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(txt_path, 'wb') as f_out:
            # On prend 80 Mo pour assurer > 200k phrases
            chunk = f_in.read(80 * 1024 * 1024)
            f_out.write(chunk)
    print(f"Prêt : {txt_path} ({os.path.getsize(txt_path) / 1024 / 1024:.1f} Mo)")
except Exception as e:
    print(f"Erreur: {e}")
finally:
    if os.path.exists(gz_path):
        os.remove(gz_path)
