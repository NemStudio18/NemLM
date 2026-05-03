import urllib.request
import os

urls = {
    "les_miserables.txt": "https://www.gutenberg.org/files/135/135-0.txt",
    "monte_cristo.txt": "https://www.gutenberg.org/files/17989/17989-0.txt",
    "germinal.txt": "https://www.gutenberg.org/files/5711/5711-0.txt",
    "bel_ami.txt": "https://www.gutenberg.org/files/14358/14358-0.txt"
}

os.makedirs("corpus_fr", exist_ok=True)

print("Téléchargement du corpus français...")
for name, url in urls.items():
    path = os.path.join("corpus_fr", name)
    if not os.path.exists(path):
        print(f"-> {name}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, path)
            print("OK")
        except Exception as e:
            print(f"Erreur: {e}")

# Concaténation en un seul fichier propre
with open("french_corpus.txt", "w", encoding="utf-8") as outfile:
    for name in urls.keys():
        path = os.path.join("corpus_fr", name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")

print("\nCorpus français prêt : french_corpus.txt")
print(f"Taille totale : {os.path.getsize('french_corpus.txt') / 1024 / 1024:.1f} Mo")
