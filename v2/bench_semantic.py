"""
Benchmark Sémantique HDC V2 - Scaling sur 200k de caractères text8.
"""

import time
import numpy as np
from hdc.semantic import SemanticIndex

def main():
    # 1. Préparation des données
    print("Chargement des données (text8 subset 200k)...")
    try:
        with open("../v1/text8", "r") as f:
            content = f.read(200000) # 200k chars
    except:
        content = "Ceci est un document de test." * 5000

    # Split
    import re
    words = content.split()
    # On groupe par 10 mots pour faire des phrases
    sentences = [" ".join(words[i:i+10]) for i in range(0, len(words), 10)]
    
    print(f"Nombre de phrases à indexer : {len(sentences)}")
    
    # 2. Bench Indexation
    dim = 30000
    index = SemanticIndex(dim=dim)
    
    t0 = time.perf_counter()
    index.build(sentences)
    t1 = time.perf_counter()
    print(f"Temps d'indexation : {t1 - t0:.2f} s")
    
    # 3. Bench Requêtes
    queries = [
        "american history and geography",
        "scientific discovery in physics",
        "the computer architecture and software",
        "military strategy in the middle ages",
        "linguistics and language family"
    ]
    
    print(f"Lancement de {len(queries)} requêtes de test...")
    
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        results = index.query(q, k=3)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        
    avg_latency = np.mean(latencies) * 1000
    print(f"Latence moyenne par requête : {avg_latency:.2f} ms")
    
    print("\n--- Top Résultats pour la dernière requête ---")
    for r in results:
        print(f"  - {r[0]} (dist: {r[1]})")

    # 4. Rapport de mémoire
    mem_mb = (len(sentences) * dim) / (8 * 1024 * 1024)
    print(f"\nEmpreinte mémoire estimée (HVs) : {mem_mb:.2f} MB")

if __name__ == "__main__":
    main()
