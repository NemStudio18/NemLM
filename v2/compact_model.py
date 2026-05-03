import sqlite3
import pickle
import numpy as np
import os

db_path = 'v2/memory.nemdb'
out_path = 'v2/memory_compact.nemdb'

if not os.path.exists(db_path):
    print("Source non trouvée.")
    exit()

print(f"Ouverture de {db_path}...")
conn = sqlite3.connect(db_path)
out_conn = sqlite3.connect(out_path)
out_conn.execute("CREATE TABLE storage (key BLOB PRIMARY KEY, data BLOB)")

cursor = conn.execute("SELECT key, data FROM storage")
count = 0
saved = 0

print("Compression en cours (int16 -> bit)...")
for key, data in cursor:
    count += 1
    entry = pickle.loads(data)
    
    # Pruning : on ne garde que si on a vu au moins 2 fois (filtre le bruit)
    total_freq = sum(entry.token_weights.values())
    if total_freq < 2:
        continue
        
    # Compression HDC : Majority Vote -> Packed Bits
    # On ne stocke plus le weighted_sum int16, mais juste le résultat final packé
    bits = (entry.weighted_sum > 0).astype(np.uint8)
    packed_sum = np.packbits(bits)
    
    # On stocke une version ultra-légère : (packed_sum, token_weights)
    compact_data = pickle.dumps((packed_sum, entry.token_weights))
    out_conn.execute("INSERT INTO storage VALUES (?, ?)", (key, compact_data))
    
    saved += 1
    if count % 10000 == 0:
        print(f"Traité: {count} | Gardés: {saved}")
        out_conn.commit()

out_conn.commit()
out_conn.close()
conn.close()

print(f"\nTerminé ! {saved} entrées sauvegardées sur {count}.")
print(f"Taille finale estimée : {os.path.getsize(out_path) / 1024**2:.2f} Mo")
