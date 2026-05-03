import sqlite3
import pickle
import os
import time

# On importe MemoryEntry pour pouvoir désérialiser les objets pickle
from hdc.memory import MemoryEntry

def distill(source_db, dest_db, top_k=5):
    if not os.path.exists(source_db):
        print(f"[!] Erreur : Base source {source_db} introuvable.")
        return

    print(f"[*] Démarrage de la distillation : {source_db} -> {dest_db}")
    start_time = time.time()

    src_conn = sqlite3.connect(source_db)
    dest_conn = sqlite3.connect(dest_db)
    
    # Configuration SQLite pour la vitesse
    dest_conn.execute("PRAGMA journal_mode = OFF")
    dest_conn.execute("PRAGMA synchronous = OFF")
    
    dest_conn.execute("CREATE TABLE IF NOT EXISTS distilled (id BLOB PRIMARY KEY, preds BLOB)")
    
    cursor = src_conn.execute("SELECT key, data FROM storage")
    
    count = 0
    skipped = 0
    
    for hv_key, data in cursor:
        try:
            entry = pickle.loads(data)
            
            # Trier les tokens par poids décroissant
            if not entry.token_weights:
                skipped += 1
                continue
                
            sorted_tokens = sorted(entry.token_weights.items(), key=lambda x: x[1], reverse=True)
            top_tokens = [t[0] for t in sorted_tokens[:top_k]]
            
            # On stocke juste la liste des Top-K tokens
            dest_conn.execute("INSERT OR REPLACE INTO distilled VALUES (?, ?)", 
                             (hv_key, pickle.dumps(top_tokens)))
            
            count += 1
            if count % 50000 == 0:
                elapsed = time.time() - start_time
                print(f"  > Traité {count:7d} entrées... ({count/elapsed:.0f} entrées/s)")
                dest_conn.commit()
        except Exception as e:
            print(f"[!] Erreur sur une entrée : {e}")
            
    dest_conn.commit()
    
    print("[*] Optimisation finale (VACUUM)...")
    dest_conn.execute("VACUUM")
    
    src_conn.close()
    dest_conn.close()
    
    duration = time.time() - start_time
    print(f"[*] Distillation terminée en {duration:.1f}s.")
    print(f"[*] Entrées distillées : {count}")
    print(f"[*] Taille finale : {os.path.getsize(dest_db) / (1024*1024):.2f} Mo")

if __name__ == "__main__":
    SOURCE = r"D:\nemlm_duel_v5_3.nemdb"
    DEST = r"D:\nemlm_v5_3_compact.nemdb"
    distill(SOURCE, DEST, top_k=5)
