import sqlite3
import os

db_path = 'v2/memory.nemdb'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT count(*) FROM storage").fetchone()[0]
    print(f"Nombre d'entrées en base : {count}")
    conn.close()
else:
    print("Base de données introuvable.")
