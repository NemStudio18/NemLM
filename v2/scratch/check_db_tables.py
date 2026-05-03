import sqlite3
db_path = r"D:\nemlm_v5_3_compact_hf.nemdb"
conn = sqlite3.connect(db_path)
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables trouvées :", cursor.fetchall())
conn.close()
