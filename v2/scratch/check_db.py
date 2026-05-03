import sqlite3
conn = sqlite3.connect('D:/nemlm_v5_3_full.nemdb')
c = conn.cursor()
c.execute('SELECT key FROM storage WHERE key LIKE "attn_head_%"')
for row in c.fetchall():
    print(row[0])
conn.close()
