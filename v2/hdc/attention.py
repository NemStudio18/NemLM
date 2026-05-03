import numpy as np
from hdc.representation import DIM

# Table de lookup pour le comptage de bits (Popcount)
POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

def hamming_batch(query_hv, keys_matrix):
    """Calcule la distance de Hamming ultra-rapide via table de lookup."""
    # query_hv: (DIM/8,) uint8, keys_matrix: (N, DIM/8) uint8
    xor_res = np.bitwise_xor(keys_matrix, query_hv)
    # On utilise la table de lookup sur le dernier axe et on somme
    return POPCOUNT_TABLE[xor_res].sum(axis=1)

def majority_vote(hvs):
    """Fusionne plusieurs HV par vote majoritaire."""
    # hvs: liste de (DIM/8,) uint8
    if not hvs: return np.zeros(DIM // 8, dtype=np.uint8)
    
    # On unpack pour compter les bits
    unpacked = np.unpackbits(np.array(hvs), axis=1)
    sums = unpacked.sum(axis=0)
    # Si sum > len/2, le bit est a 1
    res_bits = (sums > (len(hvs) / 2)).astype(np.uint8)
    return np.packbits(res_bits)

class AttentionHead:
    def __init__(self, dim=DIM, n_keys=512):
        self.dim = dim
        self.n_keys = n_keys
        # On pre-alloue pour la vitesse
        self.keys = np.zeros((n_keys, dim // 8), dtype=np.uint8)
        self.values = np.zeros((n_keys, dim // 8), dtype=np.uint8)
        self.hits = np.zeros(n_keys, dtype=np.uint32) # Compteur d'utilité
        self.ptr = 0
        self.full = False

    def learn(self, context_hv, next_token_hv):
        """Ajoute un souvenir avec remplacement par utilité (Frequence-Weighted)."""
        if not self.full:
            self.keys[self.ptr] = context_hv
            self.values[self.ptr] = next_token_hv
            self.hits[self.ptr] = 1 # Premier hit à l'insertion
            self.ptr += 1
            if self.ptr >= self.n_keys:
                self.full = True
        else:
            # On remplace le souvenir le MOINS utilisé (Darwinisme HDC)
            idx = np.argmin(self.hits)
            self.keys[idx] = context_hv
            self.values[idx] = next_token_hv
            self.hits[idx] = 1 # Reset du compteur pour le nouveau

    def attend(self, query_hv, k=8):
        """Cherche les K plus proches et compose la reponse."""
        active_keys = self.keys if self.full else self.keys[:self.ptr]
        if len(active_keys) == 0:
            return np.zeros(self.dim // 8, dtype=np.uint8)
            
        distances = hamming_batch(query_hv, active_keys)
        # On prend les indices des K plus proches
        top_k_idx = np.argsort(distances)[:k]
        
        # On renforce les hits pour ces souvenirs (ils ont été utiles)
        self.hits[top_k_idx] += 1
        
        active_values = self.values if self.full else self.values[:self.ptr]
        candidates = [active_values[i] for i in top_k_idx]
        
        return majority_vote(candidates)

    def to_dict(self):
        return {
            "keys": self.keys,
            "values": self.values,
            "hits": self.hits,
            "ptr": self.ptr,
            "full": self.full
        }

    def from_dict(self, data):
        self.keys = data["keys"]
        self.values = data["values"]
        self.hits = data.get("hits", np.zeros(self.n_keys, dtype=np.uint32))
        self.ptr = data["ptr"]
        self.full = data["full"]

class MultiHeadBinaryAttention:
    def __init__(self, dim=DIM, n_heads=8, n_keys=512):
        self.dim = dim
        self.n_heads = n_heads
        self.n_keys = n_keys
        self.heads = [AttentionHead(dim, n_keys) for _ in range(n_heads)]
        # On fixe les projections de maniere deterministe
        self.projections = [np.random.RandomState(i).permutation(dim) for i in range(n_heads)]

    def save_to_db(self, conn):
        """Sauvegarde l'etat de toutes les tetes dans la DB SQLite."""
        import pickle
        for i, head in enumerate(self.heads):
            data = pickle.dumps(head.to_dict())
            conn.execute("INSERT OR REPLACE INTO storage (key, data) VALUES (?, ?)", 
                         (f"attn_head_{i}".encode(), data))
        conn.commit()

    def load_from_db(self, conn):
        """Charge l'etat de toutes les tetes depuis la DB SQLite."""
        import pickle
        total_keys = 0
        for i in range(self.n_heads):
            cursor = conn.execute("SELECT data FROM storage WHERE key = ?", (f"attn_head_{i}".encode(),))
            row = cursor.fetchone()
            if row:
                head_data = pickle.loads(row[0])
                self.heads[i].from_dict(head_data)
                total_keys += head_data["ptr"] if not head_data["full"] else self.n_keys
        if total_keys > 0:
            print(f"[OK] Attention chargee depuis le disque : {total_keys} souvenirs repartis sur {self.n_heads} tetes.")
        else:
            print("[!] Aucune attention trouvee sur le disque. Demarrage a zero.")

    def _project(self, hv, head_idx):
        """Projette le HV pour une tete specifique (permutation)."""
        bits = np.unpackbits(hv)
        projected = bits[self.projections[head_idx]]
        return np.packbits(projected)

    def learn(self, context_hv, next_token_hv):
        for i, head in enumerate(self.heads):
            # On projette le contexte differemment pour chaque tete
            proj_ctx = self._project(context_hv, i)
            head.learn(proj_ctx, next_token_hv)

    def forward(self, query_hv, k=8):
        outputs = []
        for i, head in enumerate(self.heads):
            proj_query = self._project(query_hv, i)
            outputs.append(head.attend(proj_query, k))
        
        return majority_vote(outputs)
