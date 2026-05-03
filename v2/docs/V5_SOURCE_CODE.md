# Source Code - NemLM V5.1 Industrial

## v2/hdc/v3_engine.py

`python
import numpy as np
from hdc.representation import encode_context, DIM
from hdc.semantic import SemanticIndex
from hdc.memory import AssociativeMemory
from hdc.attention import MultiHeadBinaryAttention

class V3Engine:
    def __init__(self, dim: int = DIM, db_path: str = "v2/memory.nemdb"):
        self.dim = dim
        self.memory = AssociativeMemory(dim=dim, db_path=db_path)
        self.semantic = SemanticIndex(dim)
        self.attention = MultiHeadBinaryAttention(dim, n_heads=8, n_keys=1024)
        
        # Chargement automatique de l'attention si elle existe sur disque
        self.attention.load_from_db(self.memory.conn)
        
        self.long_term_hvs: list[np.ndarray] = []

    def commit(self):
        """Sauvegarde tout le moteur (Memoire + Attention)."""
        self.memory.commit()
        self.attention.save_to_db(self.memory.conn)

    def get_combined_hv(self, context_tokens: list[str]) -> np.ndarray:
        l_hv_packed = encode_context(context_tokens, self.dim)
        if not self.long_term_hvs:
            return l_hv_packed
        c_hv_packed = self.long_term_hvs[-1]
        return np.bitwise_xor(c_hv_packed, l_hv_packed)

    def train_step(self, sentence: list[str]):
        if len(sentence) < 2: return
        sent_hv_packed = self.semantic.encode_bow(sentence)

        for i in range(1, len(sentence)):
            context = sentence[max(0, i - 5):i]
            target = sentence[i]

            l_hv_packed = encode_context(context, self.dim)
            
            # Apprentissage Attention Binaire (Fallback)
            target_hv = self.semantic.get_word_hv(target)
            self.attention.learn(l_hv_packed, target_hv)
            
            # Apprentissage Associatif (Exact match)
            # On stocke le lien local -> target
            self.memory.learn_one_pass(l_hv_packed, target)

        self.long_term_hvs.append(sent_hv_packed)
        if len(self.long_term_hvs) > 100:
            self.long_term_hvs.pop(0)

    def predict_next(self, context_tokens: list[str], top_k: int = 5) -> list[str]:
        """Orchestre la prediction hybride : Exact Match -> Fallback Attention."""
        query_hv = encode_context(context_tokens, self.dim)
        
        # 1. Tentative Match Exact
        exact_preds = self.memory.predict_topk(query_hv, k=top_k)
        if exact_preds:
            return [p[0] for p in exact_preds]
            
        # 2. Fallback Attention Sémantique (Top-k)
        # On demande a l'attention de nous sortir un HV consensus
        attention_hv = self.attention.forward(query_hv, k=8)
        
        # On demande a l'index semantique les Top-K mots proches de ce consensus
        # (On modifie find_nearest pour supporter top_k)
        predicted_tokens = self.semantic.find_nearest_topk(attention_hv, k=top_k)
        return predicted_tokens

`

## v2/hdc/attention.py

`python
import numpy as np
from hdc.representation import DIM

def hamming_batch(query_hv, keys_matrix):
    """Calcule la distance de Hamming entre une requete et une matrice de cles."""
    # query_hv: (DIM/8,) uint8
    # keys_matrix: (N, DIM/8) uint8
    # XOR donne les bits differents
    xor_res = np.bitwise_xor(keys_matrix, query_hv)
    # On compte les bits a 1 (unpackbits est lent, on utilise une table de lookup ou sum)
    # Version simple et rapide avec NumPy
    return np.unpackbits(xor_res, axis=1).sum(axis=1)

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
        self.ptr = 0
        self.full = False

    def learn(self, context_hv, next_token_hv):
        """Ajoute un souvenir avec remplacement aleatoire si plein."""
        if not self.full:
            self.keys[self.ptr] = context_hv
            self.values[self.ptr] = next_token_hv
            self.ptr += 1
            if self.ptr >= self.n_keys:
                self.full = True
        else:
            # Remplacement aleatoire (Reservoir Sampling simple)
            idx = np.random.randint(0, self.n_keys)
            self.keys[idx] = context_hv
            self.values[idx] = next_token_hv

    def attend(self, query_hv, k=8):
        """Cherche les K plus proches et compose la reponse."""
        active_keys = self.keys if self.full else self.keys[:self.ptr]
        if len(active_keys) == 0:
            return np.zeros(self.dim // 8, dtype=np.uint8)
            
        distances = hamming_batch(query_hv, active_keys)
        # On prend les indices des K plus petites distances
        top_k_idx = np.argsort(distances)[:k]
        
        active_values = self.values if self.full else self.values[:self.ptr]
        candidates = [active_values[i] for i in top_k_idx]
        
        return majority_vote(candidates)

    def to_dict(self):
        return {
            "keys": self.keys,
            "values": self.values,
            "ptr": self.ptr,
            "full": self.full
        }

    def from_dict(self, data):
        self.keys = data["keys"]
        self.values = data["values"]
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

`

## v2/hdc/semantic.py

`python
"""
HDC Semantic Layer V3 (Bit-Packed)
"""
import numpy as np
from .representation import encode_token, rotate, DIM

class SemanticIndex:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.vocabulary = {} # token -> hv_packed
        self.id_to_token = {}
        self.token_hvs = [] # List of packed HVs for search

    def get_word_hv(self, token: str) -> np.ndarray:
        if token not in self.vocabulary:
            hv = encode_token(token, self.dim)
            idx = len(self.token_hvs)
            self.vocabulary[token] = hv
            self.id_to_token[idx] = token
            self.token_hvs.append(hv)
        return self.vocabulary[token]

    def find_nearest_topk(self, query_hv_packed: np.ndarray, k: int = 5) -> list[str]:
        if not self.token_hvs: return ["???"]
        
        hvs_matrix = np.stack(self.token_hvs)
        xor_res = np.bitwise_xor(hvs_matrix, query_hv_packed)
        diff_bits = np.unpackbits(xor_res, axis=1).sum(axis=1)
        
        top_k_idx = np.argsort(diff_bits)[:k]
        return [self.id_to_token[idx] for idx in top_k_idx]

    def find_nearest(self, query_hv_packed: np.ndarray) -> str:
        if not self.token_hvs: return "???"
        
        # On empile les HVs pour la recherche vectorisee
        hvs_matrix = np.stack(self.token_hvs)
        
        # Hamming distance vectorisee sur bits packes (XOR + count_nonzero)
        xor_res = np.bitwise_xor(hvs_matrix, query_hv_packed)
        # On unpack pour compter les bits a 1
        diff_bits = np.unpackbits(xor_res, axis=1).sum(axis=1)
        
        best_idx = np.argmin(diff_bits)
        return self.id_to_token[best_idx]

    def encode_bow(self, tokens: list[str]) -> np.ndarray:
        if not tokens: return np.zeros(self.dim // 8, dtype=np.uint8)
        hvs_bits = [np.unpackbits(self.get_word_hv(t)) for t in tokens]
        sum_hvs = np.sum(hvs_bits, axis=0)
        bits = (sum_hvs > len(tokens) // 2).astype(np.uint8)
        return np.packbits(bits)

`

## v2/scientific_duel_v5_attention.py

`python
import time
import os
import gc
import numpy as np
from hdc.v3_engine import V3Engine, encode_context
from hdc.corpus import load, tokenize
from hdc.representation import DIM
from eval_kneser_ney import KneserNeyModel

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}m {s:02d}s"

def run_duel_v5():
    print("="*60)
    print("      NEMLM V5 - DUEL PERSISTANT (HYBRID ATTENTION)")
    print("="*60)
    
    sentences = load("../europarl_fr.txt", max_tokens=2000000)
    train_sents = sentences[:25000]
    test_sents = sentences[25000:30000]

    db_path = r"D:\nemlm_v5_stable.nemdb"
    engine = V3Engine(DIM, db_path=db_path)
    kn_model = KneserNeyModel(n=5)
    
    print(f"[*] Base de connaissances : {db_path}")
    print("[*] Entrainement Kneser-Ney 5-grammes...")
    kn_model.train(train_sents)

    resume_at = 25000
    print(f"\n[>>>] RATTRAPAGE ATTENTION RAM ({resume_at} phrases)")
    start_train = time.time()
    
    for i, sent in enumerate(train_sents):
        if len(sent) < 2: continue
        for j in range(1, len(sent)):
            context = sent[max(0, j-5):j]
            target = sent[j]
            l_hv_packed = encode_context(context, engine.dim)
            target_hv = engine.semantic.get_word_hv(target)
            engine.attention.learn(l_hv_packed, target_hv)
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_train
            speed = (i + 1) / (elapsed + 0.1)
            eta = (resume_at - (i + 1)) / speed
            print(f" > [{format_time(elapsed)}] {i+1:5d}/{resume_at} | Speed: {speed:5.1f} phr/s | ETA: {format_time(eta)}")
    
    print(f"\n[OK] Attention restaurée en {format_time(time.time() - start_train)}")
    print("[*] Sauvegarde de l'Attention sur disque...")
    engine.commit() 
    
    # --- EVALUATION ---
    print(f"\n[>>>] ÉVALUATION FINALE (5000 questions)")
    hdc_hits = 0
    kn_hits = 0
    total = 0
    
    start_eval = time.time()
    for i, sent in enumerate(test_sents):
        if len(sent) < 5: continue
        context = sent[:4]
        target = sent[4]
        
        # NemLM Predict (Top-5)
        preds_hdc = engine.predict_next(context, top_k=5)
        if target in preds_hdc: hdc_hits += 1
        
        # KN Predict (Top-5)
        preds_kn = kn_model.predict_topk(context, k=5)
        if target in preds_kn: kn_hits += 1
        
        total += 1
        if total % 100 == 0:
            elapsed = time.time() - start_eval
            speed = total / (elapsed + 0.1)
            eta = (len(test_sents) - total) / speed
            print(f" > [{format_time(elapsed)}] Evalué {total:4d}/5000 | NemLM: {hdc_hits/total*100:5.2f}% | KN: {kn_hits/total*100:5.2f}% | ETA: {format_time(eta)}")

    duration = time.time() - start_eval
    
    print("\n" + "="*60)
    print("                RÉSULTATS FINAUX - DUEL V5")
    print("="*60)
    print(f" NemLM (Hybrid V5) : {hdc_hits/total*100:.2f}% (Top-5 Accuracy)")
    print(f" Kneser-Ney (Ref)  : {kn_hits/total*100:.2f}% (Top-5 Accuracy)")
    print(f" Temps Total Eval  : {format_time(duration)}")
    print(f" Status            : Modèle Persistant Sauvegardé sur D:")
    print("="*60)

if __name__ == "__main__":
    run_duel_v5()

`

