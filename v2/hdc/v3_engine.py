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
