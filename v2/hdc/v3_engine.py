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
        
        # Accumulateur de contexte global (HDC-AR)
        from hdc.representation import ContextAccumulator
        self.accumulator = ContextAccumulator(dim=dim, decay=0.95)
        
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
        
        # On réinitialise l'accumulateur au début de chaque phrase (ou on le garde pour la session ?)
        # Pour Europarl, chaque phrase est indépendante, donc on reset.
        self.accumulator.reset()

        for i in range(1, len(sentence)):
            context = sentence[max(0, i - 5):i]
            target = sentence[i]

            l_hv_packed = encode_context(context, self.dim)
            
            # Apprentissage Attention Binaire (Fallback Sémantique)
            # On utilise le XOR(local, global) pour l'attention
            global_hv = self.accumulator.get_hv()
            query_hv = np.bitwise_xor(l_hv_packed, global_hv)
            
            target_hv = self.semantic.get_word_hv(target)
            self.attention.learn(query_hv, target_hv)
            
            # Mise à jour de l'accumulateur pour le prochain token
            self.accumulator.add(target_hv)
            
            # Apprentissage Associatif Multi-Échelle (Backoff Option A)
            for n in [5, 4, 3, 2]:
                sub_context = sentence[max(0, i - (n-1)):i]
                if n > 1 and not sub_context: continue
                
                n_gram_hv = encode_context(sub_context, self.dim)
                self.memory.learn_one_pass(n_gram_hv, target)

        self.long_term_hvs.append(sent_hv_packed)
        if len(self.long_term_hvs) > 100:
            self.long_term_hvs.pop(0)

    def predict_next(self, context_tokens: list[str], top_k: int = 5) -> list[str]:
        """Orchestre la prediction hybride : HDC-Backoff -> Fallback Attention (Thematique)."""
        
        # 1. Tentative Match Exact avec Backoff (Local uniquement)
        exact_preds = self.memory.predict_with_backoff(context_tokens, self.dim, k=top_k)
        if exact_preds:
            # On met à jour l'accumulateur avec le meilleur choix pour maintenir la cohérence
            best_token = exact_preds[0]
            target_hv = self.semantic.get_word_hv(best_token)
            self.accumulator.add(target_hv)
            return exact_preds
            
        # 2. Fallback Attention Sémantique (Thematique)
        from hdc.representation import encode_context
        l_hv = encode_context(context_tokens, self.dim)
        g_hv = self.accumulator.get_hv()
        query_hv = np.bitwise_xor(l_hv, g_hv)
        
        attention_hv = self.attention.forward(query_hv, k=8)
        predicted_tokens = self.semantic.find_nearest_topk(attention_hv, k=top_k)
        
        if predicted_tokens:
            # Mise à jour de l'accumulateur
            best_token = predicted_tokens[0]
            self.accumulator.add(self.semantic.get_word_hv(best_token))
            
        return predicted_tokens
