"""
HDC-LLM V3 - Moteur de Génération Autorégressif
Fusionne le contexte sémantique global et le contexte local n-gramme.
"""

import numpy as np
from .representation import encode_token, encode_context, DIM
from .semantic import SemanticIndex
from .memory import AssociativeMemory

class V3Engine:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.memory = AssociativeMemory(dim)
        self.semantic_index = SemanticIndex(dim=dim)
        self.long_term_hvs = [] 

    def get_global_context_hv(self) -> np.ndarray:
        if not self.long_term_hvs:
            return np.zeros(self.dim, dtype=np.uint8)
        sum_hvs = np.sum(self.long_term_hvs, axis=0, dtype=np.int32)
        return (sum_hvs > len(self.long_term_hvs) // 2).astype(np.uint8)

    def predict_next(self, current_tokens: list[str]) -> str:
        l_hv = encode_context(current_tokens[-5:], self.dim)
        c_hv = self.get_global_context_hv()
        q_hv = np.bitwise_xor(c_hv, l_hv)
        # On passe None pour le vocabulaire si on n'a pas encore build de LSH/Matrix
        # Mais pour le test on va utiliser un scan exact
        return self.memory.predict_topk(q_hv, k=1)

    def generate(self, prompt: str, max_new_tokens: int = 20, vocab: list[str] = None):
        words = prompt.lower().split()
        generated = []
        
        prompt_tokens = prompt.lower().split()
        prompt_hv = self.semantic_index.encode_bow(prompt_tokens)
        self.long_term_hvs.append(prompt_hv)
        
        for _ in range(max_new_tokens):
            # On passe le vocabulaire pour le scan exact dans le test
            res = self.memory.predict_topk(self.get_combined_hv(words + generated), vocabulary=vocab, k=1)
            next_token = res[0] if res else None
            
            if not next_token or next_token == "<eos>":
                break
            generated.append(next_token)
            
            if len(generated) % 10 == 0:
                chunk = generated[-10:]
                self.long_term_hvs.append(self.semantic_index.encode_bow(chunk))
                
        return generated

    def get_combined_hv(self, tokens: list[str]) -> np.ndarray:
        l_hv = encode_context(tokens[-5:], self.dim)
        c_hv = self.get_global_context_hv()
        return np.bitwise_xor(c_hv, l_hv)

    def train_step(self, sentence: str, vocab: list[str] = None):
        tokens = sentence.lower().split()
        c_hv = self.semantic_index.encode_bow(tokens)
        
        for i in range(1, len(tokens)):
            context = tokens[:i][-5:]
            target = tokens[i]
            
            l_hv = encode_context(context, self.dim)
            q_hv = np.bitwise_xor(c_hv, l_hv)
            
            # Prédiction actuelle pour PA
            res = self.memory.predict_topk(q_hv, vocabulary=vocab, k=1)
            pred = res[0] if res else None
            
            self.memory.update_passive_aggressive(q_hv, target, pred)
