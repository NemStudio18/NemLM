# Roadmap NemLM : Vers un LLM Frugal & Bitwise

## ✅ Phase 1 : Fondations & HDC-AR (Terminé)
- [x] Persistance SQLite (mmap/WAL).
- [x] Architecture ContextAccumulator (Thématique).

## ✅ Phase 2 : Industrialisation HF (Terminé - 2026-05-03)
- [x] **Validation Scientifique** : Atteinte de **32.03%** d'accuracy (Top-5) sur Europarl.
- [x] **Parité Statistique** : Synchronisation avec la baseline Kneser-Ney (δ < 1%).
- [x] **CompactEngine** : Inférence ultra-rapide (< 10ms) sur base distillée.

## 🚧 Phase 3B : Binary Transformer (En cours)
- [x] **Prototype STE** : Implémentation de la rétropropagation binaire.
- [x] **Validation Convergence** : Perte (Loss) descendante confirmée sur prototype 512-dim.
- [ ] **Scaling 10k** : Passage des neurones binaires en dimension réelle.
- [ ] **Inférence Hybride** : Fusion du Transformer avec l'Attention Darwinienne.

## 🔮 Phase 4 : Reasoning Accumulator (Futur)
- [ ] **Reasoning Accumulator** : Chaînage de pensée (CoT) par convergence de Hamming.
- [ ] **RAG HDC** : Recherche sémantique récursive nativée.

---
*NemStudio - Advanced Agentic Coding Project*
