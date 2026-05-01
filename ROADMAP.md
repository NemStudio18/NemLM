# Roadmap HDC-LLM 🚀

## Phase 1 : Fondations (Terminé ✅)
- [x] Encodage HDC de base.
- [x] Tokenisation text8.
- [x] Moteur N-gram (V1).

## Phase 2 : Industrialisation Python (En cours 🚧)
- [x] Indexation LSH (Recherche rapide).
- [x] Apprentissage Passive-Aggressive.
- [x] **Fusion XOR(C,L)** : Guidage sémantique global + local (V3).
- [x] **Optimisation Matricielle** : Hamming via `np.dot` (Gain 10x).
- [x] **Parallélisation** : Entraînement multi-processus (Gain 4x).
- [x] **Persistance Binaire** : Format `.hdb` compact.

## Phase 3 : Performance & Rust (Prochainement ⚡)
- [ ] Portage du moteur de calcul en Rust (AVX/SSE).
- [ ] Bindings Python via PyO3.
- [ ] BTree natif pour la mémoire associative (O(log N) massif).
- [ ] Inférence temps réel < 1ms sur i3-3220.

## Phase 4 : Industrialisation (Futur 🏭)
- [ ] Interface CLI complète.
- [ ] Dashboard de monitoring (Metrics).
- [ ] Quantification agressive (1-bit / 2-bit weights).
