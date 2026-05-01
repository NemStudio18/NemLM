# HDC-LLM Roadmap 🚀

Objectif : Créer un moteur de langage 100% CPU-natif, indépendant du GPU, capable de tourner sur du matériel de 2012 (i3-3220 / FX-8350).

## Phase 1 : Fondations V2 (En cours) ✅
- [x] Dimension flexible (30k - 100k bits)
- [x] Indexation LSH (Locality Sensitive Hashing) pour recherche O(1)
- [x] Apprentissage itératif (Structured Perceptron / Passive-Aggressive)
- [x] Encodage Subword (Char N-grams) pour la morphologie
- [x] Cache de tokens pour l'accélération Python

## Phase 2 : Pivot Sémantique & Q&A (Priorité immédiate) 🎯
- [ ] **Moteur de Similarité Sémantique** : Encoder des phrases entières dans des Hypervecteurs (HV).
- [ ] **QA Engine** : Indexer des documents et répondre par recherche de similarité Hamming.
- [ ] **Benchmarking Classification** : Tester sur des datasets de classification (SST-2) pour prouver la qualité sémantique.
- [ ] **Optimisation du Recall LSH** : Ajuster L/K pour garantir >95% de précision de recherche.

## Phase 3 : Optimisation & Industrialisation 🛠️
- [ ] **Refactoring Rust** : Portage des fonctions critiques (XOR, popcount, LSH) pour atteindre <100µs.
- [ ] **Quantization int8/int16** : Intégrer des couches de composition entières pour la logique complexe.
- [ ] **BTree KV Cache** : Gérer les contextes longs via des structures de données binaires paginées.

## Phase 4 : Démocratisation & Interface 🌐
- [ ] **Streaming Binaire (Style NHTML)** : Protocol de transport minimaliste pour l'UI.
- [ ] **Multi-plateforme** : Validation sur i3-3220, FX-8350 et terminaux mobiles.
- [ ] **Modèle Cascade** : Orchestration entre petits modèles (HDC) et plus gros modèles locaux.

---
*Dernière mise à jour : 01 Mai 2026*
