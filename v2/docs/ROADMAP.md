# Roadmap NemLM : Vers un LLM Frugal & Bitwise

## ✅ Phase 1 : Industrialisation & HDC-AR (Terminé - Mai 2026)
*   [x] **Persistance SQLite** : Migration vers un stockage SSD total (WAL/mmap).
*   [x] **HDC-AR Architecture** : Implémentation du ContextAccumulator (Thematique).
*   [x] **Multi-scale Backoff** : Système de vote pondéré (5, 4, 3, 2-grammes).
*   [x] **Optimisation Popcount** : Table de lookup pour Hamming distance.
*   [x] **Multi-Worker V5.3** : Entraînement parallèle turbo (3 workers spécialisés).
*   [x] **Singleton Pruning** : Filtrage à la source pour une densité sémantique pure.
*   [x] **Darwinian Attention** : Remplacement des souvenirs par utilité (Hits).

## 🚀 Phase 2 : Inférence Compacte & Déploiement (Suivant)
*   [ ] **Inférence Turbo (C++/Rust)** : Portage du moteur de prédiction pour < 1ms/token.
*   [ ] **Distillation Finale** : Création de la `CompactInferenceEngine` (DB de < 100 Mo).
*   [ ] **Objectif Accuracy** : Consolider les 32%+ sur Europarl et viser les 40% sur texte littéraire.

## 🛠️ Phase 3 : Performance Rust & Binary Transformer (Q3 2026)
*   **NemLM Core (Rust)** : Portage des fonctions `hamming_batch` et `encode_context` via PyO3.
*   **SIMD Acceleration** : Utilisation des instructions AVX2/NEON pour les opérations binaires.
*   **Binary Transformer (BT)** : Ajout d'une couche d'attention différentiable binaire (BitNet logic).
*   **Target** : Inférence < 1ms/token sur CPU frugal.

## 🔮 Vision Long Terme
NemLM vise à devenir le premier modèle de langage capable de rivaliser avec GPT-2 Small (117M) tout en étant :
- **100% Bitwise** (pas de flottants).
- **Indépendant du GPU**.
- **Inscriptible dans 1-2 Go de RAM**.
- **Totalement souverain et local**.

---
*NemStudio - Advanced Agentic Coding Project*
