# NemLM : LLM Frugal & 100% Bitwise (V5.3 High-Fidelity)

NemLM est une architecture expérimentale de modèle de langage (LLM) conçue pour s'affranchir totalement des GPU et des calculs flottants. Basé sur le **Calcul Hyperdimensionnel (HDC)**, NemLM utilise des opérations binaires (XOR, Popcount, Rotations) pour simuler des capacités de raisonnement et de génération de texte.

## 🚀 Vision & Objectifs
- **Performance GPT-2 Small** : Atteindre la qualité d'un modèle de 117M paramètres sur un CPU legacy.
- **Frugalité Extrême** : Fonctionne avec **4-6 Go de RAM** (Entraînement) et **< 500 Mo** (Inférence).
- **Zéro Flottant** : L'inférence est 100% binaire (Bitwise).
- **Vitesse** : Inférence ultra-rapide de **< 1.3ms/token**.

## 🧠 Architecture HDC-AR (V5.3)
La version actuelle utilise une architecture **Autorégressive Hyperdimensionnelle (HDC-AR)** industrialisée :
- **Multi-Worker Training** : Entraînement parallèle haute densité (No Pruning) pour une fidélité maximale.
- **CompactEngine** : Moteur d'inférence optimisé utilisant une base SQLite distillée de 900 Mo.
- **Multi-scale Backoff** : Prédiction précise via n-grammes d'ordres 5, 4, 3, et 2 avec vote pondéré.
- **Darwinian Attention** : Fallback sémantique global via 8 têtes d'attention binaire.
- **Thematic Accumulator** : Gestion de la cohérence sémantique long-terme (Context Decay 0.95).

## 🛠️ Stack Technique
- **Core** : Python 3.10+ (Optimisé Numpy / SQLite).
- **Persistence** : SQLite (Mode WAL + mmap) avec séparation des tables `distilled` et `attention`.
- **Primitives** : XOR, Bit-Packing, Popcount Table Lookup (Turbo Mode).

## 📊 Benchmarks (Europarl FR - Baseline HF)
- **Accuracy (Top-5)** : **32.03%** (Record Industriel HF).
- **Vitesse Inférence** : **~110 tokens/s** (Mode Compact Précis).
- **Latence** : **~9.1ms** par token (100% Bitwise).

## 🚧 Prochaines Étapes
- **Phase 3B** : Implémentation du **Binary Transformer** (STE Backprop) pour la généralisation.
- **Phase 4** : RAG HDC avec **Reasoning Accumulator** émergent.

---
*Status : Phase 2 VALIDEE. Phase 3B en cours d'implémentation.*
