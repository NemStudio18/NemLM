# Architecture Technique NemLM V5.3 (High-Fidelity)

## 1. Structure du Système
NemLM est un moteur de langage 100% binaire basé sur le calcul hyperdimensionnel (HDC). La version 5.3 introduit la séparation entre le **Moteur d'Entraînement** (Lourd/Parallèle) et le **Moteur d'Inférence** (Compact/Optimisé).

```text
v2/
├── hdc/
│   ├── parallel_engine.py # Entraînement Multi-Worker (Haute Densité)
│   ├── compact_engine.py  # Inférence Rapide (Distillée + HDC-AR)
│   ├── v3_engine.py       # Orchestrateur de référence (Entraînement Attention)
│   ├── memory.py          # Mémoire Associative (SQLite Full Fidelity)
│   ├── attention.py       # Multi-Head Binary Attention (Fallback Sémantique)
│   ├── semantic.py        # Semantic Index (Mapping Vocabulaire <-> HV)
│   └── representation.py   # Primitives HDC (XOR, Rotations, Accumulator)
└── D:\                   # Stockage SSD
    ├── nemlm_v5_3_full.nemdb         # Base brute (4.8 Go)
    └── nemlm_v5_3_compact_full.nemdb # Base distillée (924 Mo)
```

## 2. Architecture HDC-AR (Autorégressif)
Fusion de la précision syntaxique des n-grammes (Local) avec la cohérence thématique (Global).

### Les trois piliers de la prédiction :
1.  **Contexte Local (N-grammes)** : Fenêtre glissante de 5 tokens encodée par rotations XOR. Match exact via SQLite avec **Early Exit** sur les ordres 5 et 4 pour une précision maximale.
2.  **Contexte Global (Thème)** : Géré par le `ContextAccumulator` (decay 0.95). Il capture "l'odeur" sémantique globale et influence le repli (fallback).
3.  **Darwinian Attention (Fallback)** : En cas d'absence de n-grammes, le système interroge 8 têtes d'attention binaire. Les souvenirs sont sélectionnés par distance de Hamming et survie darwinienne.

## 3. Pipeline de Distillation Haute Fidélité (HF)
Contrairement aux versions précédentes qui élaguaient les données, la V5.3 HF conserve toute la nuance statistique :
- **Weighted Distillation** : On ne garde pas juste les mots, mais aussi leurs poids (fréquence d'apparition) pour un vote pondéré à l'inférence (`score = weight * order^4`).
- **Top-30 Extraction** : Réduction de la taille de la base en ne gardant que les **30 meilleures prédictions** par contexte (au lieu de 5), préservant la longue traîne.
- **Attention Relocation** : Les têtes d'attention entraînées sont migrées directement dans le moteur compact.

## 4. Phase 3B : Binary Transformer Layer (En cours)
L'architecture évolue vers une structure hybride :
- **Base HDC-AR** : Fournit la précision syntaxique et les faits mémorisés.
- **Couche BT** : Couche différentiable (Binary Backprop) pour la généralisation et la structure de phrase hors-corpus.
- **Poids Latents** : Apprentissage float32 -> Inférence binaire (-1, 1).

## 5. Performance & Métriques
- **Taille** : ~1.38 Go pour le modèle HF Top-30 (Europarl 15k).
- **Vitesse d'Inférence** : ~83 questions/s (12ms par test complet).
- **Consommation RAM** : < 1 Go (Mmap SQLite).
- **Accuracy Target** : 32.5%+ en Top-5 (KN-level).

## 6. Détail des Calculs (Ingénierie)

| Opération | Formule / Algorithme | Type | Implémentation |
| :--- | :--- | :--- | :--- |
| **Binding (Local)** | `C = A XOR rotate(B)` | Bitwise | XOR + PRIMES-based roll |
| **Accumulation (Global)** | `G_n = (G_{n-1} * 0.95) + current` | Integer | `np.int16` sum |
| **Similarity** | `Hamming(A, B) = popcount(A XOR B)` | Bitwise | `POPCOUNT_TABLE` (Lookup) |
| **Poids Inférence** | `Score = weight * order^4` | Weighted | Power-based importance |
| **Early Exit** | `If order == 5 and count > 2 -> Return` | Logic | Fast-path optimization |

---
*Dernière mise à jour : Mai 2026 - Milestone High-Fidelity & Reasoning Ready.*
