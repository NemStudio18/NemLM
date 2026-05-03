# Architecture Technique NemLM V5.3 (Multi-Worker)

## 1. Structure du Système
NemLM est un moteur de langage 100% binaire basé sur le calcul hyperdimensionnel (HDC). La version 5.3 introduit une architecture **multi-processus** pour l'entraînement à haute performance.

```text
v2/
├── hdc/
│   ├── parallel_engine.py # Moteur Multi-Worker (3 Workers spécialisés)
│   ├── v3_engine.py      # Orchestrateur HDC-AR (Fusion Local/Global)
│   ├── memory.py         # Associative Memory (SQLite + Multi-scale Backoff)
│   ├── attention.py      # Multi-Head Binary Attention (Evolutionary Memory)
│   ├── semantic.py       # Semantic Index (Mapping Vocabulaire <-> HV)
│   └── representation.py  # Primitives HDC (XOR, Rotations, Accumulator)
└── D:\memory.nemdb       # Stockage concurrent (Mode WAL / busy_timeout 10s)
```

## 2. Architecture HDC-AR (Autorégressif)
NemLM V5.2 introduit le concept de **HDC-AR**, fusionnant la précision syntaxique des n-grammes avec la cohérence thématique des Transformers.

### Les deux piliers du contexte :
1.  **Contexte Local (Syntaxe)** : Fenêtre glissante de 5 tokens encodée par rotations XOR. Utilisée pour le match exact avec **Backoff multi-échelle** (5, 4, 3, 2-grammes).
2.  **Contexte Global (Thème)** : Géré par le `ContextAccumulator`. C'est une somme pondérée cumulative avec **decay (0.95)**. Il capture "l'odeur" sémantique de tout ce qui a été dit précédemment.

### Flux d'Inférence (Dataflow) :
1.  **Requête** -> Tokenization.
2.  **Backoff Search** : Interrogation SQLite sur les ordres 5, 4, 3 et 2 (Local uniquement).
3.  **Thematic Fallback** : Si aucun match exact n'est trouvé :
    - Fusion `Query = Local XOR Global`.
    - Projection sur 8 têtes d'attention binaire.
    - Consensus par vote majoritaire sur les candidats sémantiques.
4.  **Accumulation** : Le token choisi est injecté dans l'accumulateur global pour influencer le futur.

## 3. Détail des Calculs (Ingénierie)

| Opération | Formule / Algorithme | Type | Implémentation |
| :--- | :--- | :--- | :--- |
| **Binding (Local)** | `C = A XOR rotate(B)` | Bitwise | XOR + `np.roll` |
| **Accumulation (Global)** | `G_n = (G_{n-1} * 0.95) + current` | Integer | `np.int16` sum |
| **Similarity** | `Hamming(A, B) = popcount(A XOR B)` | Bitwise | `POPCOUNT_TABLE` (Lookup) |
| **Backoff weight** | `Score = count * order^3` | Weighted | Exponential decay |
| **Evolutionary Mem** | `Replacement = min(hits)` | Darwinian | Hit counter per slot |
| **Pruning (V5.3)** | `Filter = count >= 2` | Threshold | RAM-based Hash set |

## 5. Filtrage des Singletons (Élagage)
Pour maximiser la densité sémantique et réduire la taille de la base SQLite, la V5.3 implémente un filtre à seuil :
- **Hash Tracking** : Chaque worker maintient un set de hashs en RAM (`seen_once`).
- **Insertion Différée** : Un hypervecteur n'est engagé dans SQLite que lors de sa **deuxième occurrence**.
- **Impact** : Réduction drastique du bruit statistique et accélération de l'I/O disque.

## 4. Performance & Stockage
- **Moteur SQLite** : Optimisé avec `PRAGMA journal_mode = WAL` et `mmap_size = 2Go`.
- **Cache RAM** : Vrai cache LRU via `OrderedDict` pour les entrées de mémoire associative.
- **Latency Target** : < 10ms par token sur CPU i3 (100% bitwise).

---
*Dernière mise à jour : Mai 2026 - Milestone HDC-AR Phase 1 Complete.*
