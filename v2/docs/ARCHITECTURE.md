# Architecture Technique NemLM V5

## 1. Structure des Dossiers
```text
LLMonCPU/
├── SPEC_HDC_V3.md          # Spécification mère (Source de vérité)
├── CHANGELOG.md             # Historique des versions
├── ROADMAP.md               # Objectifs futurs (Rust, SIMD)
└── v2/                      # Code source V5 (Python)
    ├── scientific_duel_v5_attention.py  # Script de Benchmark principal
    ├── duel_v5_turbo.log    # Logs de progression en temps réel
    ├── hdc/                 # Cœur algorithmique
    │   ├── v3_engine.py     # Orchestrateur (Fusion Mémoire + Attention)
    │   ├── memory.py        # Gestion SQLite + Cache RAM (int16)
    │   ├── attention.py     # Multi-Head Binary Attention
    │   ├── semantic.py      # Index sémantique (Vocabulaire -> HV)
    │   └── representation.py # Encodage XOR + Rotations circulaires
    └── D:\nemlm_v5_stable.nemdb  # Base de connaissances SSD (35Go+)
```

## 2. Flux de Données (Dataflow)

### Phase d'Entraînement
1. **Input** : Phrase (String) -> **Tokenizer** -> Liste de tokens.
2. **Semantic Index** : Conversion des tokens en Hypervecteurs (10,000 bits packés).
3. **Context Encoder** : Fusion des tokens [t-4, t-3, t-2, t-1] via XOR et Rotations.
4. **Apprentissage Double** :
   - **Associative Memory** : Mise à jour de la somme pondérée sur disque (SQLite).
   - **Attention Heads** : Projection du contexte sur les 8 têtes et stockage des cibles en RAM.

### Phase d'Inférence (Génération)
1. **Requête** : "Le chat mange la..."
2. **Context Encoding** : Génération du HV de contexte actuel.
3. **Recherche Exacte (Fallback 1)** :
   - SQL SELECT dans `memory.nemdb`.
   - Si trouvé : Majority Vote -> Hamming Distance -> Top-K tokens.
4. **Attention Sémantique (Fallback 2)** :
   - Si Fallback 1 échoue (score < seuil) :
    - **SQLite Engine** : Gère à la fois la mémoire associative (Contexts) et les poids de l'Attention (Heads).
    - **Shared Storage** : Utilise une table unique `storage` avec des préfixes de clés pour séparer les types de données.
   - Projection du contexte sur les 8 têtes.
   - Consensus des têtes sur les candidats sémantiquement proches.
5. **Output** : "souris" (Consensus Global).

## 3. Détail des Calculs (Ingénierie)

| Opération | Formule / Algorithme | Type |
| :--- | :--- | :--- |
| **Binding** | `C = A XOR rotate(B)` | Bitwise |
| **Similarity** | `Hamming(A, B) = popcount(A XOR B)` | Bitwise |
| **Majority Vote** | `bit_j = 1 if sum(votes_j) > 0 else 0` | Integer |
| **Attention Key** | `key = context_hv XOR projection_matrix` | Bitwise |
| **LSH** | `bucket = sign(HVs . Hyperplanes)` | Bitwise |

---
*Document conçu pour transmission à un ingénieur système/IA.*
