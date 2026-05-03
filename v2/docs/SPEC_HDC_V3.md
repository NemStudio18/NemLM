# Spécification Technique HDC V3 (Autorégressif)

## 1. Représentation des Données
- **Dimension (D)** : 10,000 bits (par défaut).
- **Format** : Vecteurs binaires packés (uint8).
- **Bit-packing** : Chaque octet contient 8 dimensions.

## 2. Encodage du Contexte (HDC-AR)

### 2.1 Contexte Local (N-grammes)
Le contexte local est généré par la combinaison XOR de tokens décalés par rotation circulaire :
`L_hv = T_n XOR rotate(T_{n-1}, 1) XOR rotate(T_{n-2}, 2) ...`

### 2.2 Contexte Global (Accumulateur)
L'accumulateur maintient un état sémantique persistant sur la séquence :
`G_sum = (G_sum * 95) // 100 + T_n_bits`
`G_hv = (G_sum > 0)`

### 2.3 Fusion Sémantique
Pour l'attention, on fusionne les deux signaux :
`Query_hv = L_hv XOR G_hv`

## 3. Mécanisme de Backoff (Associative Memory)
La prédiction par match exact utilise un système de vote pondéré dégressif sur les ordres de n-grammes :

| Ordre (n) | Poids (n^3) | Description |
| :--- | :--- | :--- |
| 5 | 125 | Contexte quasi-certain |
| 4 | 64 | Contexte fort |
| 3 | 27 | Contexte syntaxique |
| 2 | 8 | Contexte grammatical minimal |

**Formule de score** : `Score(token) = sum(Count_n * n^3)`

## 4. Attention Darwinienne (Evolutionary)
Les têtes d'attention gèrent leur mémoire de manière adaptative :
- Chaque souvenir (`context -> target`) possède un compteur de `hits`.
- En cas de saturation, le souvenir avec `min(hits)` est écrasé.
- Une interrogation réussie (`top_k`) incrémente le compteur de hits des souvenirs concernés.

## 5. Optimisations Matérielles (CPU)
- **Hamming** : Utilisation obligatoire d'une table de pré-calcul (Lookup Table) pour le `popcount` des octets.
- **Persistence** : Stockage SQLite en mode WAL avec indexation B-Tree sur les clés binaires.

---
*Cette spécification fait autorité pour toutes les implémentations de la famille NemLM V5.x.*
