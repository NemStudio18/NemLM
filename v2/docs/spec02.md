HDC-LLM POC — Spécification technique v2.0

Objectif

Démontrer qu’un système Hyperdimensional Computing peut rivaliser avec un petit Transformer (1M–10M paramètres) sur la tâche de prédiction du mot suivant, en restant entièrement sur CPU, sans aucune opération flottante, et avec une inférence inférieure à 100 µs même pour un vocabulaire de 50 k mots.

---

Contraintes fondamentales (inchangées)

· ✅ Zéro flottant en inférence
· ✅ Zéro GPU (entraînement et inférence)
· ✅ Zéro backpropagation
· ✅ Vocabulaire final : 50 000 tokens (POC intermédiaire à 1 000 puis 10 000)
· ✅ Corpus : un standard comme text8 ou WikiText-2 (100 M tokens à terme ; POC initial sur 1M tokens)
· ✅ Langage : Python 3.10+ avec numpy
· ✅ Pas de dépendances lourdes (pas de TensorFlow, pas de PyTorch)

---

Architecture v2 — principaux changements

Composant v0.1 v2.0 Justification
Bundling XOR simple Majority vote (seuil 50%) Préserve les fréquences, améliore la précision de 15–20 pts
Contexte Bigramme fixe N-gramme variable (1 à 5) via permutation circulaire Capture des dépendances plus longues, réduit la perplexité
Indexation de la mémoire HashMap sur hash 64 bits du HV Index LSH + SQLite (Persistance industrielle sur disque) Passage à l'échelle (30Go+ de connaissances) sans crash RAM
Apprentissage Bundling naïf une seule passe Hybride (Bundling + Attention Binaire Multi-Têtes) Meilleure généralisation sémantique sur contextes inconnus
Évaluation Accuracy@1/@3 + Perplexité, top-5 accuracy, temps par token, mémoire Comparaison directe avec Kneser-Ney et modèles neuronaux
Attention Fallback Aucun Multi-Head Binary Attention (8 heads) Récupération sémantique quand le match exact échoue

---

1. Couche Représentation — hdc/representation.py

Améliorations

· Vecteurs toujours binaires 10 000 bits, générés par seed déterministe (SHA256).
· Permutation circulaire rotate(hv, n) : rotation de n bits vers la droite, utilisée pour encoder la position dans une séquence de longueur quelconque.

Nouvelle spécification de encode_context

Pour un contexte [t_{-k}, ..., t_{-1}] :

```python
result = encode(t_{-1})
for i in range(1, k):
    result = XOR(result, rotate(encode(t_{{-i-1}}), i))
```

Cela permet des contextes jusqu’à 5 tokens sans explosion mémoire. L’ordre de grandeur des rotations est conservé ; pour des contextes plus longs (>5), on peut tronquer ou utiliser un facteur multiplicatif.

---

2. Couche Mémoire — hdc/memory.py

2.1 Stockage interne

· HashMap (hash_ctx → MemoryEntry).
  hash_ctx est un entier 64 bits obtenu par LSH (voir 2.3).
· MemoryEntry :
  · vector : HV binaire accumulé (par majority vote)
  · count : nombre d’exemples ayant contribué (pour le calcul du seuil)
  · counts_per_token : dictionnaire optionnel pour un debug rapide.

2.2 Bundling : Majority Vote (pas XOR)

À l’apprentissage, pour chaque occurrence d’un contexte ctx suivi d’un token t :

```
entry = memory[hash(ctx)]
entry.counts_per_token[t] += 1
# Mise à jour paresseuse du vecteur majoritaire quand on interroge
```

Pour la prédiction, le vecteur majoritaire est recalculé bit à bit :

```
pour chaque bit j de 0 à D-1:
    somme = 0
    pour chaque token v dans token_counts:
        si le bit j de encode(v) == 1: somme += token_counts[v]
        sinon: somme -= token_counts[v]
    entry.vector[j] = 1 si somme > 0, sinon 0
```

Cette opération n’est pas effectuée à chaque ajout mais seulement lors d’une requête predict (ou en différé quand count dépasse un seuil). Ainsi, l’apprentissage reste léger, et l’inférence reste un simple calcul de distance de Hamming entre ce vecteur majoritaire et les vecteurs candidats.

2.3 Index LSH (v2 obligatoire pour le passage à l’échelle)

Objectif : trouver les contextes similaires à celui donné sans parcourir toute la mémoire.

Mécanisme :

· On génère L = 10 tables de hachage.
· Chaque table contient K = 16 hyperplans aléatoires (vecteurs binaires de dimension D).
· Pour chaque vecteur de contexte ctx_hv, on calcule une empreinte binaire de K bits en signant le produit scalaire (Hamming) avec les hyperplans (strictement : bit = 1 si hamming(ctx_hv, plane) < D/2, car le vecteur est binaire). Le D/2 correspond à l’absence de corrélation.
· On insère hash_ctx dans le bucket correspondant.
· À la requête, on calcule l’empreinte du contexte demandé, on récupère les buckets correspondants dans chaque table, et on ne compare que les contextes uniques ainsi rassemblés (quelques centaines au lieu de toute la mémoire).

La structure de la mémoire devient :

```python
tables = [ { bucket_key: set(hash_ctx) } for _ in range(L) ]
main_memory = dict(hash_ctx → MemoryEntry)
```

La prédiction top-k est alors :

```
candidats = union des hash_ctx des L buckets touchés
scores = [hamming(main_memory[h].vector, token_vector) for h in candidats]
retourner les k tokens avec les plus petites distances
```

---

### 2.4 Couche d'Attention Binaire (V3 Fallback)

Objectif : Fournir une prédiction sémantique lorsque le contexte exact est absent de la mémoire associative.

Mécanisme :
- **Multi-Têtes** : 8 têtes indépendantes.
- **Projections** : Chaque tête utilise une permutation aléatoire fixe du vecteur de contexte.
- **Mémoire Associative Binaire** : Chaque tête stocke les HVs cibles associés à ses clés projetées.
- **Fusion** : Les scores des 8 têtes sont moyennés pour extraire le candidat le plus probable par consensus sémantique.

Cette couche permet à NemLM de "deviner" intelligemment un mot même s'il n'a jamais vu la séquence exacte de tokens auparavant, en se basant sur la similarité hyperdimensionnelle.

---

3. Apprentissage itératif (perceptron binaire)

L’accumulation par majority vote est puissante, mais elle ne corrige pas les erreurs de prédiction. La v2 intègre une passe d’ajustement optionnelle, sans flottants.

Algorithme :

· Après avoir construit la mémoire initiale par bundling majoritaire, on reparcourt le corpus d’entraînement une à trois fois.
· Pour chaque phrase, pour chaque position :
  · Contexte ctx, token cible true_next.
  · Prédiction du modèle : predicted = memory.predict(ctx) (top-1).
  · Si predicted != true_next :
    · On renforce l’association vers true_next : on ajoute à memory[hash(ctx)].counts_per_token[true_next] un poids additionnel (ex. +2).
    · On pénalise l’association vers predicted : on décrémente son poids dans ce même contexte (ex. -1).
· Ces ajustements sont purement entiers (pas de gradient). Après plusieurs époques, le vecteur majoritaire reflète mieux les cooccurrences corrigées.

Résultat : gain de 5–15 % d’accuracy sur les splits de test, rapprochant le système d’un petit réseau feedforward.

---

4. Métriques d’évaluation (mise à jour)

Métrique Seuil cible v2 (corpus 1M–10M tokens) Seuil « rivalise avec petit réseau »
Accuracy @1 (test) 45 % 60 %
Accuracy @3 (test) 70 % 80 %
Perplexité < 100 < 40
Temps d’inférence (moyen) < 50 µs (vocab 50k) < 30 µs
RAM totale < 100 Mo (vocab 50k) < 200 Mo
Flottants utilisés 0 (conserve l’inférence bitwise) 0

---

5. Plan d’implémentation incrémental (pour rester découplé)

Phase 1 — V2 core

· Remplacer XOR par majority bundling.
· Étendre encode_context à longueur variable (1–5).
· Garder la recherche exhaustive (vocab < 2 000).
· Valider sur corpus de 100–200 phrases avec des métriques.

Phase 2 — Index LSH + scalabilité

· Implémenter les tables LSH et le routage des requêtes.
· Passer à un vocabulaire de 10 000 → 30 000 mots sur un corpus de quelques millions de tokens (ex. text8 tronqué).
· Mesurer la progression de la vitesse.

Phase 3 — Apprentissage correctif

· Ajouter la boucle de correction d’erreur (perceptron binaire).
· Mesurer le gain de précision et de perplexité.

Phase 4 — Benchmarks comparatifs

· Évaluer sur WikiText-2 (petit) ou text8 complet.
· Comparer directement à un petit LSTM (1M paramètres) entraîné sur le même corpus, inférence CPU.
· Afficher des courbes précision/temps.

---

6. Structure des fichiers (inchangée, enrichie)

```
hdc_poc/
├── main.py
├── corpus.txt (ou text8.txt)
├── hdc/
│   ├── __init__.py
│   ├── representation.py   # +rotate, encode_context len variable
│   ├── memory.py           # +majority vote, LSH, perceptron
│   ├── lsh.py              # nouvellement extrait
│   ├── corpus.py
│   └── eval.py             # +perplexité, top-5
└── results/
    └── bench_v2_YYYYMMDD.json
```

---

7. Faisabilité sur CPU et respect de la philosophie

· Toutes les opérations restent bitwise (XOR, rotations, comptage de bits, comparaisons entières). Les seuls « calculs » sont des additions/soustractions d’entiers dans le majority vote – toujours aucun flottant.
· L’apprentissage correctif n’introduit aucun calcul de gradient ; il ne nécessite pas de rétropropagation.
· L’index LSH utilise des hyperplans et des tables de hachage, entièrement implémentables avec numpy et des bitarrays.
· L’objectif final est de tourner sur un CPU standard (voire un ARM sans FPU) en dessous de 100 µs par token, avec une qualité de prédiction comparable à celle d’un petit modèle neuronal entraîné par SGD, démontrant qu’il est possible de rivaliser avec des architectures GPU sur leur propre terrain, tout en restant sur CPU.