# HDC-LLM POC — Spécification technique v0.1

## Objectif

Prouver qu'un système Hyperdimensional Computing peut prédire un token suivant
de façon cohérente, sur CPU pur, sans aucun flottant.

---

## Contraintes fondamentales

- Zéro flottant en inférence
- Zéro GPU requis
- Zéro backpropagation
- Vocabulaire : 50-150 mots
- Corpus : 100-200 phrases françaises simples
- Langage : Python 3.10+
- Dépendances : numpy uniquement (pour bitarray rapide)

---

## Architecture

### 1. Couche Représentation — `hdc/representation.py`

**Responsabilité** : associer chaque token à un vecteur binaire hyperdimensionnel stable.

```
Token (str) → HyperVector (bitarray 10000 bits)
```

**Règles** :
- Vecteur généré une seule fois par token, aléatoire, stocké en mémoire
- Reproductible via seed dérivé du token (hash SHA256 → seed numpy)
- Immutable après génération

**Opérations exposées** :
- `encode(token: str) → HV`
- `encode_context(tokens: list[str]) → HV` — XOR + rotation par position
- `hamming(hv1: HV, hv2: HV) → int` — distance binaire

**Détail encode_context** :
```
context ["le", "chat"] :
  hv_le   = encode("le")
  hv_chat = rotate(encode("chat"), 1)  # rotation de 1 bit pour l'ordre
  result  = XOR(hv_le, hv_chat)
```

---

### 2. Couche Mémoire — `hdc/memory.py`

**Responsabilité** : stocker et mettre à jour les associations contexte → token suivant.

**Structure interne** :
```
BTreeMap<HV_hash, AssociativeMemory>
  clé   : hash 64 bits du HV contexte (pour indexation rapide)
  valeur: HV accumulé des tokens suivants observés
```

**Apprentissage (bundling)** :
```
Pour chaque trigramme (t-2, t-1, t) dans le corpus :
  ctx_hv = encode_context([t-2, t-1])
  mem[ctx_hv] = XOR(mem[ctx_hv], encode(t))   # accumulation par XOR
```

**Prédiction** :
```
ctx_hv = encode_context(contexte_courant)
Pour chaque token v dans le vocabulaire :
  score[v] = hamming(mem[ctx_hv], encode(v))
Retourner argmin(score)   # plus proche = plus similaire
```

**Opérations exposées** :
- `learn(context: list[str], next_token: str)`
- `predict(context: list[str]) → str`
- `predict_topk(context: list[str], k: int) → list[tuple[str, int]]`

---

### 3. Couche Corpus — `hdc/corpus.py`

**Responsabilité** : charger, tokenizer, fournir les n-grammes.

**Format corpus** : fichier texte, une phrase par ligne, UTF-8.

**Tokenisation** : split sur espaces + ponctuation basique. Pas de stemming.

**Opérations exposées** :
- `load(path: str) → list[list[str]]`
- `ngrams(sentences, n: int) → Iterator[tuple]`

---

### 4. Couche Évaluation — `hdc/eval.py`

**Responsabilité** : mesurer les performances.

**Métriques** :
- `accuracy@1` — le token prédit est le bon
- `accuracy@3` — le bon token est dans le top 3
- `inference_time_us` — microsecondes par prédiction
- `memory_bytes` — RAM utilisée par la mémoire HDC
- `float_ops` — doit rester 0

**Protocole** :
- Split 80/20 train/test sur le corpus
- Rapport terminal + export JSON

---

### 5. Point d'entrée — `main.py`

```
python main.py --corpus corpus.txt --dim 10000 --train --eval
python main.py --corpus corpus.txt --interactive
```

**Mode interactif** :
```
> le chat
→ Top 3 : mange (dist: 4821), dort (dist: 5103), court (dist: 5289)
→ Temps : 47µs
```

---

## Fichiers

```
hdc_poc/
├── main.py
├── corpus.txt
├── hdc/
│   ├── __init__.py
│   ├── representation.py
│   ├── memory.py
│   ├── corpus.py
│   └── eval.py
└── results/
    └── bench_YYYYMMDD.json
```

---

## Critères de succès POC

| Métrique | Seuil minimal | Seuil bon |
|---|---|---|
| accuracy@1 sur corpus train | > 60% | > 85% |
| accuracy@1 sur corpus test | > 30% | > 50% |
| accuracy@3 sur corpus test | > 50% | > 70% |
| Temps inférence | < 1ms | < 100µs |
| Flottants utilisés | 0 | 0 |
| RAM mémoire HDC | < 50MB | < 10MB |

---

## Limites connues du POC

- Corpus trop petit pour généralisation réelle
- Contexte fixe à 2 tokens (extensible)
- Pas de gestion OOV (out of vocabulary)
- Bundling XOR simple — perd de l'information sur fréquence

## Extensions V2 (hors scope POC)

- Contexte variable longueur
- Bundling majoritaire (majority vote sur N vecteurs)
- Persistance mémoire sur disque (BTree sérialisé)
- Port Rust
- Corpus 10k+ phrases
