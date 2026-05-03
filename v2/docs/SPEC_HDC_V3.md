# Spécification NemLM HDC V3.5 - Industrial Edition

Cette version "Industrielle" fait passer le POC à un moteur de production capable de gérer des dizaines de gigaoctets de connaissances sur disque tout en conservant une inférence ultra-rapide sur CPU.

## 1. Représentation Binaire Compacte
Un Hypervecteur (HV) de dimension $D=10,000$ est stocké sous forme de $D/8$ octets (1250 octets).
- **Format** : `np.packbits` (MSB first).
- **Opérations** : Les calculs de distance sémantique s'effectuent via XOR binaire suivi d'un comptage de bits (Hamming).

## 2. Architecture Hybride V5
NemLM V3.5 utilise deux couches de mémoire complémentaires :
### 2.1 Mémoire Associative (Exact Match)
- **Stockage** : SQLite sur SSD (`D:\`).
- **Structure** : `int16[D]`. L'utilisation d'entiers 16 bits permet d'accumuler des milliers d'exemples par contexte sans saturation (contrairement à l'int8).
- **Récupération** : Majority Vote dynamique sur les sommes pondérées.

### 2.2 Attention Binaire Multi-Têtes (V5)
- **Architecture** : 8 têtes de 1024 clés de 10 000 bits.
- **Persistance** : Stockage des matrices de clés et de valeurs dans la table `storage` de SQLite (`attn_head_0` à `attn_head_7`).
- **Inférence** : Consensus par vote majoritaire sur les Top-K résultats des 8 têtes.
- **Accuracy** : Évaluée en Top-5 (5 candidats sémantiques les plus proches via Hamming).

## 3. Persistance & Scalabilité
- **Moteur** : SQLite 3 avec mode WAL (Write-Ahead Logging).
- **Optimisations** : 
  - `mmap_size = 2 Go` : Mapping mémoire pour accès SSD instantanés.
  - `cache_size = 2 Go` : Cache de pages pour les contextes fréquents.
  - **Pruning** : Élagage automatique des objets RAM vers le disque.

## 4. Algorithmique Bit-à-Bit (Inférence)
- **Vitesse** : < 1ms par token sur CPU standard.
- **Zéro Flottant** : Aucune opération de multiplication matricielle ou de virgule flottante durant la génération.
- **Fusion sémantique** : XOR entre le contexte local (N-grammes) et la mémoire à long terme.

## 5. Spécificités Linguistiques
- **Tokenisation** : Regex optimisée pour le français (Europarl).
- **Contexte** : Fenêtre glissante de 5 tokens (Trigrammes/Pentagrammes).
- **Encodage** : Permutations circulaires pour préserver l'ordre des mots dans l'hypervecteur de contexte.

---
*État du projet : Industrialisé sur CPU (Python/NumPy). Prochaine étape : Portage Rust.*
