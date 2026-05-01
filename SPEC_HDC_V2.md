# Spécifications Techniques HDC-LLM V3

## 1. Architecture du Moteur (XOR-Fusion)
Le moteur repose sur la fusion de deux types de représentations :
- **Contexte Sémantique (C)** : Vecteur de Haute Dimension (D=30k) représentant le sens global (Bag of Sentences).
- **Contexte Local (L)** : Vecteur représentant la séquence immédiate des tokens (N-grams avec rotations).
- **Vecteur de Requête (Q)** : $Q = XOR(C, L)$

## 2. Persistance Binaire (Format .hdb)
Pour garantir la performance et la portabilité vers Rust, le format binaire suit la structure suivante :

### Header (16 octets)
- `[4b]` Magic Number : `b"HDC3"`
- `[4b]` Version : `uint32` (Actuel : 1)
- `[4b]` DIM : `uint32` (Dimension des vecteurs)
- `[4b]` Reserved : `uint32` (0)

### Corps (N entrées)
Chaque entrée dans la mémoire associative est stockée de manière séquentielle :
- `[8b]` Hash Key : `uint64` (Identifiant unique du contexte)
- `[Db/8]` Packed HV : `bits` (Hypervecteur compressé par 8 bits par octet)

## 3. Mécanisme de Génération
La génération est autorégressive :
1. Calcul de `C` (Context sémantique glissant).
2. Calcul de `L` (Tokens récents).
3. Retrieval $Q \to Token_{next}$ via recherche Hamming dans le BTree.
4. Mise à jour de `L` (append $Token_{next}$).
5. Mise à jour périodique de `C`.

## 4. Objectifs Inférence (Cible Rust)
- Latence cible par token : **< 100µs**.
- CPU : Intel i3-3220 (SSE4.2).
- RAM : < 1 Go pour 1M d'entrées.
