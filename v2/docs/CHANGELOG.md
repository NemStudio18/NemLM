# Changelog - NemLM (LLMonCPU)

## [V3.5.0] - 2026-05-02 (Session Industrielle)

### Ajouté
- **Moteur Hybride V5** : Intégration de l'Attention Binaire Multi-Têtes (8 têtes) comme mécanisme de repli sémantique.
- **V5.1 (Industrial Persistent)** :
    - Implémentation de la persistance SQLite pour l'Attention Binaire (8 têtes).
    - Passage au Top-5 Accuracy pour le duel scientifique (équité vs Kneser-Ney).
    - Système de télémétrie en temps réel avec calcul d'ETA et vitesse d'inférence.
    - Correction du bug de variable `start_ff` et optimisation du rattrapage RAM.
- **V5.0 (Stable Migration)** : 
    - Migration de la mémoire associative vers SQLite (SSD D:).
SQLite optimisé avec `mmap` et cache de pages de 2 Go.
- **Système de Reprise (Resume)** : Capacité théorique à reprendre un apprentissage après interruption (testé durant la migration disque).
- **Reporting ETA** : Calcul en temps réel du temps restant par tranche de 500 phrases.

### Modifié
- **Optimisation RAM** : Passage d'un mode tout-en-mémoire (instable) à un mode hybride disque performant.
- **Migration Stockage** : Déplacement des bases de données massives (30 Go+) du lecteur système C: vers le SSD D: pour éviter la saturation.
- **Stabilité** : Correction des conflits avec les serveurs de langue (LSP) et harmonisation de la Garbage Collection (`gc.collect`).

### Corrigé
- **Bug de saturation disk** : Résolution du crash "Espace insuffisant" par déportation des données sur D:.
- **Corruption SQLite** : Détection et nettoyage des bases malformées après crash système.
- **Memory Leak** : Stabilisation de la consommation privée de Python à ~1,8 Go.

---
## [V3.0.0] - Précédent
- Architecture de base NemLM V3.
- Implémentation initiale de l'HDC (Vecteurs de 10 000 bits).
