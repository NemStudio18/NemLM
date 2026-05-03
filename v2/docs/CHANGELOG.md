# Changelog - NemLM (LLMonCPU)

## [V5.3] - 2026-05-03 (Parallel Turbo Milestone)
### Ajouté
- **Moteur Parallèle (V3ParallelEngine)** : Architecture multi-processus avec 3 workers spécialisés par ordre de n-gramme.
- **Singleton Pruning (Filtrage)** : Élagage à la source des associations uniques pour réduire le bruit sémantique et la taille de la base (-70% sur Europarl).
- **Performance Boost** : Accélération massive de l'entraînement, passant de 3 phr/s à ~230 phr/s (+7500% de gain).
- **Multi-Connection SQLite** : Gestion sécurisée des écritures concurrentes en mode WAL.


## [V5.2] - 2026-05-03 (HDC-AR Milestone)
### Ajouté
- **HDC-AR (Autorégressif)** : Nouvelle architecture fusionnant contexte Local (syntaxe) et Global (thématique).
- **ContextAccumulator** : Gestion de la cohérence long-terme avec decay exponentiel (0.95).
- **Multi-scale Backoff** : Recherche pondérée sur n-grammes d'ordres 5, 4, 3, et 2.
- **Darwinian Attention** : Mécanisme de survie des souvenirs basé sur les "hits" (fréquence d'utilisation).
- **Popcount Turbo** : Table de lookup pour accélérer la distance de Hamming sur CPU.

### Amélioré
- **LRU Cache** : Remplacement du cache RAM brutal par un `OrderedDict` intelligent.
- **Auto-Commit** : Sécurisation des données via un commit SQLite toutes les 60s ou 500 entrées.

## [V5.1] - 2026-05-02 (Industrial Persistent)

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
