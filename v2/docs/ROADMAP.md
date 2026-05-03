# Roadmap NemLM - Moteur Génératif HDC

## Phase 1 : Industrialisation Python (Terminée) ✅
- [x] Architecture Hybride (Exact Match + Binary Attention).
- [x] Persistance SQLite sur SSD (Support 35Go+ / 2M+ tokens).
- [x] Persistance de l'Attention (V5.1).
- [x] Benchmark scientifique vs Kneser-Ney (Duel V5.1 Top-5).
- [x] Optimisation Turbo RAM (300% de gain de vitesse).

## Phase 2 : Performance Native (Prochaine étape) 🚀
- [ ] **Portage Rust Core** : Réécriture des boucles critiques (`_project`, `hamming_batch`) en Rust via PyO3.
- [ ] **Multi-threading** : Parallélisation de l'apprentissage des têtes d'attention.
- [ ] **SIMD Optimization** : Utilisation des instructions AVX-512 pour les calculs de distance de Hamming.
- [ ] **Objectif** : Diviser par 10 le temps d'apprentissage (passage de 1h à 6 min).

## Phase 3 : Intelligence & Scalabilité (V6) 🧠
- [ ] **Multi-scale Encoding** : Superposition des n-grammes (5, 4, 3) dans le même opcode pour égaler le backoff de KN.
- [ ] **Modèle Multi-Contextuel** : Fusion de contextes globaux et locaux.
- [ ] **Fine-tuning HDC** : Apprentissage par renforcement binaire (RLHF sans flottants).
- [ ] **Compaction HDC** : Gel du modèle et passage en 1-bit pour réduire 35 Go -> 1.5 Go.
- [ ] **API Inference** : Serveur minimaliste ultra-rapide (<1ms latence).

## Phase 4 : Déploiement Embarqué 📱
- [ ] **Portage Mobile/Edge** : Exécution sur Raspberry Pi et Smartphones.
- [ ] **NemLM OS** : Micro-noyau dédié à l'exécution de modèles HDC sur matériel minimaliste.

## Vision & Hypothèses Scientifiques 🎯

### Objectif : La "Troisième Voie" de l'IA
L'objectif final de NemLM n'est pas seulement de battre Kneser-Ney, mais de fusionner deux mondes :
1.  **Rigueur Statistique** : Atteindre **40% à 50% d'Accuracy@5** en intégrant le Multi-scale Encoding (5-4-3-2-1).
2.  **Raisonnement Sémantique** : Égaler les capacités de généralisation des Transformers (GPT) en utilisant l'Attention Binaire Profonde.

### Hypothèses Clés :
- **Hypothèse 1 (Fuzzy-Match)** : La recherche par distance de Hamming sur les clés de contexte permettra de trouver des réponses là où les N-grams classiques échouent.
- **Hypothèse 2 (Efficacité)** : Un modèle HDC de 10 Go peut surpasser un modèle statistique de 50 Go grâce à la superposition d'informations (Bundling).
- **Hypothèse 3 (Indépendance)** : Le raisonnement logique peut émerger d'opérations binaires massives sans aucun calcul flottant (FP32).

### Stratégie Technique : Le chemin vers les 50%
Pour dépasser Kneser-Ney, nous allons implémenter trois leviers technologiques :
1.  **Superposition Multi-échelle (Bundling)** : Au lieu de chercher une clé exacte, nous allons compresser les n-grammes (5, 4, 3) dans un seul HV. Si le 5-gramme est inconnu, le 4-gramme "résonnera" toujours dans le vecteur et déclenchera la réponse.
2.  **Recherche Floue (Hamming Key Search)** : En passant au Rust, nous pourrons comparer le contexte actuel à **tous** les contextes connus, même s'ils diffèrent de quelques bits. C'est l'équivalent sémantique du "backoff" de KN, mais en beaucoup plus puissant.
3.  **Expansion de l'Attention** : Passer de 8 à 32 ou 64 têtes d'attention pour capturer des relations sémantiques plus fines (grammaire, style, logique).

---
*Dernière mise à jour : 03/05/2026 02h22 - Phase 1 Terminée.*
