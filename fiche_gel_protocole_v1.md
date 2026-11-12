# Fiche de gel du protocole — v1

Étude : *When Does Fuzzy Scheduling Help Knowledge Distillation? A Regime Analysis on 3D Medical Image Classification*

Rédigée avant la Phase B, sur la base du pilote (3 datasets, 3 seeds, validation seulement).

**Après le gel : plus aucune modification** du protocole, des bornes, des règles fuzzy, des teachers, des students ou de la définition des verdicts. Toute adaptation à un dataset doit être déclarée comme telle.

## 1. Datasets (MedMNIST 3D, 28³)

| Dataset | Rôle | Classes | Train / val / test |
| --- | --- | ---: | --- |
| OrganMNIST3D | facile, contrôle négatif | 11 | 971 / 161 / 610 |
| AdrenalMNIST3D | intermédiaire | 2 | 1 188 / 98 / 298 |
| FractureMNIST3D | difficile | 3 | 1 027 / 103 / 240 |

- Splits officiels de MedMNIST, sans modification. Aucun doublon exact entre splits ni dans le train. Une fuite au niveau patient ne peut pas être vérifiée.
- Normalisation : moyenne et écart-type scalaires **du train de chaque dataset**, appliqués au train et à la validation (et au test).
- Augmentation (train seulement, appliquée après la normalisation) : translation de ±2 voxels par axe (padding par réplication) ; échelle d'intensité 1 + 0,1 · U(−1, 1) ; décalage d'intensité 0,1 · N(0, 1) ; bruit gaussien 0,05 · N(0, 1). **Pas de flips.**

## 2. Modèles

**Students (CNN 3D simple : Conv3d 3×3×3, BatchNorm, ReLU, MaxPool, 4 étages, pooling global, couche linéaire)** :

| Student | Largeurs | Paramètres (11 classes) |
| --- | --- | ---: |
| S1 | 2, 4, 8, 16 | 4 867 |
| S2 | 4, 8, 16, 32 | 18 795 |
| S3 | 8, 16, 32, 64 | 73 867 |

**Teachers (ResNet 3D à blocs basiques, stem Conv3d 3×3×3 stride 1)** :

| Teacher | Blocs par étage | Largeurs | Paramètres (11 classes) | MFLOPs |
| --- | --- | --- | ---: | ---: |
| T1 = res_w16_b1 | 1-1-1-1 | 16, 32, 64, 128 | 898 779 | 1 058 |
| T2 = res_w32_b1 | 1-1-1-1 | 32, 64, 128, 256 | 3 588 523 | 4 192 |
| T3 = res_w32_b2 | 2-2-2-2 | 32, 64, 128, 256 | 8 290 603 | 8 894 |

- **Règle de choix des teachers :** les 3 candidats avec le plus de couples acceptés sur les 3 datasets avec augmentation (res_w32_b2 : 7 sur 9 ; res_w32_b1 : 5 ; res_w16_b1 : 5 ; plain_w32 : 3, écarté). Règle formulée après avoir vu une partie des résultats ; les trois teachers sont des ResNet 3D (condition « au moins deux architectures » non respectée) : **déclaré comme limite**.
- **Règle d'acceptation :** IC bootstrap à 95 % de la NLL de validation (student moins teacher) entièrement au-dessus de zéro, avec 5 seeds. Un couple sans marge est rapporté comme « sans marge pour la KD » et **n'est pas remplacé**.
- **Les teachers sont entraînés une seule fois par dataset** (seed 0), avec le protocole de la section 3, checkpoint choisi sur la NLL de validation, puis sauvegardés, gelés et réutilisés dans toutes les KD. Les students repartent de zéro à chaque run.

## 3. Entraînement (identique pour tous les modèles et tous les datasets)

- 80 epochs ; AdamW, learning rate 1e-3, weight decay 1e-4 ; batch 32 (un batch de moins de 2 exemples est ignoré) ; schedule cosinus sur 80 epochs, mis à jour à chaque epoch ; précision 32 bits ; pas de gradient clipping.
- Validation à chaque epoch ; **checkpoint = epoch de plus faible NLL de validation**.
- Seed s ∈ {0, …, 4} (et 5 à 14 pour le couple pivot, section 7) : fixe l'initialisation, l'ordre des données et l'augmentation. Les méthodes d'un même couple avec la même seed partagent donc le même départ (comparaisons appariées). Le déterminisme GPU n'est **pas forcé** (raisons et bruit mesuré en section 10).

## 4. Loss KD

```
L = (1 − α) · CE + α · T² · KL( softmax(z_teacher / T) ‖ softmax(z_student / T) )
```

Le teacher est en mode évaluation, gelé, et reçoit la **même entrée augmentée** que le student. Même formulation pour toutes les méthodes.

## 5. Méthodes

**Bornes communes :** T ∈ [2, 8], α ∈ [0,2 ; 0,8]. Confirmées par un test de sensibilité (T = 16 et α = 0,9, 3 seeds, 3 datasets) : aucune amélioration au-delà du bruit entre seeds.

| Méthode | T | α |
| --- | --- | --- |
| Student | n/a | 0 (CE seule) |
| Vanilla KD | 4 | 0,5 |
| Static KD | meilleur point de la grille (voir ci-dessous) | idem |
| lin_T | 8 → 2 | 0,5 |
| lin_A | 4 | 0,8 → 0,2 |
| lin_TA | 8 → 2 | 0,8 → 0,2 |
| FTS | fuzzy | 0,5 |
| FAS | 4 | fuzzy |
| DFS | fuzzy | fuzzy |

- **Progression** p = epoch / 79 (epochs numérotées de 0 à 79). Linear : valeur = début + p · (fin − début).
- **Static KD, par couple teacher–student et par dataset :** grille T ∈ {2, 4, 8} × α ∈ {0,2 ; 0,5 ; 0,8}, 9 points sur les **seeds 0 à 2** ; on retient le point de plus faible NLL de validation moyenne (égalité : plus petit T, puis plus petit α) ; ce point est ensuite lancé sur les seeds 3 et 4 (les seeds 0 à 2 sont réutilisées). Le test n'intervient jamais.

## 6. Contrôleur fuzzy

- **Entrées :** (1) progression p ∈ [0, 1] ; (2) écart teacher–student relatif g. Pour l'epoch e : G_e = moyenne pondérée par la taille des batchs de KL(softmax(z_teacher) ‖ softmax(z_student)) **à T = 1**, sur les batchs d'entraînement de l'epoch (student en mode train, entrée augmentée) ; g = 1 à l'epoch 0 ; pour e ≥ 1, g = clip(G_{e−1} / G_0, 0, 1). Signal causal, calculé sur le train.
- **Ensembles :** partition triangulaire Low / Medium / High sur [0, 1] : μ_low = clip((0,5 − x) / 0,5), μ_med = clip(1 − |x − 0,5| / 0,5), μ_high = clip((x − 0,5) / 0,5).
- **Règles (9, la même base pour T et pour α) :**

| Progression \ Écart | Bas | Moyen | Haut |
| --- | --- | --- | --- |
| Basse | Haut | Haut | Haut |
| Moyenne | Moyen | Moyen | Haut |
| Haute | Bas | Moyen | Moyen |

- **Défuzzification (moyenne pondérée de singletons) :** w_k = max, sur les règles de sortie k, de min(μ_p, μ_g) ; u = (0,5 · w_moyen + 1 · w_haut) / (w_bas + w_moyen + w_haut) ; T = 2 + 6u ; α = 0,2 + 0,6u.
- **Propriétés connues :** pour p < 0,5 et g ≤ 0,5, le contrôleur est identique à Linear ; en seconde moitié, il garde T et α plus hauts quand l'écart persiste ; T et α bougent ensemble (même u). *(Première propriété inexacte : voir l'erratum n° 1, section 13.)*
- FTS ne change que T, FAS ne change que α, DFS change les deux avec les mêmes règles.

## 7. Plan d'expériences

**Principal :** 3 datasets × 3 teachers × 3 students × **5 seeds** (0 à 4), méthodes de la section 5.

| Bloc | Runs |
| --- | ---: |
| Teachers (une fois par dataset) | 9 |
| Student seul | 45 |
| Vanilla, lin_T, lin_A, lin_TA, FTS, FAS, DFS | 945 |
| Grille Static (9 points × 3 seeds × 27 couples) | 729 |
| Meilleur point Static, seeds 3 et 4 | 54 |

**Confirmatoire (déclaré avant les résultats) :** un couple **pivot** par dataset, **res_w32_b2 × S1**, choisi par une règle fixée sur la marge de KD (c'est le couple accepté sur les 3 datasets qui a la plus grande marge teacher–student, d'après le criblage, indépendamment du gain d'une méthode). Sur ce couple, 9 méthodes (Student, Vanilla, Static au meilleur point, lin_T, lin_A, lin_TA, FTS, FAS, DFS) sont lancées sur les **seeds 5 à 14**, soit 15 seeds au total : 3 datasets × 9 méthodes × 10 seeds = 270 runs. Objectif : des intervalles plus étroits sur les ablations E (fuzzy contre Linear) et F (fuzzy contre Static), là où le bruit du pilote est le plus grand.

**Interdit :** ajouter des seeds ou des couples après avoir vu les résultats (arrêt optionnel).

Temps total estimé : environ 17 h pour le GPU 0 et 13 h pour le GPU 1 (estimation à partir des durées mesurées : 26 s par run de KD avec T1, 46 à 56 s avec T2, 74 s avec T3).

## 8. Métriques (par run, au checkpoint choisi sur la NLL de validation)

- **Primaire : Macro-F1 sur le test.** Secondaires : Balanced Accuracy, ECE (15 bins), AUC (binaire : probabilité de la classe 1 ; multiclasse : un-contre-tous, moyenne macro), NLL sur le test.
- Enregistrés aussi : métriques de validation, meilleur epoch, paramètres, FLOPs, temps, T, α, écart g et loss à chaque epoch.
- **Protection du test.** Le test n'est jamais utilisé pour choisir quoi que ce soit et n'est évalué qu'**une seule fois, uniquement pour les runs retenus**, après la sélection du checkpoint sur la validation :
  - **jamais** calculé pour les points de la grille Static (les checkpoints de la grille sont sauvegardés) ;
  - le point Static retenu (choisi sur la NLL de validation, seeds 0 à 2) est évalué au test après sa sélection, depuis ces checkpoints ;
  - **jamais** calculé pendant l'entraînement des teachers : leurs métriques de test sont calculées à l'analyse finale, depuis leur checkpoint ;
  - les métriques de test sont écrites dans les CSV mais **jamais affichées dans les logs** ;
  - le mode de contrôle (`--check`) ne calcule aucun test.

## 9. Statistiques et verdicts

- **Gain** d'une méthode M contre le Student : Macro-F1(M) − Macro-F1(Student), par seed, par couple et par dataset (appariement par seed).
- **Intervalle :** IC à 95 % de **Student (t, n − 1 degrés de liberté)** sur les différences appariées ; le bootstrap est rapporté en complément (à 5 seeds, l'IC bootstrap est trop étroit). Écart-type et moyenne rapportés.
- **Comparaisons clés (par dataset) :** FTS, FAS, DFS contre le Student ; ablations A à D (Vanilla contre FTS ; Vanilla contre FAS ; FTS contre FAS ; FTS + FAS contre DFS) ; E (FTS contre lin_T, FAS contre lin_A, DFS contre lin_TA) ; F (FTS, FAS, DFS contre Static au meilleur point). **Correction de Holm** sur chaque famille, par dataset.
- **Marge d'équivalence : δ = 0,02 en Macro-F1 (test), identique pour les 3 datasets.** Justification : plus petite différence ayant un intérêt pratique (environ 5 volumes sur Fracture, 6 sur Adrenal, 12 sur Organ) ; fixée sans regarder les résultats.
- **Quatre verdicts :**
  - **gagne :** l'IC est entièrement au-dessus de zéro et le p ajusté (Holm) est inférieur à 0,05 ;
  - **perd :** l'IC est entièrement en dessous de zéro et le p ajusté est inférieur à 0,05 ;
  - **égalité :** l'IC est entièrement dans [−0,02 ; +0,02] ;
  - **non concluant :** tous les autres cas.
- **Attendu, à écrire d'avance :** sur Adrenal et Fracture, l'écart-type du gain en F1 était de 0,02 à 0,10 au pilote (validation). Avec 5 seeds, la demi-largeur de l'IC est environ 1,24 fois cet écart-type (±0,025 à ±0,10) : « égalité » sera presque inatteignable et beaucoup de verdicts seront « non concluant ». À 15 seeds (couple pivot), la demi-largeur tombe à environ 0,55 fois l'écart-type (±0,011 à ±0,044). Ce n'est pas un échec : c'est le niveau de preuve réel. Sur Organ (F1 saturé), presque tout sera « égalité » ou « non concluant ».
- **Pas d'agrégation entre datasets** autre que descriptive (3 datasets ne permettent pas de test sur le dataset).

## 10. Reproductibilité et bruit

- Code : `06_screen_models.py`, `07_acceptance.py`, `08_pilot_kd.py` (pilote), `09_phase_b.py` (Phase B, testé avec `--check` et `--smoke`), `10_analysis.py` (analyse finale). Environnement : conda `kd-medmnist`, 2 × RTX A4000. Versions des bibliothèques notées avant le lancement.
- **Le déterminisme GPU n'est pas forcé.** Le mode déterministe de PyTorch (`torch.use_deterministic_algorithms`) lève une erreur sur la rétropropagation de `MaxPool3d` et de `AdaptiveAvgPool3d` sur GPU, deux couches présentes dans tous les modèles de l'étude (documentation PyTorch). Le forcer exigerait de changer les architectures, donc de refaire tout le criblage. Décision : ne pas le forcer, déclarer le non-déterminisme et mesurer le bruit.
- **Bruit mesuré.** Deux exécutions de `09_phase_b.py` (mode `--check`, S1 sur OrganMNIST3D, mêmes seeds 0 à 4) donnent des NLL de validation qui diffèrent de −0,0040 à +0,0069 par seed (écart-type des différences 0,0047, soit environ 0,0033 de bruit par run, environ 9 % de la variance entre seeds, dont l'écart-type est de 0,010 à 0,012 sur ces 5 seeds). Au pilote, à configuration identique et mêmes seeds, la NLL de validation différait de 0,001–0,002 sur Organ et de 0,003 à 0,016 sur Adrenal et Fracture. Conséquence : l'appariement par seed n'est pas parfait ; les intervalles de Student incluent ce bruit (il fait partie de la variabilité mesurée). À rapporter comme seuil de bruit.

## 11. Limites et écarts déclarés

- Teachers : une seule famille (ResNet 3D), 4 candidats, règle de choix formulée après avoir vu une partie des résultats, plain_w32 écarté.
- Chaque teacher est entraîné une fois par dataset : la variation due à l'entraînement du teacher n'est pas mesurée.
- Teachers, students, augmentation, bornes et défuzzification choisis sur la validation (augmentation : Organ et Fracture) : aucun dataset n'est une validation externe de ces choix.
- Validations petites (98, 103 et 161 volumes) : sélection de checkpoint bruitée.
- Static est choisi sur la validation des 3 premières seeds (favorisé sur la validation, pas sur le test).
- Une seule base de règles fuzzy ; T et α bougent ensemble ; le fuzzy est identique à Linear dans la première moitié de l'entraînement *(inexact : voir l'erratum n° 1, section 13)*.
- Une marge de teacher acceptée par la règle ne garantit pas un gain de KD (vu sur Adrenal et Fracture).
- Résolution 28³ seulement ; pas d'identifiants patients.

## 12. Validation

Vérifications faites avant le gel :

- Script de la Phase B (`09_phase_b.py`) testé : contrôle de cohérence avec le pilote (`--check`, aucun test) et test rapide (`--smoke`).
- Gardes de reprise vérifiées : pas de jobs en double, nombre d'epochs identique à la reprise.
- Les valeurs de cette fiche correspondent au code (bornes, grille, règles, δ, seeds).
- Les 9 teachers peuvent être entraînés avec le protocole de la section 3.

## 13. Registre des écarts après le gel

Cette section note les écarts constatés après le gel. Elle ne modifie aucune règle du protocole.

**Écart n° 1 : exposition partielle de métriques de test pendant la Phase B.** Le fichier `results_gpu1.csv` d'OrganMNIST3D a été ouvert et son contenu (colonnes `test_*` comprises) a été collé dans la conversation de travail avec l'assistant, avant la fin des runs. Portée : OrganMNIST3D seulement, couple res_w32_b2 × S1, seeds 0 à 4 pour Vanilla, lin_T, lin_A, lin_TA, FTS, FAS et DFS, et seeds 0 à 2 pour le point Static retenu (soit 8 méthodes). Aucune valeur de test n'a été commentée ni utilisée. Aucun choix du protocole n'a été modifié, aucun run n'a été relancé ni arrêté, et les scripts sont inchangés. À déclarer dans l'article (section Limites).

**Mesures prises :** (1) l'analyse finale reste celle de cette fiche (sections 8 et 9) ; (2) le plan d'analyse est celui des sections 8 et 9 de cette fiche ; le script d'analyse (`10_...`) sera écrit après la fin des runs et se limitera à implémenter ce plan à la lettre ; toute différence entre le script et la fiche sera déclarée dans l'article ; (3) pour consulter les résultats pendant les runs, on lit les CSV sans leurs colonnes de test (`usecols=lambda c: not c.startswith("test_")`).

**Précisions d'implémentation du plan d'analyse (écrites avant toute analyse).** La fiche est ambiguë sur trois points ; le script `10_analysis.py` les tranche ainsi, et l'article le déclarera : (i) l'ablation D (FTS + FAS contre DFS) est évaluée par deux comparaisons, DFS contre FTS et DFS contre FAS ; (ii) la correction de Holm de la portée exploratoire (9 couples, 5 seeds) s'applique à chaque famille de comparaisons, par dataset, en regroupant tous les couples ; la portée confirmatoire (couple pivot, 15 seeds) a ses propres familles ; (iii) en cas de chevauchement « gagne » et « égalité », l'ordre de la fiche prévaut (gagne d'abord), une colonne signalant les gains inférieurs à δ. Le script s'arrête sans rien analyser si un contrôle d'intégrité échoue (jobs manquants ou en double, epochs différentes, run divergent, test présent pour la grille ou absent pour un run retenu, point Static non reproductible depuis la validation).

**Écart n° 2 : changement de cadrage de l'article après les résultats.** Le plan de cette fiche était centré sur la comparaison de schedules de KD, les ordonnanceurs fuzzy en tête. Après avoir vu les résultats de la Phase B et de l'analyse prévue, l'angle de l'article a été déplacé vers une carte des régimes où la KD aide (trois datasets, trois teachers, trois students) ; les fuzzy deviennent des schedules évalués parmi d'autres. Aucun verdict, aucune règle et aucun script de cette fiche n'ont été modifiés. La carte des régimes est rendue confirmatoire par une phase B2a (seeds 15 à 24, Student seul et Vanilla sur les 27 conditions, mêmes teachers, règles de classement et hypothèses déclarées dans un addendum rédigé avant tout calcul). À déclarer dans l'article.

**Écart n° 3 : analyses exploratoires non prévues par cette fiche**, réalisées après l'analyse prévue et à étiqueter comme telles dans l'article : moyennes des gains sur les 9 conditions de chaque dataset, comparaison des méthodes entre elles et classement (avec test de permutation par condition), gains par teacher et par student, corrélations avec des variables de régime (non valables statistiquement : pseudo-réplication), tableaux de gains par métrique secondaire (balanced accuracy, AUC, ECE, NLL), comparaison FTS, FAS et DFS entre eux sur toutes les métriques. Aucune n'a servi à modifier le protocole.

**Erratum n° 1 : description du contrôleur (sections 6 et 11).** Les phrases « pour p < 0,5 et g ≤ 0,5, le contrôleur est identique à Linear » (section 6) et « le fuzzy est identique à Linear dans la première moitié » (section 11) sont inexactes. Vérifié sur la fonction fuzzy de `09_phase_b.py` : le contrôleur est égal à Linear quand g = 0 ; pour p ≤ 0,5 et g ≤ 0,5, il s'en écarte d'au plus 0,043 sur u (environ 0,26 sur T et 0,026 sur α) ; plus tard, ou quand g > 0,5, il garde T et α plus hauts (jusqu'à 0,54 sur u). Le code, les règles et les résultats ne changent pas : seule la description était fausse. L'article utilise la description corrigée.

**Écart n° 4 : addendum B2a non enregistré à l'extérieur.** L'addendum (`addendum_B2a_v1.md`) a été rédigé avant tout run B2a, mais il n'a pas été enregistré à l'extérieur avant le lancement des runs. Son contenu n'a pas été modifié ensuite.
