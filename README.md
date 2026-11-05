# GEI723 — Locomotion hexapodale par réseaux de neurones à décharges

## Présentation

Ce projet a été réalisé dans le cadre du cours **GEI723 — Neurosciences computationnelles et applications en traitement de l'information**.

L'objectif est de simuler la commande locomotrice d'un hexapode à l'aide de neurones à décharges dans **Brian2**. Le modèle ne cherche pas à reproduire la mécanique exacte des pattes : les spikes représentent les **instants auxquels des commandes motrices sont émises**.

Le projet comprend deux parties :

- **Partie I — système non autonome** : marche avant, marche arrière, variation de vitesse, rotation sur place, déplacement en cercle et généralisation à un grand nombre de pattes ;
- **Partie II — système pseudo-autonome** : adaptation du comportement en fonction de capteurs simulés détectant des obstacles à gauche, à droite, devant ou derrière.

Le code principal se trouve dans :

```text
GEI723_Locomotion_Hexapodale_Projet.ipynb
```

Les figures produites par le notebook sont automatiquement enregistrées dans :

```text
results_hexapod/
```

---

## Dépendances

Le projet utilise Python et les bibliothèques suivantes :

```text
brian2
numpy
pandas
matplotlib
jupyter
```

Installation avec `pip` :

```bash
pip install brian2 numpy pandas matplotlib jupyter
```

---

## Exécution

1. Ouvrir le notebook :

```text
GEI723_Locomotion_Hexapodale_Projet.ipynb
```

2. Exécuter les cellules **dans l'ordre**, depuis le début du notebook.

3. Pour reproduire l'ensemble des expériences, utiliser :

```text
Run All
```

Le notebook fixe une graine aléatoire :

```python
RANDOM_SEED = 7
```

et un pas de simulation Brian2 de :

```python
defaultclock.dt = 0.5 * ms
```

afin de faciliter la reproductibilité.

---

# Architecture du réseau

## 1. CPG distribués

Chaque patte possède son propre **Central Pattern Generator (CPG)** modélisé par un neurone LIF tonique.

Pour trois paires de pattes, les CPG sont :

```text
L1  L2  L3  R1  R2  R3
```

Ils sont répartis en deux groupes locomoteurs :

```text
Groupe A : L1, R2, L3
Groupe B : R1, L2, R3
```

Les connexions excitatrices et inhibitrices locales entre CPG permettent de maintenir une alternance entre les deux groupes.

Le CPG détermine principalement **quand une patte doit recevoir une commande**.

---

## 2. Couche prémotrice

Chaque patte possède deux neurones prémoteurs :

```text
PM-F : voie Forward
PM-B : voie Backward
```

Chaque CPG excite les deux voies :

```text
CPG_i → PM-F_i
CPG_i → PM-B_i
```

Un spike du CPG seul n'est normalement pas suffisant pour faire spiker le prémoteur. La voie réellement sélectionnée dépend donc aussi du contexte central.

Les deux voies d'une même patte s'inhibent réciproquement :

```text
PM-F_i ─| PM-B_i
PM-B_i ─| PM-F_i
```

Cette compétition limite l'activation simultanée des commandes Forward et Backward.

---

## 3. Motoneurones

Les neurones prémoteurs excitent les motoneurones correspondants :

```text
PM-F_i → F_i
PM-B_i → B_i
```

Les motoneurones représentent la **sortie finale de commande** de chaque patte.

---

# Variables de modulation principales

Les prémoteurs utilisent trois influences différentes :

### `I_bias`

Biais descendant utilisé principalement dans la **Partie I** pour imposer une direction de déplacement.

### `I_cmd`

Influence synaptique lente provenant des neurones centraux `FORWARD` et `BACKWARD`.

```text
FORWARD  → excite PM-F
FORWARD  ─| PM-B

BACKWARD → excite PM-B
BACKWARD ─| PM-F
```

`I_cmd` décroît progressivement lorsqu'il n'est plus alimenté.

### `I_turn`

Influence inhibitrice transitoire utilisée pour les virages.

```text
TURN_LEFT  ─| PM-F gauches
TURN_RIGHT ─| PM-F droits
```

Le virage est donc produit **au niveau prémoteur**, sans imposer une fréquence différente aux CPG gauche et droit.

Cela permet de préserver au mieux l'organisation temporelle des groupes locomoteurs.

---

# Partie I — Système non autonome

Les capteurs sont désactivés :

```python
sensor_control=False
```

## Marche avant

Les prémoteurs Forward sont favorisés sur les deux côtés :

```text
CPG → PM-F → F
```

Les motoneurones Backward restent silencieux.

## Marche arrière

Les prémoteurs Backward sont favorisés :

```text
CPG → PM-B → B
```

Les motoneurones Forward restent silencieux.

## Variation de vitesse

La fréquence cible des CPG peut être modifiée grâce aux paramètres :

```python
left_freq
right_freq
```

La fonction `target_frequency_to_drive()` convertit la fréquence cible en courant nécessaire au neurone LIF.

L'expérience montre qu'une augmentation de la fréquence cible entraîne une augmentation de la fréquence réellement mesurée des CPG.

## Rotation sur place

La rotation est obtenue en commandant les deux côtés dans des directions opposées.

Exemple pour une rotation vers la gauche :

```text
côté gauche : Backward
côté droit  : Forward
```

## Déplacement en cercle

Pour le déplacement en cercle, les fréquences intrinsèques des CPG gauche et droit sont conservées identiques.

Le virage est créé par l'activation tonique du neurone central de virage via :

```python
turn_command="left"
```

ou :

```python
turn_command="right"
```

Par exemple, pour un cercle à gauche :

```text
TURN_LEFT
    ↓
inhibition des PM-F gauches
    ↓
moins de commandes Forward à gauche
    ↓
plus de commandes Forward à droite qu'à gauche
    ↓
courbure vers la gauche
```

L'asymétrie est donc mesurée au niveau des **sorties motrices**, et non en imposant une différence de fréquence aux CPG.

---

# Partie II — Système pseudo-autonome

La commande sensorielle est activée avec :

```python
sensor_control=True
```

Quatre neurones sensoriels sont utilisés :

```text
S-LEFT
S-FRONT
S-RIGHT
S-REAR
```

La distance à un obstacle est convertie en fréquence sensorielle :

```text
obstacle proche → fréquence élevée
obstacle éloigné → fréquence faible
obstacle hors de portée → 0 Hz
```

La relation utilisée est :

```text
x = clip(1 - d / d_max, 0, 1)
f_sensoriel = x * f_max
```

---

## Neurones centraux de décision

Cinq neurones centraux sélectionnent les comportements :

```text
FORWARD
BACKWARD
TURN_LEFT
TURN_RIGHT
ACCELERATE
```

Les connexions sensorielles sont :

```text
S-RIGHT → TURN_LEFT
S-LEFT  → TURN_RIGHT
S-FRONT → BACKWARD
S-REAR  → ACCELERATE
```

Le capteur frontal inhibe également `FORWARD`.

En absence de danger, `FORWARD` possède une activité tonique et représente le comportement locomoteur par défaut.

---

## Réactions aux obstacles

### Obstacle à droite

```text
S-RIGHT
   ↓
TURN_LEFT
   ↓
inhibition des PM-F gauches
   ↓
virage vers la gauche
```

### Obstacle à gauche

```text
S-LEFT
   ↓
TURN_RIGHT
   ↓
inhibition des PM-F droits
   ↓
virage vers la droite
```

### Obstacle frontal

```text
S-FRONT
   ↓
BACKWARD
   ↓
PM-B favorisés
PM-F inhibés
   ↓
recul
```

Pendant cette phase, `BACKWARD` inhibe fortement `FORWARD`, ce qui permet à la marche arrière de prendre la priorité.

### Stimulation arrière

```text
S-REAR
   ↓
ACCELERATE
   ↓
I_mod augmente dans tous les CPG
   ↓
augmentation globale du rythme locomoteur
```

`ACCELERATE` est le seul comportement de la Partie II qui modifie directement les CPG.

---

# Scénario pseudo-autonome utilisé

Le notebook exécute notamment un scénario de 14 secondes comportant quatre événements successifs :

```text
2–4 s   : obstacle à droite
5–7 s   : obstacle à gauche
8–10 s  : obstacle frontal
11–13 s : stimulation arrière
```

Les obstacles se rapprochent progressivement afin de tester l'influence de leur intensité sur la réponse neuronale.

---

# Mesure de l'intensité du virage

Pour quantifier un virage, on calcule l'asymétrie entre les sorties Forward droite et gauche :

```text
Δf = f_Forward,droit - f_Forward,gauche
```

Pour un obstacle situé à droite :

```text
obstacle plus proche
→ fréquence de S-RIGHT plus élevée
→ TURN_LEFT plus actif
→ inhibition plus forte des PM-F gauches
→ Δf plus grand
→ virage à gauche plus marqué
```

Les résultats montrent également une saturation pour les obstacles très proches : lorsque les prémoteurs Forward du côté intérieur sont déjà fortement inhibés, augmenter encore la stimulation produit peu d'asymétrie supplémentaire.

---

# Généralisation du nombre de pattes

La taille du réseau est contrôlée par un seul paramètre :

```python
n_pairs
```

Exemple :

```python
simulate_locomotion(
    n_pairs=200,
    ...
)
```

correspond à :

```text
200 paires = 400 pattes = 400 CPG
```

Le test à grande échelle montre que l'alternance globale entre les groupes A et B reste visible.

Cependant, avec 200 paires, les CPG d'un même groupe ne sont plus parfaitement synchronisés : des décalages de phase apparaissent sous forme d'ondes dans le raster.

Ce comportement est lié au caractère **local et distribué** du couplage entre CPG.

---

# Principales fonctions

### `leg_names(n_pairs)`

Construit les noms des pattes.

### `tripod_groups(n_pairs)`

Attribue chaque CPG au groupe locomoteur A ou B.

### `target_frequency_to_drive(freq_hz)`

Convertit une fréquence cible en courant de commande du neurone LIF.

### `distance_to_sensor_rate(...)`

Convertit une distance d'obstacle en fréquence sensorielle.

### `build_distance_profile(...)`

Construit les profils temporels des obstacles/stimulations.

### `simulate_locomotion(...)`

Construit et exécute l'ensemble du réseau.

Principaux paramètres :

```python
n_pairs
duration
left_freq
right_freq
left_direction
right_direction
coupling_gain
turn_command
turn_drive
sensor_control
sensor_events
detection_distance_cm
max_sensor_rate_hz
turn_step
accel_step
record_state
```

---

# Figures générées

Les figures sont sauvegardées dans :

```text
results_hexapod/
```

Le notebook génère notamment :

```text
01_forward_cpg.png
02_forward_motor.png
03_backward_motor.png
...
10_auto_distances.png
11_auto_sensor_rates.png
12_auto_sensor_raster.png
13_auto_decision_raster.png
14_auto_premotor_selection.png
14b_auto_premotor_raster.png
15_auto_premotor_turn.png
15b_auto_cpg_acceleration.png
16_auto_cpg_raster.png
17_auto_motor_raster.png
18_distance_vs_turn_left.png
19_distance_vs_motor_turn_strength.png
20_large_network_synchronization.png
```

---

# Limites du modèle

Le modèle comporte plusieurs simplifications importantes :

- les spikes représentent des **instants de commande**, pas la mécanique réelle d'une patte ;
- aucune dynamique de corps, force, angle articulaire ou contact réel avec le sol n'est simulée ;
- les distances des obstacles sont imposées dans le scénario et ne proviennent pas d'un environnement physique 3D ;
- les poids synaptiques et constantes de temps ont été réglés pour obtenir des comportements interprétables ;
- plusieurs stimuli simultanés nécessiteraient une stratégie de décision plus élaborée ;
- avec un très grand nombre de pattes, le couplage local conserve une coordination globale mais ne garantit pas une synchronisation parfaite de tous les CPG d'un même groupe.

---

# Reproductibilité

Pour reproduire les résultats :

1. installer les dépendances ;
2. ouvrir le notebook ;
3. exécuter toutes les cellules dans l'ordre ;
4. ne pas modifier `RANDOM_SEED` si l'on souhaite reproduire les mêmes conditions ;
5. consulter les figures générées dans `results_hexapod/`.

Les paramètres expérimentaux sont indiqués directement dans les cellules correspondant à chaque expérience.

---

## Auteurs

```text
Torres Diego
Natarianni Maxime
```

## Cours

```text
GEI723 — Neurosciences computationnelles et applications en traitement de l'information
Automne 2026
```
