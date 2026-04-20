# SAPGGO — Rapport Technique Complet

> **SAPGGO** (Simulated Articulated Porter with Guided Gait Optimization)
> Agent humanoïde articulé entraîné par renforcement pour porter une charge sur la tête tout en marchant.

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Architecture du workspace](#2-architecture-du-workspace)
3. [Modèle humanoïde MuJoCo](#3-modèle-humanoïde-mujoco)
4. [Environnement RL (`sapggo-env`)](#4-environnement-rl-sapggo-env)
5. [Agent et politique (`sapggo-agent`)](#5-agent-et-politique-sapggo-agent)
6. [Curriculum d'apprentissage (`sapggo-curriculum`)](#6-curriculum-dapprentissage-sapggo-curriculum)
7. [Boucle d'entraînement (`sapggo-train`)](#7-boucle-dentraînement-sapggo-train)
8. [Évaluation (`sapggo-eval`)](#8-évaluation-sapggo-eval)
9. [Visualisation (`sapggo-viz`)](#9-visualisation-sapggo-viz)
10. [Fonction de récompense](#10-fonction-de-récompense)
11. [Hyperparamètres PPO](#11-hyperparamètres-ppo)
12. [Fichiers générés automatiquement](#12-fichiers-générés-automatiquement)
13. [Commandes de lancement](#13-commandes-de-lancement)
14. [Résumé des corrections appliquées](#14-résumé-des-corrections-appliquées)

---

## 1. Vue d'ensemble du projet

SAPGGO est un système de **reinforcement learning** (apprentissage par renforcement) qui entraîne un humanoïde simulé à :

1. **Se tenir debout** avec une charge posée sur la tête.
2. **Maintenir l'équilibre** malgré des perturbations (vent, masse variable).
3. **Marcher en avant** tout en gardant la charge stable.
4. **Parcourir de longues distances** sur des terrains variés.

### Stack technique

| Composant | Technologie |
|-----------|-------------|
| **Langage** | Rust (édition 2021) |
| **Moteur physique** | MuJoCo 3.x via `mujoco-rs` |
| **Algorithme RL** | PPO (Proximal Policy Optimization) |
| **Réseau de neurones** | MLP 2 couches cachées (implémenté à la main avec `ndarray`) |
| **Sérialisation** | `serde` + `serde_json` (checkpoints), `toml` (configs) |
| **Logging** | `tracing` + `tracing-subscriber` |
| **CLI** | `clap` (derive) |

### Organisation Cargo Workspace

```
Cargo.toml                  # Workspace root
├── crates/
│   ├── sapggo-env/         # Environnement MuJoCo + reward + senseurs
│   ├── sapggo-agent/       # Politique MLP + PPO + normalisation
│   ├── sapggo-curriculum/  # Progression par étapes
│   ├── sapggo-train/       # Boucle d'entraînement principale
│   ├── sapggo-eval/        # Pipeline d'évaluation
│   └── sapggo-viz/         # Visualisation 3D
├── assets/
│   └── robot_humanoid_load.xml   # Modèle MuJoCo
├── configs/
│   ├── train_default.toml        # Configuration d'entraînement
│   └── eval.toml                 # Configuration d'évaluation
├── checkpoints/            # Poids sauvegardés (créé automatiquement)
└── runs/                   # Logs CSV (créé automatiquement)
```

---

## 2. Architecture du workspace

### Crate `sapggo-env` — Environnement physique

| Module | Rôle |
|--------|------|
| `robot.rs` | Constantes du robot : `N_JOINTS=24`, `OBS_DIM=92`, `ACT_DIM=24`, noms des joints, couples max |
| `sim.rs` | Abstraction MuJoCo avec double backend (réel + stub sans physique) |
| `environment.rs` | Interface RL `reset()`/`step()`, lissage d'actions, randomisation de domaine |
| `sensor.rs` | Extraction du vecteur d'observation 92-dim depuis la simulation |
| `reward.rs` | Calcul de la récompense dense par pas + bonus/pénalités |
| `load.rs` | Placement de la charge sur la tête et détection de chute |
| `noise.rs` | Bruit gaussien pour randomisation des observations |
| `terrain.rs` | Résolution des fichiers d'assets terrain |
| `error.rs` | Types d'erreurs typés (`EnvError`) |

### Crate `sapggo-agent` — Intelligence artificielle

| Module | Rôle |
|--------|------|
| `policy.rs` | `MlpActor` (politique gaussienne), `LinearLayer`, buffers de gradient, trait `Policy` |
| `value.rs` | `MlpCritic` (fonction de valeur), accumulation de gradients |
| `ppo.rs` | `PpoConfig`, `compute_gae()`, `normalize_advantages()` |
| `rollout.rs` | `RolloutBuffer` et `Transition` pour collecter les trajectoires |
| `normalize.rs` | `RunningNormalizer` (Welford) pour normaliser les observations |

### Crate `sapggo-curriculum` — Apprentissage progressif

| Module | Rôle |
|--------|------|
| `stage.rs` | Enum `CurriculumStage` (Stand → Balance → Walk → Distance → Robust → Master) |
| `params.rs` | `CurriculumParams` : masse, friction, vent, gravité, bruit, vitesse, durée par étape |
| `manager.rs` | `CurriculumManager` : fenêtre glissante de 50 épisodes, promotion automatique |

### Crate `sapggo-train` — Entraînement

| Module | Rôle |
|--------|------|
| `main.rs` | Point d'entrée CLI avec `clap` |
| `config.rs` | `TrainConfig` chargé depuis TOML |
| `trainer.rs` | Boucle principale : collecte → GAE → PPO update → logs → checkpoints |
| `logger.rs` | `TrainingLogger` : CSV `metrics.csv` (créé automatiquement) |
| `episode_log.rs` | `EpisodeLog` : CSV `episodes.csv` (créé automatiquement) |
| `checkpoint.rs` | Sauvegarde/chargement JSON des poids |
| `bin/visual_train.rs` | Entraînement avec visualisation 3D MuJoCo en temps réel |

### Crate `sapggo-eval` — Évaluation

| Module | Rôle |
|--------|------|
| `main.rs` | CLI d'évaluation, charge une politique et lance des épisodes |
| `evaluator.rs` | `run_evaluation()` : épisodes déterministes, agrège les métriques |

---

## 3. Modèle humanoïde MuJoCo

**Fichier** : `assets/robot_humanoid_load.xml`

### Paramètres de simulation

| Paramètre | Valeur |
|-----------|--------|
| Timestep | 0.002 s (2 ms) |
| Intégrateur | RK4 |
| Gravité | (0, 0, -9.81) m/s² |
| Modèle de contact | Cône elliptique |
| Solveur | Newton, 50 itérations |

### Anatomie du robot (24 joints articulés)

```
root_body (pos 0 0 1.06, freejoint = 6 DOF)
│   mass: 0.01 kg (connecteur)
│
├── torso (2 hinges: torso_flex, torso_lat)
│   mass: 25.0 kg, capsule r=0.07 L=0.50
│   │
│   ├── neck (2 hinges: neck_tilt, neck_rot)
│   │   mass: 1.5 kg, capsule r=0.03 L=0.14
│   │   │
│   │   └── head (pas de joint propre)
│   │       mass: 4.5 kg, sphère r=0.11
│   │       + head_top: sphère r=0.06 (surface de contact pour la charge)
│   │
│   ├── upper_arm_L (3 hinges: shoulder_flex/add/rot)
│   │   mass: 2.5 kg, capsule r=0.04 L=0.28
│   │   └── lower_arm_L (1 hinge: elbow_flex)
│   │       mass: 1.5 kg, capsule r=0.035 L=0.24
│   │
│   └── upper_arm_R (3 hinges: shoulder_flex/add/rot)
│       mass: 2.5 kg, capsule r=0.04 L=0.28
│       └── lower_arm_R (1 hinge: elbow_flex)
│           mass: 1.5 kg, capsule r=0.035 L=0.24
│
├── upper_leg_L (3 hinges: hip_flex/add/rot)
│   mass: 7.0 kg, capsule r=0.05 L=0.40
│   └── lower_leg_L (1 hinge: knee)
│       mass: 4.5 kg, capsule r=0.04 L=0.36
│       └── foot_L (2 hinges: ankle_flex/inv)
│           mass: 1.5 kg, boîte 0.20×0.10×0.04
│
└── upper_leg_R (miroir de la jambe gauche)
    mass identiques

load (freejoint, pos initiale 0 0 1.65)
    mass: 5.0 kg, boîte 0.30×0.30×0.20
```

### Masse totale du corps

| Partie | Masse (kg) |
|--------|-----------|
| root_body | 0.01 |
| Torso | 25.0 |
| Cou | 1.5 |
| Tête | 4.5 |
| Bras gauche (haut + bas) | 4.0 |
| Bras droit (haut + bas) | 4.0 |
| Jambe gauche (haut + bas + pied) | 13.0 |
| Jambe droite (haut + bas + pied) | 13.0 |
| **Total corps** | **~65 kg** |
| Charge (variable) | 2–20 kg |

### Liste des 24 joints articulés

| # | Nom | Type | Axe | Plage (rad) | Couple max (Nm) |
|---|-----|------|-----|-------------|----------------|
| 0 | hip_flex_L | hinge | X | [-0.7, 1.2] | 150 |
| 1 | hip_add_L | hinge | Z | [-0.5, 0.5] | 150 |
| 2 | hip_rot_L | hinge | Y | [-0.5, 0.5] | 150 |
| 3 | knee_L | hinge | X | [-1.8, 0.0] | 200 |
| 4 | ankle_flex_L | hinge | X | [-0.7, 0.5] | 80 |
| 5 | ankle_inv_L | hinge | Z | [-0.4, 0.4] | 80 |
| 6 | hip_flex_R | hinge | X | [-0.7, 1.2] | 150 |
| 7 | hip_add_R | hinge | Z | [-0.5, 0.5] | 150 |
| 8 | hip_rot_R | hinge | Y | [-0.5, 0.5] | 150 |
| 9 | knee_R | hinge | X | [-1.8, 0.0] | 200 |
| 10 | ankle_flex_R | hinge | X | [-0.7, 0.5] | 80 |
| 11 | ankle_inv_R | hinge | Z | [-0.4, 0.4] | 80 |
| 12 | torso_flex | hinge | X | [-0.5, 0.5] | 120 |
| 13 | torso_lat | hinge | Y | [-0.3, 0.3] | 120 |
| 14 | neck_tilt | hinge | X | [-0.4, 0.4] | 40 |
| 15 | neck_rot | hinge | Z | [-0.6, 0.6] | 40 |
| 16 | shoulder_flex_L | hinge | X | [-1.5, 3.14] | 80 |
| 17 | shoulder_add_L | hinge | Z | [-0.5, 1.5] | 80 |
| 18 | shoulder_rot_L | hinge | Y | [-1.0, 1.0] | 60 |
| 19 | elbow_flex_L | hinge | X | [-2.3, 0.0] | 60 |
| 20 | shoulder_flex_R | hinge | X | [-1.5, 3.14] | 80 |
| 21 | shoulder_add_R | hinge | Z | [-0.5, 1.5] | 80 |
| 22 | shoulder_rot_R | hinge | Y | [-1.0, 1.0] | 60 |
| 23 | elbow_flex_R | hinge | X | [-2.3, 0.0] | 60 |

### Capteurs embarqués

| Capteur | Type MuJoCo | Valeurs |
|---------|-------------|---------|
| torso_quat | framequat | 4 (quaternion w,x,y,z) |
| torso_gyro | gyro | 3 (vitesse angulaire) |
| torso_acc | accelerometer | 3 |
| head_quat | framequat | 4 |
| load_pos | framepos | 3 |
| load_quat | framequat | 4 |
| load_gyro | gyro | 3 |
| load_acc | accelerometer | 3 |
| foot_L_force | force | 3 (Fx, Fy, Fz) |
| foot_R_force | force | 3 (Fx, Fy, Fz) |

---

## 4. Environnement RL (`sapggo-env`)

### Interface `SapggoEnv`

L'environnement implémente le cycle standard RL :

```
obs₀ = env.reset()
loop {
    action = agent.act(obs)
    (obs, reward, done, info) = env.step(action)
    if done { obs = env.reset() }
}
```

### Procédure `reset()`

1. **`sim.reset_data()`** — Remise à zéro de toutes les positions et vitesses.
2. **`apply_domain_randomization()`** — Applique la gravité scalée et le vent aléatoire.
3. **`place_load_on_head()`** — Place la charge sur la tête avec offset XY aléatoire (±1.5 cm).
4. **200 pas de simulation passifs** — La charge se stabilise naturellement sur la tête (0.4 s physique).
5. **Retourne l'observation initiale** (92 dimensions).

### Procédure `step(action)`

1. **Lissage d'action** (filtre passe-bas) :
   ```
   smoothed[i] = 0.8 × smoothed[i] + 0.2 × action[i] × MAX_TORQUE[i]
   clamped = clamp(smoothed, -MAX_TORQUE, +MAX_TORQUE)
   sim.set_ctrl(i, clamped)
   ```
2. **10 pas de simulation physique** (= 20 ms de temps réel par step de contrôle).
3. **Accumulation de la distance parcourue** (uniquement vers l'avant, `max(0)`).
4. **Vérification de terminaison** :
   - **Charge tombée** : `load_z - head_z < -0.05 m`
   - **Robot tombé** : `torso_z < 0.5 m`
   - **Timeout** : `steps >= max_steps` (défini par le curriculum)
5. **Calcul de la récompense dense** + bonus de jalons (tous les 10 m) + pénalité de terminaison.
6. **Extraction de l'observation** (92 dimensions).

### Vecteur d'observation (92 dimensions)

| Indices | Contenu | Dimensions |
|---------|---------|-----------|
| [0..24] | Angles articulaires | 24 |
| [24..48] | Vitesses articulaires | 24 |
| [48..52] | Quaternion du torse (w, x, y, z) | 4 |
| [52..55] | Gyroscope du torse (ωx, ωy, ωz) | 3 |
| [55..58] | Forces de contact pied gauche (Fx, Fy, Fz) | 3 |
| [58..61] | Forces de contact pied droit (Fx, Fy, Fz) | 3 |
| [61..64] | Offset charge/tête (dx, dy, dz) | 3 |
| [64..67] | Vitesse angulaire de la charge | 3 |
| [67..91] | Action précédente | 24 |
| [91] | Vitesse cible (curriculum) | 1 |

### Randomisation de domaine

À chaque épisode, les paramètres physiques sont randomisés selon le stade du curriculum :

- **Gravité** : scalée par `gravity_scale` (0.5× au stade Stand, 1.0× après).
- **Vent** : force latérale aléatoire sur la charge (`[-wind_max, +wind_max]` N).
- **Position de la charge** : offset XY aléatoire ±1.5 cm.
- **Bruit d'observation** : bruit gaussien ajouté à chaque capteur.

### Gestion de la charge

- **Placement** : au-dessus de la tête (`head_z + 0.22 m`), quaternion identité.
- **Détection de chute** : si la charge descend > 5 cm sous la tête.
- **Surface de contact** : sphère `head_top` avec friction élevée (0.9).

---

## 5. Agent et politique (`sapggo-agent`)

### Architecture du réseau Actor (MlpActor)

```
Observation (92 dim)
    │
    ▼
Linear(92 → 256) + Tanh       ← Xavier uniform init
    │
    ▼
Linear(256 → 256) + Tanh      ← Xavier uniform init
    │
    ├──→ Linear(256 → 24)      → mean (μ)
    │
    └──→ log_std (vecteur appris, 24 dim, init = -0.5)
         clamp [-4.0, 0.0] → exp → std (σ)
```

**Sortie** : Distribution gaussienne diagonale N(μ, σ²) pour chaque articulation.

**Échantillonnage stochastique** :
```
a[i] ~ N(mean[i], std[i]²)
log π(a|s) = Σ [-0.5 × ((a-μ)/σ)² - ln(σ) - 0.5 × ln(2π)]
```

**Action déterministe** : `a = mean` (utilisé en évaluation).

### Architecture du réseau Critic (MlpCritic)

```
Observation (92 dim)
    │
    ▼
Linear(92 → 256) + Tanh
    │
    ▼
Linear(256 → 256) + Tanh
    │
    ▼
Linear(256 → 1)              → V(s) (scalaire)
```

### Rétropropagation et accumulation de gradient

La mise à jour PPO utilise l'**accumulation de gradient par batch** (pas de SGD par échantillon) :

1. **Phase d'accumulation** : Pour chaque transition du rollout, on calcule le gradient mais on ne modifie pas les poids. Les gradients sont stockés dans des buffers (`ActorGradBuffer`, `CriticGradBuffer`).

2. **Phase d'application** : Le gradient moyen (divisé par N) est clampé à ±1.0 par composante puis appliqué en un seul pas SGD.

```
Pour chaque époque:
    grad.reset()
    Pour chaque transition i:
        cache = actor.forward_with_cache(obs[i])
        calcul ratio, surrogate, gradient
        actor.accumulate_grad(cache, d_mean, d_log_std, grad)
        critic.accumulate_grad(obs, return[i], critic_grad)
    actor.apply_grad(grad, N, lr)
    critic.apply_grad(critic_grad, N, lr)
```

### Normalisation des observations (RunningNormalizer)

Algorithme de **Welford** pour calculer la moyenne et la variance en ligne :

```
Pour chaque observation :
    count += 1
    delta = obs[i] - mean[i]
    mean[i] += delta / count
    delta2 = obs[i] - mean[i]
    var[i] += delta × delta2

Normalisation :
    obs_norm[i] = (obs[i] - mean[i]) / sqrt(var[i]/(count-1) + ε)
```

Cela stabilise l'entraînement en maintenant les entrées du réseau autour de zéro avec une variance unitaire.

---

## 6. Curriculum d'apprentissage (`sapggo-curriculum`)

### Stades de progression

| # | Stade | Objectif | max_steps | Seuil promotion | Gravité | Vent max | Charge (kg) | Bruit obs | Vitesse cible |
|---|-------|----------|-----------|-----------------|---------|----------|-------------|-----------|---------------|
| 0 | **Stand** | Rester debout | 250 | 15.0 | 0.5× | 0 N | 2 | 0 | 0 m/s |
| 1 | **Balance** | Équilibrer la charge | 500 | 25.0 | 1.0× | 0 N | 2–5 | 0 | 0 m/s |
| 2 | **Walk** | Commencer à marcher | 600 | 50.0 | 1.0× | 2 N | 2–7 | 0.002 | 0.8 m/s |
| 3 | **Distance** | Parcourir plus loin | 800 | 120.0 | 1.0× | 5 N | 5–10 | 0.005 | 1.0 m/s |
| 4 | **Robust** | Résister aux perturbations | 1000 | 250.0 | 1.0× | 10 N | 5–15 | 0.008 | 1.0 m/s |
| 5 | **Master** | Maîtrise complète | 1000 | ∞ | 1.0× | 15 N | 10–20 | 0.01 | 1.2 m/s |

### Mécanisme de promotion

Le `CurriculumManager` maintient une **fenêtre glissante de 50 épisodes**. Quand la récompense moyenne dépasse le seuil de promotion :

1. L'agent passe au stade suivant.
2. La fenêtre est vidée.
3. Les paramètres de l'environnement sont mis à jour (`env.set_params()`).

### Stratégie de difficulté progressive

- **Stand** : Gravité réduite à 50%, pas de vent, charge légère fixe, épisodes courts → l'agent apprend d'abord à se tenir debout.
- **Balance** : Gravité normale, charge variable → l'agent stabilise la charge.
- **Walk** : Vent léger, bruit de capteur, vitesse cible → l'agent commence à marcher.
- **Distance/Robust/Master** : Conditions de plus en plus dures → robustesse.

---

## 7. Boucle d'entraînement (`sapggo-train`)

### Processus principal (`Trainer::run()`)

```
Initialisation:
    1. Charger le modèle MuJoCo
    2. Créer actor MLP (92 → 256 → 256 → 24)
    3. Créer critic MLP (92 → 256 → 256 → 1)
    4. Créer normalizer, curriculum manager
    5. Créer fichiers CSV (runs/metrics.csv, runs/episodes.csv)
    6. Créer dossier checkpoints/

Boucle principale (tant que global_step < total_steps):
    1. COLLECTE — Accumuler 4096 transitions :
       - Normaliser l'observation
       - Échantillonner action stochastique (actor)
       - Estimer la valeur (critic)
       - Exécuter env.step(action)
       - Si épisode terminé → logger, vérifier promotion curriculum
    
    2. AVANTAGES — Calculer GAE (Generalized Advantage Estimation) :
       - δ_t = r_t + γ × V(s_{t+1}) × mask − V(s_t)
       - A_t = Σ (γλ)^k × δ_{t+k}
       - R_t = A_t + V(s_t)
       - Normaliser les avantages (zero mean, unit variance)
    
    3. MISE À JOUR PPO (6 époques) :
       - Pour chaque époque :
         a. Accumuler les gradients sur toutes les 4096 transitions
         b. Appliquer le gradient moyen (clamp ±1.0) en un pas
       - Actor : objectif clippé + bonus d'entropie
       - Critic : perte MSE sur les returns
    
    4. LOGGING — Écrire dans metrics.csv :
       - train/reward_mean, train/policy_loss, train/value_loss
       - train/entropy, train/best_reward, curriculum/stage
    
    5. CHECKPOINT — Sauvegarder les poids tous les 100 000 pas
       - checkpoints/sapggo_step_{N}.bin
       - checkpoints/best_actor.json (meilleur épisode)
       - checkpoints/best_critic.json
```

### Objectif PPO clippé

Pour chaque transition :

```
ratio = exp(log π_new(a|s) − log π_old(a|s))
surr1 = ratio × A(s,a)
surr2 = clip(ratio, 1−ε, 1+ε) × A(s,a)
L_policy = −min(surr1, surr2)
```

Le clipping (ε = 0.2) empêche les mises à jour trop agressives de la politique.

### Bonus d'entropie

```
H = 0.5 × ln(2πeσ²) par dimension
L_total = L_policy − entropy_coef × H
```

Encourage l'exploration en pénalisant les distributions trop concentrées.

---

## 8. Évaluation (`sapggo-eval`)

### Processus

1. Charge la politique depuis un checkpoint JSON.
2. Exécute N épisodes en mode **déterministe** (action = mean, pas d'exploration).
3. Agrège les métriques :
   - **Distance moyenne** (m)
   - **Pas moyens** par épisode
   - **Taux de chute** de la charge (%)
   - **Vitesse moyenne** (m/s)

### Commande

```bash
cargo run --release --bin sapggo-eval -- --config configs/eval.toml
```

---

## 9. Visualisation (`sapggo-viz`)

### Mode entraînement visuel

Le binaire `sapggo-visual-train` lance l'entraînement PPO **et** un viewer 3D MuJoCo simultanément :

- Synchronise les `qpos`/`qvel` de la simulation vers le viewer à chaque pas.
- Limité à ~60 Hz pour le rendu.
- Même logique PPO que l'entraîneur principal.

### Synchronisation

```rust
// Copie l'état physique vers le viewer
unsafe {
    copy_nonoverlapping(qpos_src, viz_data.qpos, nq);
    copy_nonoverlapping(qvel_src, viz_data.qvel, nv);
}
viz_data.forward();  // Recalcule les positions cartésiennes
viewer.sync();       // Met à jour l'affichage
```

---

## 10. Fonction de récompense

### Récompense dense par pas

```
r = w_vel   × velocity_x          (tracking de vitesse : +)
  − w_pitch × |pitch|             (pénalité inclinaison avant : −)
  − w_roll  × |roll|              (pénalité inclinaison latérale : −)
  − w_energy × mean(τ²)           (pénalité énergie : −)
  − w_load_x × |load_dx|          (pénalité offset X charge : −)
  − w_load_y × |load_dy|          (pénalité offset Y charge : −)
  − w_load_z × max(−load_dz, 0)   (pénalité chute charge : −)
  − w_jerk   × mean(Δa²)          (pénalité jerk action : −)
  + w_alive                        (bonus survie : +)
```

### Poids par défaut

| Composante | Poids | Rôle |
|-----------|-------|------|
| `vel` | 2.0 | Encourage la marche vers l'avant |
| `pitch` | 0.3 | Penalise l'inclinaison avant/arrière du torse |
| `roll` | 0.2 | Penalise l'inclinaison latérale du torse |
| `energy` | 0.001 | Penalise les couples élevés (économie d'énergie) |
| `load_x` | 1.0 | Penalise le décalage X de la charge |
| `load_y` | 1.0 | Penalise le décalage Y de la charge |
| `load_z` | 2.0 | Penalise la chute de la charge |
| `jerk` | 0.05 | Penalise les changements brusques d'action |
| `alive` | 1.0 | Bonus de survie par pas |

### Extraction pitch/roll (convention ZYX Euler)

```
pitch = asin(clamp(2(qw×qy − qz×qx), −1, 1))
roll  = atan2(2(qw×qx + qy×qz), 1 − 2(qx² + qy²))
```

### Récompenses sparse

| Événement | Valeur |
|-----------|--------|
| Jalon tous les 10 m | +25.0 |
| Complétion 1 km | +100.0 |
| Terminaison (chute/perte de charge) | −10.0 |

---

## 11. Hyperparamètres PPO

### Configuration actuelle (`configs/train_default.toml`)

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `lr` | 0.003 | Taux d'apprentissage |
| `epochs` | 6 | Époques PPO par rollout |
| `rollout_steps` | 4096 | Transitions par rollout |
| `clip_epsilon` | 0.2 | Plage de clipping PPO |
| `gamma` | 0.99 | Facteur d'actualisation |
| `gae_lambda` | 0.95 | Lambda GAE (compromis biais/variance) |
| `entropy_coef` | 0.01 | Coefficient bonus entropie |
| `value_coef` | 0.5 | Coefficient perte de valeur |
| `max_grad_norm` | 0.5 | Norme max du gradient |
| `total_steps` | 20 000 000 | Pas d'entraînement total |
| `checkpoint_interval` | 100 000 | Sauvegarde tous les N pas |

### Lissage d'action

```
alpha = 0.8  (poids de la commande précédente)
beta  = 0.2  (poids de la nouvelle action)
smoothed[i] = alpha × smoothed[i] + beta × action[i] × MAX_TORQUE[i]
```

---

## 12. Fichiers générés automatiquement

Tous les fichiers suivants sont **créés automatiquement au lancement de l'entraînement** (pas besoin de les créer manuellement) :

| Fichier | Créé par | Contenu |
|---------|---------|---------|
| `runs/metrics.csv` | `TrainingLogger::new()` | `tag,value,global_step` — métriques par rollout |
| `runs/episodes.csv` | `EpisodeLog::new()` | `episode,global_step,steps,reward,distance_m,load_dropped,robot_fallen,best_reward,is_best,curriculum_stage` |
| `checkpoints/` (dossier) | `ensure_checkpoint_dir()` | Créé si absent |
| `checkpoints/best_actor.json` | Trainer, à chaque nouveau record | Poids du meilleur actor |
| `checkpoints/best_critic.json` | Trainer, à chaque nouveau record | Poids du meilleur critic |
| `checkpoints/sapggo_step_{N}.bin` | Trainer, tous les 100k pas | Checkpoint périodique |
| `checkpoints/sapggo_final.bin` | Trainer, en fin d'entraînement | Checkpoint final |

### Mécanisme de création automatique

- **`TrainingLogger::new(log_dir)`** : appelle `fs::create_dir_all(log_dir)` → crée `runs/` s'il n'existe pas, puis ouvre `metrics.csv` en mode append (et écrit l'en-tête CSV si le fichier est nouveau).
- **`EpisodeLog::new(log_dir)`** : même logique, crée `episodes.csv` avec en-tête.
- **`ensure_checkpoint_dir(dir)`** : appelle `fs::create_dir_all(dir)` pour `checkpoints/`.
- Les fichiers JSON de checkpoint sont écrits par `save_checkpoint()` à chaque meilleur épisode et à intervalles réguliers.

---

## 13. Commandes de lancement

### Entraînement standard (sans visualisation)

```bash
cargo run --release --bin sapggo-train -- --config configs/train_default.toml
```

### Entraînement avec visualisation 3D (nécessite feature `visual` et MuJoCo installé)

```bash
cargo run --release --bin sapggo-visual-train --features visual -- --config configs/train_default.toml
```

### Évaluation d'une politique entraînée

```bash
cargo run --release --bin sapggo-eval -- --config configs/eval.toml
```

### Options CLI supplémentaires

```bash
# Changer la seed aléatoire
cargo run --release --bin sapggo-train -- --config configs/train_default.toml --seed 123

# Reprendre depuis un checkpoint
cargo run --release --bin sapggo-train -- --config configs/train_default.toml --resume checkpoints/best_actor.json
```

### Vérification rapide (compilation seule)

```bash
cargo build --release
```

---

## 14. Résumé des corrections appliquées

### Bugs critiques corrigés

| # | Bug | Impact | Correction |
|---|-----|--------|-----------|
| 1 | **Hauteur initiale trop élevée** (root z=1.35, pieds à 29 cm du sol) | Robot tombait à chaque épisode avant de pouvoir apprendre | root_body z: 1.35 → 1.06 |
| 2 | **Overflow capteur pied** (lecture de 4 valeurs au lieu de 3) | Observations corrompues, apprentissage impossible | Boucle `0..4` → `0..3` |
| 3 | **Pitch/roll inversés** dans la récompense | Pénalités de posture appliquées au mauvais angle | Formules corrigées (convention ZYX) |
| 4 | **Pas de bras** dans le modèle humanoïde | Modèle irréaliste, pas de contrepoids pour l'équilibre | Ajout de 4 corps + 8 joints + 8 actuateurs |
| 5 | **Masse trop légère** (~42 kg) | Dynamique irréaliste | Torso 10 → 25 kg (total ~65 kg) |
| 6 | **`gravity_scale` jamais appliqué** | Randomisation de domaine incomplète | Ajout `set_gravity_z()` dans `apply_domain_randomization()` |
| 7 | **Bug `max_steps`** (`.max(1000)` ignorait le curriculum) | Stand stage à 1000 pas au lieu de 250, signal d'apprentissage dilué | Logique conditionnelle : utilise curriculum si > 0 |
| 8 | **Seuil de chute** inadapté (0.6 m pour root à 1.06) | Détection de chute trop/pas assez sensible | Ajusté à 0.5 m |
| 9 | **Pas de stabilisation passive** suffisante (50 steps) | Charge pas encore posée quand l'épisode commence | 50 → 200 pas (0.4 s physique) |

### Changements de dimensions

| Constante | Avant | Après |
|-----------|-------|-------|
| `N_JOINTS` | 16 | 24 |
| `ACT_DIM` | 16 | 24 |
| `OBS_DIM` | 70 | 92 |

---

## Résumé

SAPGGO est un projet de reinforcement learning complet en Rust qui combine :

- Un **modèle humanoïde réaliste** à 24 degrés de liberté (~65 kg) simulé dans MuJoCo.
- Un **environnement RL** avec randomisation de domaine, lissage d'action et shaping de récompense.
- Un **agent PPO** avec accumulation de gradient par batch, politique gaussienne et critique MLP.
- Un **curriculum d'apprentissage** en 6 stades de difficulté croissante.
- Un **pipeline de logging automatique** (CSV) et de checkpointing (JSON).

Tous les fichiers de sortie (CSV, checkpoints, dossiers) sont créés **automatiquement** au lancement de l'entraînement.
