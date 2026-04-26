![WODAABE (MBORORO) PEOPLE_ THE NOMADIC FULANI SUB-TRIBE THAT CULTIVATE BEAUTY AND THEIR UNIQUE GEREWOL FESTIVAL](https://github.com/user-attachments/assets/7295d311-8ae3-4508-b276-065d048ba3bd)


<div align="center">

# SAPGGO

### Stable Adaptive Policy for Gait and Ground-Object Balance

**Version 0.1.0** &nbsp;·&nbsp; MIT License &nbsp;·&nbsp; Rust 1.85+

*Teaching an agent to walk in perfect balance with a head-carried load —*
*inspired by the ancient tradition of African traders and mothers.*

---

**Simulate &rarr; Learn &rarr; Balance &rarr; Walk**

---

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Language-Rust%201.85+-orange.svg)](https://www.rust-lang.org)
[![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo%203.2+-blue.svg)](https://mujoco.org)
[![Status](https://img.shields.io/badge/Status-Research%20Alpha-red.svg)]()

</div>

---

## The Vision

For thousands of years, across the entire African continent, traders, farmers, and mothers have carried heavy loads on their heads — baskets of fruit, water jugs, bundles of wood — walking long distances over uneven terrain, upright, efficient, and balanced. No harness. No hands. Pure mastery of equilibrium earned through years of practice.

**SAPGGO** is a reinforcement learning project that teaches a bipedal agent to reproduce this extraordinary skill inside a physics simulation. The agent must learn to walk indefinitely, on varied terrain, under external perturbations, with a load balanced freely on its head — learning from scratch, through trial, error, and reward, exactly as humans do.

This is not only an engineering challenge. It is a tribute to ancient human knowledge, translated into the language of modern artificial intelligence.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why This Problem Matters](#2-why-this-problem-matters)
3. [Architecture](#3-architecture)
4. [Environment Design](#4-environment-design)
5. [Observation and Action Spaces](#5-observation-and-action-spaces)
6. [Reward Function](#6-reward-function)
7. [Training Strategy](#7-training-strategy)
8. [Curriculum Learning](#8-curriculum-learning)
9. [Domain Randomization](#9-domain-randomization)
10. [Tools and Technologies](#10-tools-and-technologies)
11. [Installation](#11-installation)
12. [Usage](#12-usage)
13. [Expected Results](#13-expected-results)
14. [Roadmap](#14-roadmap)
15. [Contributing](#15-contributing)
16. [License and Credits](#16-license-and-credits)

---

## 1. Project Overview

**SAPGGO** (Stable Adaptive Policy for Gait and Ground-Object Balance) is a reinforcement learning framework built entirely in Rust, using the MuJoCo physics engine, designed to train a humanoid agent to:

- Walk stably on flat, rolling, and rough terrain.
- Maintain a freely-resting load balanced on its head without dropping it.
- Generalize to unseen load weights, terrain slopes, wind forces, and surface textures.
- Cover long distances (1 km+ simulated) continuously and without failure.

The agent is trained using **Proximal Policy Optimization (PPO)**, guided by a carefully shaped reward signal and a progressive difficulty curriculum, under full domain randomization for robustness.

**Core design principles:**

- **Physics first** — realism comes from the simulation, not from graphics.
- **Sensor-based learning** — the agent uses vector observations (joints, IMU, contact forces), not pixels.
- **Curriculum-driven progression** — difficulty increases automatically as the agent improves.
- **Rust throughout** — performance, safety, and determinism from the ground up.

---

## 2. Why This Problem Matters

### A Scientific Challenge

Head-load balancing during bipedal locomotion is one of the most demanding motor control problems in robotics and AI. It uniquely combines:

- **Multi-joint dynamic control** — 24 joints must coordinate simultaneously while an unsecured object rests on the head.
- **Unstable free-body dynamics** — the load is not fixed; it can slide, tilt, and fall at any time.
- **Delayed consequences** — a small imbalance may not cause failure for several steps, creating a hard temporal credit assignment problem.
- **Multi-objective optimization** — walk fast, walk efficiently, stay upright, and keep the load centered, all at once.
- **Robustness requirements** — wind, terrain variation, and unpredictable load behavior must all be handled gracefully.

These properties make SAPGGO a meaningful and non-trivial benchmark for robust bipedal locomotion research.

### Real-World Applications

A policy trained in SAPGGO has direct industrial and social applications:

| Application | Description |
|-------------|-------------|
| **Delivery robotics** | Autonomous bipedal robots transporting goods in difficult terrain — rural zones, disaster areas, stairways without elevators. |
| **Exoskeleton assistance** | Posture correction guidance and motor support for workers carrying heavy loads over long distances. |
| **Carrier training simulation** | A virtual tool to help human workers optimize their gait, reduce fatigue, and prevent injury. |
| **Authentic character animation** | Generating physically correct animations for cinema and video games depicting head-carrying characters. |

### A Cultural Tribute

The head-carrying tradition is not just a technique — it is a refined form of embodied intelligence, passed down across generations, representing an extraordinary adaptation to the demands of daily life. By modeling it in AI, SAPGGO seeks to preserve and honor this knowledge, and to understand, from a biomechanical and computational perspective, why this technique is superior to shoulder or back carrying for long-distance transport.

---

## 3. Architecture

```
+------------------------------------------------------------------+
|                        SAPGGO Framework                          |
|                                                                    |
|  +---------------+    actions: Vec<f64>   +--------------------+  |
|  |  PPO Agent    | ---------------------->|  SAPGGO Environ.   |  |
|  | (Policy NN)   |                        | (Rust + mujoco-rs) |  |
|  |               | <----------------------|                    |  |
|  +---------------+  obs: Vec<f64>         |  MuJoCo XML model  |  |
|                      reward: f64          |  Sensor extraction |  |
|                      done: bool           |  Reward shaping    |  |
|                                           |  Curriculum logic  |  |
|                                           +--------+-----------+  |
|                                                    |               |
|                                           +--------v-----------+  |
|                                           |   MuJoCo Physics   |  |
|                                           |  Rigid body dyn.   |  |
|                                           |  Contact solver    |  |
|                                           |  Heightfield terr. |  |
|                                           +--------------------+  |
+------------------------------------------------------------------+
```

### Project Structure

```
sapggo/
+-- Cargo.toml                     # Workspace root
+-- crates/
|   +-- sapggo-env/                # MuJoCo environment (reset, step, reward)
|   +-- sapggo-agent/              # PPO policy and value network
|   +-- sapggo-train/              # Training loop, logging, checkpoints
|   +-- sapggo-eval/               # Evaluation and benchmark runner
|   +-- sapggo-curriculum/         # Automatic difficulty progression
|   +-- sapggo-viz/                # Visualization helpers
+-- assets/
|   +-- robot_humanoid_load.xml    # MuJoCo humanoid + free load
|   +-- terrain_flat.png           # Flat terrain heightmap
|   +-- terrain_rolling.png        # Rolling terrain heightmap
|   +-- terrain_rough.png          # Rough terrain heightmap
+-- configs/
|   +-- train_default.toml         # Default PPO training config
|   +-- train_curriculum.toml      # Curriculum training config
|   +-- eval.toml                  # Evaluation config
+-- examples/
|   +-- random_agent.rs            # Random policy baseline
|   +-- manual_control.rs          # Keyboard-controlled demo
|   +-- trained_demo.rs            # Load and run a saved policy
+-- checkpoints/                   # Saved policy weights
+-- runs/                          # Training logs
+-- demo/                          # Generated evaluation videos
```

---

## 4. Environment Design

### 4.1 The Robot Model

The agent is a **24-degree-of-freedom** humanoid (~65 kg) defined in MuJoCo XML. The skeleton includes legs (6 DoF each), arms (4 DoF each), a torso (2 DoF), and a neck (2 DoF).

```
Joint map (24 actuated degrees of freedom)
----------------------------------------------
LEFT LEG           RIGHT LEG          TORSO/NECK
hip_flex_L   [0]   hip_flex_R   [6]   torso_flex  [12]
hip_add_L    [1]   hip_add_R    [7]   torso_lat   [13]
hip_rot_L    [2]   hip_rot_R    [8]   neck_tilt   [14]
knee_L       [3]   knee_R       [9]   neck_rot    [15]
ankle_flex_L [4]   ankle_flex_R [10]
ankle_inv_L  [5]   ankle_inv_R  [11]
shoulder_flex_L  [16]   shoulder_flex_R  [20]
shoulder_add_L   [17]   shoulder_add_R   [21]
shoulder_rot_L   [18]   shoulder_rot_R   [22]
elbow_flex_L     [19]   elbow_flex_R     [23]
----------------------------------------------
Total: 24 actuated joints
```

### 4.2 The Load

The load is a rigid body placed on the robot's head at the beginning of each episode. It rests on the head through contact forces alone — no constraint, no attachment — exactly replicating the real practice.

Key contact parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Head-load friction | 0.6–1.2 (randomized) | Simulates cloth, leather, or bare skin |
| Restitution (bounce) | 0.05 | Nearly no bounce — realistic soft contact |
| Load damping joint | 0.05 | Minimal, almost free body |
| Initial placement | 0 ± 15 mm offset | Randomized for robustness |

### 4.3 Episode Initialization Sequence

```
reset()
 |
 +-- 1. Reload MuJoCo model, zero all velocities
 +-- 2. Set robot to upright standing pose
 +-- 3. Apply domain randomization (gravity scale, wind)
 +-- 4. Place load on head (with small random offset)
 +-- 5. Run 200 passive simulation steps (0.4 s)
 |       -> load settles under gravity
 |       -> robot absorbs passive weight
 +-- 6. Return first observation (stable starting state)
```

This passive settling phase is critical. It ensures the agent always starts from a physically consistent state where the load is already resting, not falling.

### 4.4 Terrain

Three terrain types are available and introduced progressively via the curriculum:

| Terrain | Description | Slope | When Introduced |
|---------|-------------|-------|-----------------|
| Flat | Checkered floor, perfectly level | 0deg | Stage 0 |
| Rolling | Gentle bumps (max +-5 cm height variation) | 0–3deg | Stage 3 |
| Rough | Rocky, irregular, variable slope | 0–10deg | Stage 4 |

Terrain is loaded from heightmap PNG images into MuJoCo's native `hfield` element. Terrain changes every N episodes during training.

### 4.5 Visual Environment

During human observation (not used by the agent), the environment renders:

- Textured ground with checkered or dirt pattern.
- Directional sunlight with dynamic shadows.
- Gradient sky (blue to dark horizon).
- Dust particles rising from foot contacts (intensity proportional to walking speed).
- Camera modes: side view (default), front view, free orbit.

---

## 5. Observation and Action Spaces

### 5.1 Observation Space (98 dimensions, continuous)

| Component | Dims | Indices | Source |
|-----------|------|---------|--------|
| Joint angles | 24 | [0..24] | `data.qpos` (actuated joints only) |
| Joint velocities | 24 | [24..48] | `data.qvel` |
| Torso quaternion | 4 | [48..52] | `data.xquat[torso_body]` |
| Torso angular velocity | 3 | [52..55] | Simulated gyroscope on torso |
| Torso linear acceleration | 3 | [55..58] | Torso accelerometer |
| Foot L contact forces | 3 | [58..61] | `foot_L_force` sensor (Fx, Fy, Fz) |
| Foot R contact forces | 3 | [61..64] | `foot_R_force` sensor (Fx, Fy, Fz) |
| Load position offset | 3 | [64..67] | Load center relative to head center |
| Load angular velocity | 3 | [67..70] | `load_gyro` sensor |
| Load linear acceleration | 3 | [70..73] | `load_acc` sensor |
| Previous action | 24 | [73..97] | Last applied torque command (normalized) |
| Target velocity | 1 | [97] | Current curriculum forward speed target |
| **Total** | **98** | |

Gaussian noise with sigma = 0–0.01 is added to all sensor readings during training for robustness.

### 5.2 Action Space (24 dimensions, continuous, range [-1, 1])

| Group | Count | Max Torque |
|-------|-------|----------|
| Hips (3 DoF x 2 legs) | 6 | 150 Nm |
| Knees (1 DoF x 2 legs) | 2 | 200 Nm |
| Ankles (2 DoF x 2 legs) | 4 | 80 Nm |
| Torso (2 DoF) | 2 | 120 Nm |
| Neck (2 DoF) | 2 | 40 Nm |
| Shoulders (3 DoF x 2 arms) | 6 | 60–80 Nm |
| Elbows (1 DoF x 2 arms) | 2 | 60 Nm |

Action smoothing (low-pass filter) is applied before sending to MuJoCo:

```
torque_applied[t] = 0.6 * torque_applied[t-1]  +  0.4 * action[t] * max_torque[i]
```

---

## 6. Reward Function

### 6.1 Dense Per-Step Reward

```
r(t) =   w_vel    * v_x                          (forward velocity)
       - w_tilt   * tilt_angle                    (projection-based upright posture)
       - w_energy * mean(torques^2)               (efficiency)
       - w_load_x * |delta_x_load|               (load x-alignment)
       - w_load_y * |delta_y_load|               (load y-alignment)
       - w_load_z * max(0, -delta_z_load)         (load drop penalty)
       - w_jerk   * mean((a_t - a_{t-1})^2)       (smoothness)
       + w_alive                                  (survival per step)
```

### 6.2 Reward Weights

| Symbol | Weight | Rationale |
|--------|--------|-----------|
| `w_vel` | 2.0 | Primary objective — must walk forward |
| `w_tilt` | 0.5 | Projection-based upright posture (singularity-free) |
| `w_energy` | 0.001 | Penalize wasted torque without blocking movement |
| `w_load_x` | 1.0 | Load must stay centered left-right |
| `w_load_y` | 1.0 | Load must stay centered front-back |
| `w_load_z` | 2.0 | Load dropping is heavily penalized |
| `w_jerk` | 0.05 | Smooth actions protect joints and load |
| `w_alive` | 1.0 | Bonus just for surviving each step |

### 6.3 Sparse Milestones

A sparse bonus is awarded every time the agent reaches a new distance milestone without dropping the load:

```
+25.0   per 10 m walked without load drop
+100.0  for completing a full 1 km episode
```

### 6.4 Episode Termination Penalty

```
-50.0   when episode ends due to load drop, robot fall, or stuck detection
```

---

## 7. Training Strategy

### 7.1 Algorithm: PPO

Proximal Policy Optimization is selected because:
- It is the most reliable algorithm for continuous locomotion tasks.
- The clip mechanism prevents catastrophically large policy updates during unstable early training.
- It works well with moderate hardware (single GPU or even CPU for smaller experiments).

### 7.2 Policy Network Architecture

```
Input: observation vector (98 dimensions)
         |
    Linear(98 -> 256)  → LayerNorm → Tanh
         |
    Linear(256 -> 256) → LayerNorm → Tanh
         |
    Linear(256 -> 128) → LayerNorm → Tanh
         |
    +----+----+
    |         |
  Actor    Critic
  128->24  128->1
  + log_std  (value)
  (Gaussian  (no activation)
   policy)
```

### 7.3 Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Total training steps | 20,000,000 |
| Steps per rollout | 4096 |
| Minibatch size | 512 |
| PPO update epochs | 6 |
| Learning rate | 0.003 |
| Discount factor (gamma) | 0.99 |
| GAE lambda | 0.95 |
| PPO clip epsilon | 0.2 |
| Value loss coefficient | 0.5 |
| Entropy coefficient | 0.01 |
| Max gradient norm | 0.5 |

---

## 8. Curriculum Learning

The curriculum is a critical component — without it, the agent almost always fails in under 0.5 seconds with no useful reward signal, and learning never starts.

| Stage | Name | Goal | Load | Terrain | Gravity |
|-------|------|------|------|---------|---------|
| 0 | Stand | Stay upright 5 s | 2 kg | Flat | 0.5x |
| 1 | Balance | Hold load 10 s, stationary | 2 kg | Flat | Full |
| 2 | Walk | Walk 10 m, load stable | 5 kg | Flat | Full |
| 3 | Distance | Walk 100 m | 5–10 kg | Rolling | Full |
| 4 | Robustness | Walk 500 m, wind + noise | 5–15 kg | Rough | Full |
| 5 | Master | Walk 1 km, all conditions | 10–20 kg | All | Full |

Stage promotion is triggered automatically when the mean reward over the last 50 episodes exceeds the stage threshold **and** (for Walk+ stages) the mean distance exceeds a minimum distance threshold.

---

## 9. Domain Randomization

Between every episode, the following parameters are sampled randomly to force the agent to generalize rather than memorize:

| Parameter | Min | Max |
|-----------|-----|-----|
| Load mass | 2 kg | 20 kg |
| Head-load friction | 0.4 | 1.5 |
| Ground friction | 0.5 | 1.2 |
| Terrain slope | 0deg | 10deg |
| Lateral wind force | 0 N | 15 N |
| Observation noise (sigma) | 0.0 | 0.01 |
| Action delay (timesteps) | 0 | 1 |
| Joint friction multiplier | 0.8x | 1.2x |
| Gravity | 0.95x | 1.05x |

---

## 10. Tools and Technologies

| Component | Tool | Role |
|-----------|------|------|
| Language | Rust 1.85+ | Core environment, training |
| Physics | MuJoCo 3.2+ | Rigid body simulation |
| MuJoCo bindings | `mujoco-rs` | Rust access to MuJoCo |
| RL algorithm | Hand-rolled PPO | Custom batch gradient accumulation |
| Neural networks | `ndarray` (manual) | MLP with LayerNorm, backprop from scratch |
| Logging | `tracing` | Structured training logs |
| Config | `toml` + `serde` | Hyperparameter files |
| CLI | `clap` | Command-line interface |
| Visualization | MuJoCo viewer (`mjv`) | Real-time 3D rendering |

---

## 11. Installation

### Prerequisites

```bash
# 1. Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 2. MuJoCo 3.2+
mkdir -p ~/.mujoco
# Download the appropriate archive from:
# https://github.com/google-deepmind/mujoco/releases
tar -xzf mujoco-3.2.x-linux-x86_64.tar.gz -C ~/.mujoco/
echo 'export MUJOCO_PATH=$HOME/.mujoco/mujoco-3.2.x' >> ~/.bashrc
source ~/.bashrc

# 3. C compiler (for mujoco-rs FFI)
sudo apt-get install build-essential
```

### Build

```bash
git clone https://github.com/your-org/sapggo.git
cd sapggo
cargo build --release
```

### Verify

```bash
cargo test --workspace
cargo run --example random_agent -- --episodes 2
```

---

## 12. Usage

### Train (default curriculum, 20M steps)

```bash
cargo run --release --bin sapggo-train
```

### Train with custom config

```bash
cargo run --release --bin sapggo-train -- --config configs/train_curriculum.toml
```

### Resume from checkpoint

```bash
cargo run --release --bin sapggo-train -- --resume checkpoints/sapggo_step_5000000.pt
```

### Evaluate a trained policy

```bash
cargo run --release --bin sapggo-eval -- \
  --policy checkpoints/sapggo_final.pt \
  --episodes 20 --render
```

### Keyboard controls during visualization

| Key | Action |
|-----|--------|
| C | Cycle camera (side / front / free) |
| R | Reset episode |
| Space | Pause / resume |
| S | Half speed |
| F | Double speed |
| Esc | Quit |

---

## 13. Expected Results

After 20 million training steps (12–24 hours on a single GPU):

| Metric | Target |
|--------|--------|
| Mean episode distance | > 500 m |
| Load drop rate | < 5% |
| Mean load tilt angle | < 4deg |
| Mean walking speed | 0.8–1.2 m/s |
| Max load generalization | 15 kg (unseen) |
| Max slope generalization | 8deg (unseen) |
| Wind resistance | Up to 10 N lateral |

---

## 14. Roadmap

| Version | Milestone | Deliverables |
|---------|-----------|-------------|
| v0.1.0 | Month 1 | Core environment, random agent, flat terrain |
| v0.2.0 | Month 2 | Full PPO, curriculum stages 0–2, reward tuning |
| v0.3.0 | Month 3 | Curriculum stages 3–5, rough terrain, domain randomization |
| v0.4.0 | Month 4 | Full evaluation suite, demo video, benchmark metrics |
| v1.0.0 | Month 6 | Stable release, documentation, arXiv preprint |

---

## 15. Contributing

Contributions are welcome. Open an issue first before submitting a pull request.

High-impact contribution areas:

- SAC or TD3 algorithm alternative to PPO
- Arm articulation for active load balance assistance
- Bevy renderer integration (particles, weather, post-processing)
- Sim-to-sim transfer (Isaac Gym / Genesis -> SAPGGO)
- Real robot deployment interface (Unitree H1 or similar)

---

## 16. License and Credits

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

MuJoCo is free for research use. See [mujoco.org](https://mujoco.org).

**Acknowledgments:**
- To the carriers of Africa — the true masters of this art.
- To the DeepMind team for MuJoCo.
- To the Rust community for `mujoco-rs`, `rlkit`, and `Burn`.

---

<div align="center">

**SAPGGO v0.1.0** &nbsp;·&nbsp; *Simulate &rarr; Learn &rarr; Balance &rarr; Walk*

Inspired by the ancient wisdom of African carriers.
Built with Rust. Powered by physics.

*"The art of carrying is the art of balance. The art of balance is the art of presence."*

</div>
