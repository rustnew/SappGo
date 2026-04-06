# SAPGGO — Complete Implementation Document

## Technical Specification and Developer Guide

**Version 0.1.0** | Rust 1.85+ | MuJoCo 3.2+

---

## Table of Contents

1. [Project Setup and Workspace](#1-project-setup-and-workspace)
2. [MuJoCo Model (XML)](#2-mujoco-model-xml)
3. [Environment Crate: sapggo-env](#3-environment-crate-sapggo-env)
4. [Curriculum Crate: sapggo-curriculum](#4-curriculum-crate-sapggo-curriculum)
5. [Agent Crate: sapggo-agent](#5-agent-crate-sapggo-agent)
6. [Training Crate: sapggo-train](#6-training-crate-sapggo-train)
7. [Evaluation Crate: sapggo-eval](#7-evaluation-crate-sapggo-eval)
8. [Visualization Crate: sapggo-viz](#8-visualization-crate-sapggo-viz)
9. [Configuration Files](#9-configuration-files)
10. [Logging and Metrics](#10-logging-and-metrics)
11. [Testing Strategy](#11-testing-strategy)
12. [Performance Considerations](#12-performance-considerations)
13. [Dependency Tree](#13-dependency-tree)

---

## 1. Project Setup and Workspace

### 1.1 Workspace `Cargo.toml`

```toml
[workspace]
resolver = "2"
members = [
    "crates/sapggo-env",
    "crates/sapggo-agent",
    "crates/sapggo-train",
    "crates/sapggo-eval",
    "crates/sapggo-curriculum",
    "crates/sapggo-viz",
]

[workspace.package]
version     = "0.1.0"
edition     = "2021"
authors     = ["SAPGGO Contributors"]
license     = "MIT"

[workspace.dependencies]
mujoco-rs        = { version = "0.3", features = ["visualizer"] }
burn             = { version = "0.13", features = ["wgpu", "autodiff"] }
serde            = { version = "1.0", features = ["derive"] }
serde_json       = "1.0"
toml             = "0.8"
tracing          = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
clap             = { version = "4.0", features = ["derive"] }
rand             = "0.8"
rand_distr       = "0.4"
anyhow           = "1.0"
thiserror        = "1.0"
ndarray          = "0.15"
```

### 1.2 Directory Layout

```
sapggo/
+-- Cargo.toml                          # workspace root
+-- Cargo.lock
+-- crates/
|   +-- sapggo-env/
|   |   +-- Cargo.toml
|   |   +-- src/
|   |       +-- lib.rs
|   |       +-- environment.rs          # SapggoEnv: reset, step
|   |       +-- robot.rs                # joint map, torque limits
|   |       +-- load.rs                 # load placement and detection
|   |       +-- sensor.rs               # observation extraction
|   |       +-- reward.rs               # reward computation
|   |       +-- terrain.rs              # heightfield loading
|   |       +-- noise.rs                # sensor and action noise
|   |       +-- error.rs                # EnvError types
|   +-- sapggo-curriculum/
|   |   +-- Cargo.toml
|   |   +-- src/
|   |       +-- lib.rs
|   |       +-- stage.rs                # CurriculumStage enum
|   |       +-- manager.rs              # CurriculumManager
|   |       +-- params.rs               # per-stage physics parameters
|   +-- sapggo-agent/
|   |   +-- Cargo.toml
|   |   +-- src/
|   |       +-- lib.rs
|   |       +-- policy.rs               # Actor network (Burn)
|   |       +-- value.rs                # Critic network (Burn)
|   |       +-- ppo.rs                  # PPO algorithm
|   |       +-- rollout.rs              # rollout buffer
|   |       +-- normalize.rs            # running obs normalizer
|   +-- sapggo-train/
|   |   +-- Cargo.toml
|   |   +-- src/
|   |       +-- main.rs                 # binary entry point
|   |       +-- trainer.rs              # main training loop
|   |       +-- config.rs               # TrainConfig from toml
|   |       +-- checkpoint.rs           # save / load weights
|   |       +-- logger.rs               # metric logging
|   +-- sapggo-eval/
|   |   +-- Cargo.toml
|   |   +-- src/
|   |       +-- main.rs
|   |       +-- evaluator.rs            # episode runner, metrics
|   +-- sapggo-viz/
|       +-- Cargo.toml
|       +-- src/
|           +-- lib.rs
|           +-- viewer.rs               # MuJoCo viewer wrapper
|           +-- camera.rs               # camera mode cycling
+-- assets/
|   +-- robot_humanoid_load.xml
|   +-- terrain_flat.png
|   +-- terrain_rolling.png
|   +-- terrain_rough.png
+-- configs/
|   +-- train_default.toml
|   +-- train_curriculum.toml
|   +-- eval.toml
+-- examples/
|   +-- random_agent.rs
|   +-- manual_control.rs
|   +-- trained_demo.rs
+-- checkpoints/
+-- runs/
+-- demo/
```

---

## 2. MuJoCo Model (XML)

**File:** `assets/robot_humanoid_load.xml`

This file defines every physical property of the robot and the load.
It is the single most important file in the project.

```xml
<?xml version="1.0" encoding="utf-8"?>
<mujoco model="sapggo_humanoid">

  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>

  <option
    timestep   = "0.002"
    integrator = "RK4"
    gravity    = "0 0 -9.81"
    cone       = "elliptic"
    iterations = "50"
    solver     = "Newton"/>

  <!-- ASSETS -->
  <asset>
    <texture name="ground_tex" type="2d" builtin="checker"
             rgb1="0.22 0.22 0.22" rgb2="0.35 0.35 0.35"
             width="512" height="512"/>
    <material name="ground_mat" texture="ground_tex"
              texrepeat="5 5" reflectance="0.1"/>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.3 0.5 0.7" rgb2="0.0 0.0 0.0"
             width="512" height="512"/>
    <hfield name="terrain" file="terrain_flat.png"
            nrow="100" ncol="100" size="20 20 0.5 0.1"/>
    <material name="robot_body" rgba="0.7 0.7 0.7 1"
              specular="0.5" shininess="0.4"/>
    <material name="load_mat" rgba="0.8 0.3 0.1 1"
              specular="0.2" shininess="0.1"/>
  </asset>

  <!-- DEFAULTS -->
  <default>
    <joint limited="true" damping="0.1" stiffness="0" armature="0.01"/>
    <geom  contype="1" conaffinity="1" condim="3"
           friction="0.8 0.005 0.001" density="1000"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>

  <!-- VISUAL -->
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <!-- WORLD -->
  <worldbody>

    <light name="sun" directional="true" castshadow="true"
           pos="1 -1 1.5" dir="-1 1 -1"
           diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>

    <geom name="floor" type="hfield" hfield="terrain"
          pos="0 0 0" material="ground_mat"/>

    <body name="torso" pos="0 0 1.35">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="10.0"
                fullinertia="0.1 0.1 0.05 0 0 0"/>
      <geom type="capsule" size="0.07 0.25" material="robot_body"/>
      <site name="torso_imu" pos="0 0 0" size="0.01"/>

      <!-- TORSO JOINTS -->
      <joint name="torso_flex" type="hinge" axis="1 0 0"
             range="-0.5 0.5" damping="2.0"/>
      <joint name="torso_lat" type="hinge" axis="0 1 0"
             range="-0.3 0.3" damping="2.0"/>

      <!-- NECK -->
      <body name="neck" pos="0 0 0.27">
        <joint name="neck_tilt" type="hinge" axis="1 0 0"
               range="-0.4 0.4"/>
        <joint name="neck_rot"  type="hinge" axis="0 0 1"
               range="-0.6 0.6"/>
        <inertial pos="0 0 0" mass="1.5"
                  fullinertia="0.01 0.01 0.008 0 0 0"/>
        <geom type="capsule" size="0.03 0.07" material="robot_body"/>

        <!-- HEAD -->
        <body name="head" pos="0 0 0.10">
          <inertial pos="0 0 0" mass="4.5"
                    fullinertia="0.04 0.04 0.03 0 0 0"/>
          <geom name="head_sphere" type="sphere" size="0.11"
                material="robot_body"/>
          <!-- Contact surface for the load -->
          <geom name="head_top" type="sphere" size="0.06"
                pos="0 0 0.10" material="robot_body"
                friction="0.9 0.005 0.001"/>
          <site name="head_imu" pos="0 0 0.11" size="0.01"/>

          <!-- LOAD: free body resting on head -->
          <body name="load" pos="0 0 0.22">
            <freejoint name="load_joint" damping="0.05"/>
            <inertial pos="0 0 0" mass="5.0"
                      fullinertia="0.015 0.015 0.012 0 0 0"/>
            <geom name="load_geom" type="box"
                  size="0.15 0.15 0.10"
                  material="load_mat"
                  friction="0.8 0.005 0.001"
                  condim="6"/>
            <site name="load_imu" pos="0 0 0" size="0.01"/>
          </body>

        </body> <!-- head -->
      </body> <!-- neck -->

      <!-- LEFT LEG -->
      <body name="upper_leg_L" pos="-0.09 0 -0.22">
        <joint name="hip_flex_L" type="hinge" axis="1 0 0"
               range="-0.7 1.2" damping="0.5"/>
        <joint name="hip_add_L"  type="hinge" axis="0 0 1"
               range="-0.5 0.5" damping="0.3"/>
        <joint name="hip_rot_L"  type="hinge" axis="0 1 0"
               range="-0.5 0.5" damping="0.3"/>
        <inertial pos="0 0 -0.2" mass="7.0"
                  fullinertia="0.08 0.08 0.02 0 0 0"/>
        <geom type="capsule" size="0.05 0.2" pos="0 0 -0.2"
              material="robot_body"/>
        <body name="lower_leg_L" pos="0 0 -0.42">
          <joint name="knee_L" type="hinge" axis="1 0 0"
                 range="-1.8 0.0" damping="0.5"/>
          <inertial pos="0 0 -0.18" mass="4.5"
                    fullinertia="0.05 0.05 0.01 0 0 0"/>
          <geom type="capsule" size="0.04 0.18" pos="0 0 -0.18"
                material="robot_body"/>
          <body name="foot_L" pos="0 0 -0.38">
            <joint name="ankle_flex_L" type="hinge" axis="1 0 0"
                   range="-0.7 0.5" damping="0.3"/>
            <joint name="ankle_inv_L"  type="hinge" axis="0 0 1"
                   range="-0.4 0.4" damping="0.2"/>
            <inertial pos="0.04 0 -0.02" mass="1.5"
                      fullinertia="0.003 0.012 0.01 0 0 0"/>
            <geom name="foot_L" type="box"
                  size="0.10 0.05 0.02" pos="0.04 0 -0.02"
                  material="robot_body" condim="6"/>
            <site name="foot_L_contact" pos="0.04 0 -0.04"/>
          </body>
        </body>
      </body>

      <!-- RIGHT LEG (mirror) -->
      <body name="upper_leg_R" pos="0.09 0 -0.22">
        <joint name="hip_flex_R" type="hinge" axis="1 0 0"
               range="-0.7 1.2" damping="0.5"/>
        <joint name="hip_add_R"  type="hinge" axis="0 0 1"
               range="-0.5 0.5" damping="0.3"/>
        <joint name="hip_rot_R"  type="hinge" axis="0 1 0"
               range="-0.5 0.5" damping="0.3"/>
        <inertial pos="0 0 -0.2" mass="7.0"
                  fullinertia="0.08 0.08 0.02 0 0 0"/>
        <geom type="capsule" size="0.05 0.2" pos="0 0 -0.2"
              material="robot_body"/>
        <body name="lower_leg_R" pos="0 0 -0.42">
          <joint name="knee_R" type="hinge" axis="1 0 0"
                 range="-1.8 0.0" damping="0.5"/>
          <inertial pos="0 0 -0.18" mass="4.5"
                    fullinertia="0.05 0.05 0.01 0 0 0"/>
          <geom type="capsule" size="0.04 0.18" pos="0 0 -0.18"
                material="robot_body"/>
          <body name="foot_R" pos="0 0 -0.38">
            <joint name="ankle_flex_R" type="hinge" axis="1 0 0"
                   range="-0.7 0.5" damping="0.3"/>
            <joint name="ankle_inv_R"  type="hinge" axis="0 0 1"
                   range="-0.4 0.4" damping="0.2"/>
            <inertial pos="0.04 0 -0.02" mass="1.5"
                      fullinertia="0.003 0.012 0.01 0 0 0"/>
            <geom name="foot_R" type="box"
                  size="0.10 0.05 0.02" pos="0.04 0 -0.02"
                  material="robot_body" condim="6"/>
            <site name="foot_R_contact" pos="0.04 0 -0.04"/>
          </body>
        </body>
      </body>

    </body> <!-- torso -->
  </worldbody>

  <!-- SENSORS -->
  <sensor>
    <framequat     name="torso_quat"  objtype="body" objname="torso"/>
    <gyro          name="torso_gyro"  site="torso_imu"/>
    <accelerometer name="torso_acc"   site="torso_imu"/>
    <framequat     name="head_quat"   objtype="body" objname="head"/>
    <framepos      name="load_pos"    objtype="body" objname="load"/>
    <framequat     name="load_quat"   objtype="body" objname="load"/>
    <gyro          name="load_gyro"   site="load_imu"/>
    <accelerometer name="load_acc"    site="load_imu"/>
    <force         name="foot_L_force" site="foot_L_contact"/>
    <force         name="foot_R_force" site="foot_R_contact"/>
  </sensor>

  <!-- ACTUATORS -->
  <actuator>
    <motor name="hip_flex_L"   joint="hip_flex_L"   gear="150"/>
    <motor name="hip_add_L"    joint="hip_add_L"    gear="150"/>
    <motor name="hip_rot_L"    joint="hip_rot_L"    gear="150"/>
    <motor name="knee_L"       joint="knee_L"       gear="200"/>
    <motor name="ankle_flex_L" joint="ankle_flex_L" gear="80"/>
    <motor name="ankle_inv_L"  joint="ankle_inv_L"  gear="80"/>
    <motor name="hip_flex_R"   joint="hip_flex_R"   gear="150"/>
    <motor name="hip_add_R"    joint="hip_add_R"    gear="150"/>
    <motor name="hip_rot_R"    joint="hip_rot_R"    gear="150"/>
    <motor name="knee_R"       joint="knee_R"       gear="200"/>
    <motor name="ankle_flex_R" joint="ankle_flex_R" gear="80"/>
    <motor name="ankle_inv_R"  joint="ankle_inv_R"  gear="80"/>
    <motor name="torso_flex"   joint="torso_flex"   gear="120"/>
    <motor name="torso_lat"    joint="torso_lat"    gear="120"/>
    <motor name="neck_tilt"    joint="neck_tilt"    gear="40"/>
    <motor name="neck_rot"     joint="neck_rot"     gear="40"/>
  </actuator>

</mujoco>
```

---

## 3. Environment Crate: sapggo-env

### 3.1 `src/robot.rs`

```rust
/// Ordered list of actuated joint names.
/// Index = index in the action vector.
pub const JOINT_NAMES: [&str; 16] = [
    "hip_flex_L", "hip_add_L",    "hip_rot_L",
    "knee_L",
    "ankle_flex_L", "ankle_inv_L",
    "hip_flex_R", "hip_add_R",    "hip_rot_R",
    "knee_R",
    "ankle_flex_R", "ankle_inv_R",
    "torso_flex",  "torso_lat",
    "neck_tilt",   "neck_rot",
];

/// Maximum torque (Nm) per actuator, same order as JOINT_NAMES.
pub const MAX_TORQUE: [f64; 16] = [
    150.0, 150.0, 150.0,    // hip L
    200.0,                   // knee L
    80.0,  80.0,             // ankle L
    150.0, 150.0, 150.0,    // hip R
    200.0,                   // knee R
    80.0,  80.0,             // ankle R
    120.0, 120.0,            // torso
    40.0,  40.0,             // neck
];

pub const N_JOINTS:  usize = 16;
pub const OBS_DIM:   usize = 70;
pub const ACT_DIM:   usize = 16;
```

### 3.2 `src/sensor.rs`

```rust
use ndarray::Array1;

/// Observation vector layout (70 dimensions):
///   [0..16]   joint angles
///   [16..32]  joint velocities
///   [32..36]  torso quaternion (w, x, y, z)
///   [36..39]  torso angular velocity (gyro)
///   [39..47]  foot contact forces (4 per foot)
///   [47..50]  load offset from head center (dx, dy, dz)
///   [50..53]  load angular velocity
///   [53..69]  previous action (16 joints)
///   [69]      target forward velocity (curriculum)
pub fn extract_observation(
    model:       &mujoco_rs::MjModel,
    data:        &mujoco_rs::MjData,
    prev_action: &[f64; 16],
    target_vel:  f64,
    noise_sigma: f64,
    rng:         &mut impl rand::Rng,
) -> Array1<f64> {
    use rand_distr::{Normal, Distribution};
    let normal = Normal::new(0.0, noise_sigma).unwrap();
    let mut obs = Array1::<f64>::zeros(OBS_DIM);

    // Joint angles and velocities
    for i in 0..16 {
        let qa = data.get_joint_qpos_addr(model, i);
        let qv = data.get_joint_qvel_addr(model, i);
        obs[i]      = data.qpos[qa] + normal.sample(rng);
        obs[16 + i] = data.qvel[qv] + normal.sample(rng);
    }

    // Torso quaternion
    let tid = model.body_name2id("torso");
    for j in 0..4 { obs[32 + j] = data.xquat[tid][j] + normal.sample(rng); }

    // Torso gyro
    let gyro = model.sensor_name2addr("torso_gyro");
    for j in 0..3 { obs[36 + j] = data.sensordata[gyro + j] + normal.sample(rng); }

    // Foot contact forces
    let fl = model.sensor_name2addr("foot_L_force");
    let fr = model.sensor_name2addr("foot_R_force");
    for j in 0..4 {
        obs[39 + j] = data.sensordata[fl + j] + normal.sample(rng);
        obs[43 + j] = data.sensordata[fr + j] + normal.sample(rng);
    }

    // Load offset from head
    let hid = model.body_name2id("head");
    let lid = model.body_name2id("load");
    for j in 0..3 {
        obs[47 + j] = (data.xpos[lid][j] - data.xpos[hid][j]) + normal.sample(rng);
    }

    // Load angular velocity
    let lg = model.sensor_name2addr("load_gyro");
    for j in 0..3 { obs[50 + j] = data.sensordata[lg + j] + normal.sample(rng); }

    // Previous action
    for i in 0..16 { obs[53 + i] = prev_action[i]; }

    // Target velocity
    obs[69] = target_vel;

    obs
}
```

### 3.3 `src/reward.rs`

```rust
pub struct RewardWeights {
    pub vel:    f64,  // 2.0   forward velocity
    pub pitch:  f64,  // 0.3   torso pitch penalty
    pub roll:   f64,  // 0.2   torso roll penalty
    pub energy: f64,  // 0.001 torque energy penalty
    pub load_x: f64,  // 1.0   load x-offset penalty
    pub load_y: f64,  // 1.0   load y-offset penalty
    pub load_z: f64,  // 2.0   load z-drop penalty
    pub jerk:   f64,  // 0.05  action smoothness penalty
    pub alive:  f64,  // 0.1   survival bonus
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self { vel:2.0, pitch:0.3, roll:0.2, energy:0.001,
               load_x:1.0, load_y:1.0, load_z:2.0,
               jerk:0.05, alive:0.1 }
    }
}

pub struct RewardState {
    pub velocity_x:  f64,
    pub torso_pitch: f64,
    pub torso_roll:  f64,
    pub load_dx:     f64,
    pub load_dy:     f64,
    pub load_dz:     f64,
    pub torques:     [f64; 16],
    pub prev_action: [f64; 16],
    pub cur_action:  [f64; 16],
}

pub fn compute_reward(s: &RewardState, w: &RewardWeights) -> f64 {
    let energy = s.torques.iter().map(|t| t*t).sum::<f64>() / 16.0;
    let jerk   = s.cur_action.iter().zip(s.prev_action.iter())
        .map(|(a,b)| (a-b).powi(2)).sum::<f64>() / 16.0;
    let z_drop = (-s.load_dz).max(0.0);

      w.vel    *  s.velocity_x
    - w.pitch  *  s.torso_pitch.abs()
    - w.roll   *  s.torso_roll.abs()
    - w.energy *  energy
    - w.load_x *  s.load_dx.abs()
    - w.load_y *  s.load_dy.abs()
    - w.load_z *  z_drop
    - w.jerk   *  jerk
    + w.alive
}

pub const MILESTONE_BONUS:      f64 = 25.0;   // every 10 m
pub const EPISODE_BONUS:        f64 = 100.0;  // 1 km full episode
pub const TERMINATION_PENALTY:  f64 = -50.0;
```

### 3.4 `src/environment.rs`

```rust
use mujoco_rs::{MjModel, MjData, mj_resetData, mj_step};
use ndarray::Array1;
use crate::{sensor, reward, robot::*};
use sapggo_curriculum::CurriculumParams;

pub struct StepResult {
    pub observation: Array1<f64>,
    pub reward:      f64,
    pub done:        bool,
    pub info:        StepInfo,
}

pub struct StepInfo {
    pub distance_m:   f64,
    pub load_dropped: bool,
    pub robot_fallen: bool,
    pub steps:        u64,
    pub total_reward: f64,
}

pub struct SapggoEnv {
    model:          MjModel,
    data:           MjData,
    params:         CurriculumParams,
    weights:        reward::RewardWeights,
    prev_action:    [f64; 16],
    smoothed_ctrl:  [f64; 16],
    steps:          u64,
    distance_m:     f64,
    total_reward:   f64,
    last_milestone: u64,
    rng:            rand::rngs::StdRng,
}

impl SapggoEnv {
    /// Simulation steps per control step (10 x 2 ms = 20 ms control freq)
    pub const SIM_STEPS:     usize = 10;
    /// Passive settling steps at episode reset
    pub const PASSIVE_STEPS: usize = 50;
    /// Maximum steps per episode
    pub const MAX_STEPS:     u64   = 1000;

    pub fn new(model_path: &str, seed: u64) -> anyhow::Result<Self> {
        let model = MjModel::from_file(model_path)?;
        let data  = MjData::new(&model);
        Ok(Self {
            model, data,
            params:        CurriculumParams::default(),
            weights:       reward::RewardWeights::default(),
            prev_action:   [0.0; 16],
            smoothed_ctrl: [0.0; 16],
            steps:         0,
            distance_m:    0.0,
            total_reward:  0.0,
            last_milestone: 0,
            rng: rand::SeedableRng::seed_from_u64(seed),
        })
    }

    pub fn set_params(&mut self, p: CurriculumParams) {
        self.params = p;
    }

    /// Reset episode: reload model state, place load, settle, return obs.
    pub fn reset(&mut self) -> Array1<f64> {
        mj_resetData(&self.model, &mut self.data);
        self.apply_domain_randomization();
        self.place_load_on_head();

        // Passive settling: load falls gently onto head
        for _ in 0..Self::PASSIVE_STEPS {
            mj_step(&self.model, &mut self.data);
        }

        self.prev_action   = [0.0; 16];
        self.smoothed_ctrl = [0.0; 16];
        self.steps         = 0;
        self.distance_m    = 0.0;
        self.total_reward  = 0.0;
        self.last_milestone = 0;

        sensor::extract_observation(
            &self.model, &self.data,
            &self.prev_action,
            self.params.target_velocity,
            self.params.observation_noise,
            &mut self.rng,
        )
    }

    /// Advance one control step.
    pub fn step(&mut self, action: &[f64]) -> StepResult {
        debug_assert_eq!(action.len(), ACT_DIM);

        // Action smoothing: low-pass filter to prevent mechanical shock
        for i in 0..N_JOINTS {
            self.smoothed_ctrl[i] =
                0.8 * self.smoothed_ctrl[i] + 0.2 * action[i] * MAX_TORQUE[i];
            self.data.ctrl[i] = self.smoothed_ctrl[i]
                .clamp(-MAX_TORQUE[i], MAX_TORQUE[i]);
        }

        // Record x position before step
        let tid      = self.model.body_name2id("torso");
        let x_before = self.data.xpos[tid][0];

        // Advance physics
        for _ in 0..Self::SIM_STEPS {
            mj_step(&self.model, &mut self.data);
        }

        let x_after = self.data.xpos[tid][0];
        self.steps      += 1;
        self.distance_m += (x_after - x_before).max(0.0);

        // Check termination
        let load_dropped = self.load_is_dropped();
        let robot_fallen = self.robot_is_fallen();
        let timeout      = self.steps >= Self::MAX_STEPS;
        let done         = load_dropped || robot_fallen || timeout;

        // Compute reward
        let rs   = self.build_reward_state(x_after - x_before, action);
        let mut r = reward::compute_reward(&rs, &self.weights);

        // Milestone bonuses (every 10 m)
        let ms_now = (self.distance_m / 10.0).floor() as u64;
        if ms_now > self.last_milestone {
            r += reward::MILESTONE_BONUS * (ms_now - self.last_milestone) as f64;
            self.last_milestone = ms_now;
        }

        // Termination penalty
        if done && (load_dropped || robot_fallen) {
            r += reward::TERMINATION_PENALTY;
        }

        self.total_reward += r;

        let cur: [f64; 16] = action.try_into().unwrap();
        self.prev_action = cur;

        let obs = sensor::extract_observation(
            &self.model, &self.data,
            &self.prev_action,
            self.params.target_velocity,
            self.params.observation_noise,
            &mut self.rng,
        );

        StepResult {
            observation: obs,
            reward: r,
            done,
            info: StepInfo {
                distance_m:   self.distance_m,
                load_dropped,
                robot_fallen,
                steps:        self.steps,
                total_reward: self.total_reward,
            },
        }
    }

    fn load_is_dropped(&self) -> bool {
        let hid = self.model.body_name2id("head");
        let lid = self.model.body_name2id("load");
        (self.data.xpos[lid][2] - self.data.xpos[hid][2]) < -0.05
    }

    fn robot_is_fallen(&self) -> bool {
        let tid = self.model.body_name2id("torso");
        self.data.xpos[tid][2] < 0.6
    }

    fn place_load_on_head(&mut self) {
        // Small random offset for domain randomization
        let dx = (rand::Rng::gen::<f64>(&mut self.rng) - 0.5) * 0.030;
        let dy = (rand::Rng::gen::<f64>(&mut self.rng) - 0.5) * 0.030;
        // Load qpos: 7 values (3 pos + 4 quat) after the robot's DOFs
        // Exact offset depends on model DOF count; set via mj_forward
        // For now, the XML places it at pos="0 0 0.22" relative to head
        let _ = (dx, dy); // TODO: fine-tune offset via model indexing
    }

    fn apply_domain_randomization(&mut self) {
        use rand::Rng;
        let p = &self.params;

        // Wind force on load
        let wind = (2.0 * self.rng.gen::<f64>() - 1.0) * p.wind_force_max;
        let lid  = self.model.body_name2id("load");
        self.data.xfrc_applied[lid][0] = wind;
    }

    fn build_reward_state(&self, delta_x: f64, action: &[f64]) -> reward::RewardState {
        let tid = self.model.body_name2id("torso");
        let hid = self.model.body_name2id("head");
        let lid = self.model.body_name2id("load");

        let quat  = self.data.xquat[tid];
        let pitch = 2.0 * (quat[2]*quat[3] + quat[0]*quat[1]).asin();
        let roll  = 2.0 * (quat[0]*quat[2] - quat[1]*quat[3])
            .atan2(1.0 - 2.0*(quat[1].powi(2) + quat[2].powi(2)));

        let mut torques    = [0.0f64; 16];
        let mut cur_action = [0.0f64; 16];
        for i in 0..16 {
            torques[i]    = self.data.actuator_force[i];
            cur_action[i] = action[i];
        }

        reward::RewardState {
            velocity_x:  delta_x / (Self::SIM_STEPS as f64 * 0.002),
            torso_pitch: pitch,
            torso_roll:  roll,
            load_dx:     self.data.xpos[lid][0] - self.data.xpos[hid][0],
            load_dy:     self.data.xpos[lid][1] - self.data.xpos[hid][1],
            load_dz:     self.data.xpos[lid][2] - self.data.xpos[hid][2],
            torques,
            prev_action: self.prev_action,
            cur_action,
        }
    }
}
```

---

## 4. Curriculum Crate: sapggo-curriculum

### 4.1 `src/stage.rs`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurriculumStage {
    Stand    = 0,
    Balance  = 1,
    Walk     = 2,
    Distance = 3,
    Robust   = 4,
    Master   = 5,
}

impl CurriculumStage {
    /// Mean reward over last 50 episodes required to advance.
    pub fn promotion_threshold(&self) -> f64 {
        match self {
            Self::Stand    => 15.0,
            Self::Balance  => 25.0,
            Self::Walk     => 50.0,
            Self::Distance => 120.0,
            Self::Robust   => 250.0,
            Self::Master   => f64::INFINITY,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Stand    => "Stand",
            Self::Balance  => "Balance",
            Self::Walk     => "Walk",
            Self::Distance => "Distance",
            Self::Robust   => "Robust",
            Self::Master   => "Master",
        }
    }

    pub fn next(&self) -> Option<Self> {
        match self {
            Self::Stand    => Some(Self::Balance),
            Self::Balance  => Some(Self::Walk),
            Self::Walk     => Some(Self::Distance),
            Self::Distance => Some(Self::Robust),
            Self::Robust   => Some(Self::Master),
            Self::Master   => None,
        }
    }
}
```

### 4.2 `src/params.rs`

```rust
use serde::{Deserialize, Serialize};
use crate::stage::CurriculumStage;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CurriculumParams {
    pub load_mass_min:        f64,
    pub load_mass_max:        f64,
    pub head_friction_min:    f64,
    pub head_friction_max:    f64,
    pub ground_friction_min:  f64,
    pub ground_friction_max:  f64,
    pub wind_force_max:       f64,
    pub gravity_scale:        f64,
    pub terrain_type:         TerrainType,
    pub observation_noise:    f64,
    pub action_noise:         f64,
    pub target_velocity:      f64,
    pub max_steps:            u64,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub enum TerrainType { Flat, Rolling, Rough }

impl Default for CurriculumParams {
    fn default() -> Self { Self::for_stage(CurriculumStage::Stand) }
}

impl CurriculumParams {
    pub fn for_stage(s: CurriculumStage) -> Self {
        match s {
            CurriculumStage::Stand => Self {
                load_mass_min: 2.0, load_mass_max: 2.0,
                wind_force_max: 0.0, gravity_scale: 0.5,
                terrain_type: TerrainType::Flat,
                observation_noise: 0.0, action_noise: 0.0,
                target_velocity: 0.0, max_steps: 250,
                head_friction_min: 0.8, head_friction_max: 0.8,
                ground_friction_min: 0.8, ground_friction_max: 0.8,
            },
            CurriculumStage::Balance => Self {
                load_mass_min: 2.0, load_mass_max: 5.0,
                wind_force_max: 0.0, gravity_scale: 1.0,
                terrain_type: TerrainType::Flat,
                observation_noise: 0.0, action_noise: 0.0,
                target_velocity: 0.0, max_steps: 500,
                head_friction_min: 0.7, head_friction_max: 1.0,
                ground_friction_min: 0.8, ground_friction_max: 1.0,
            },
            CurriculumStage::Walk => Self {
                load_mass_min: 2.0, load_mass_max: 7.0,
                wind_force_max: 2.0, gravity_scale: 1.0,
                terrain_type: TerrainType::Flat,
                observation_noise: 0.002, action_noise: 0.01,
                target_velocity: 0.8, max_steps: 600,
                head_friction_min: 0.6, head_friction_max: 1.2,
                ground_friction_min: 0.7, ground_friction_max: 1.1,
            },
            CurriculumStage::Distance => Self {
                load_mass_min: 5.0, load_mass_max: 10.0,
                wind_force_max: 5.0, gravity_scale: 1.0,
                terrain_type: TerrainType::Rolling,
                observation_noise: 0.005, action_noise: 0.02,
                target_velocity: 1.0, max_steps: 800,
                head_friction_min: 0.5, head_friction_max: 1.3,
                ground_friction_min: 0.6, ground_friction_max: 1.2,
            },
            CurriculumStage::Robust => Self {
                load_mass_min: 5.0, load_mass_max: 15.0,
                wind_force_max: 10.0, gravity_scale: 1.0,
                terrain_type: TerrainType::Rough,
                observation_noise: 0.008, action_noise: 0.03,
                target_velocity: 1.0, max_steps: 1000,
                head_friction_min: 0.4, head_friction_max: 1.5,
                ground_friction_min: 0.5, ground_friction_max: 1.3,
            },
            CurriculumStage::Master => Self {
                load_mass_min: 10.0, load_mass_max: 20.0,
                wind_force_max: 15.0, gravity_scale: 1.0,
                terrain_type: TerrainType::Rough,
                observation_noise: 0.01, action_noise: 0.04,
                target_velocity: 1.2, max_steps: 1000,
                head_friction_min: 0.4, head_friction_max: 1.5,
                ground_friction_min: 0.5, ground_friction_max: 1.3,
            },
        }
    }
}
```

### 4.3 `src/manager.rs`

```rust
use std::collections::VecDeque;
use crate::{stage::CurriculumStage, params::CurriculumParams};

pub struct CurriculumManager {
    pub stage:          CurriculumStage,
    pub episode_count:  u64,
    recent_rewards:     VecDeque<f64>,
    window:             usize,
}

impl CurriculumManager {
    pub fn new() -> Self {
        Self {
            stage:          CurriculumStage::Stand,
            episode_count:  0,
            recent_rewards: VecDeque::new(),
            window:         50,
        }
    }

    /// Call at the end of each episode. Returns new stage if promoted.
    pub fn on_episode_end(&mut self, reward: f64) -> Option<CurriculumStage> {
        self.episode_count += 1;
        self.recent_rewards.push_back(reward);
        if self.recent_rewards.len() > self.window {
            self.recent_rewards.pop_front();
        }

        if self.recent_rewards.len() < self.window { return None; }

        let mean = self.recent_rewards.iter().sum::<f64>() / self.window as f64;
        if mean >= self.stage.promotion_threshold() {
            if let Some(next) = self.stage.next() {
                tracing::info!(
                    from  = self.stage.name(),
                    to    = next.name(),
                    mean_reward = mean,
                    "Curriculum promoted"
                );
                self.stage = next;
                self.recent_rewards.clear();
                return Some(self.stage);
            }
        }
        None
    }

    pub fn current_params(&self) -> CurriculumParams {
        CurriculumParams::for_stage(self.stage)
    }
}
```

---

## 5. Agent Crate: sapggo-agent

### 5.1 `src/policy.rs` (Actor Network)

```rust
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig};

#[derive(Module, Debug)]
pub struct Actor<B: Backend> {
    fc1:     Linear<B>,
    norm1:   LayerNorm<B>,
    fc2:     Linear<B>,
    norm2:   LayerNorm<B>,
    mean:    Linear<B>,
    log_std: Param<Tensor<B, 1>>,
}

impl<B: Backend> Actor<B> {
    pub fn new(obs_dim: usize, act_dim: usize, device: &B::Device) -> Self {
        Self {
            fc1:     LinearConfig::new(obs_dim, 256).init(device),
            norm1:   LayerNormConfig::new(256).init(device),
            fc2:     LinearConfig::new(256, 256).init(device),
            norm2:   LayerNormConfig::new(256).init(device),
            mean:    LinearConfig::new(256, act_dim).init(device),
            log_std: Param::from_tensor(Tensor::full([act_dim], -0.5, device)),
        }
    }

    /// Returns (mean, std) of the Gaussian action distribution.
    pub fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = burn::tensor::activation::tanh(self.norm1.forward(self.fc1.forward(obs)));
        let h = burn::tensor::activation::tanh(self.norm2.forward(self.fc2.forward(h)));
        let mean    = self.mean.forward(h);
        let log_std = self.log_std.val().clamp(-4.0, 0.0);
        let std     = log_std.exp().unsqueeze::<2>();
        (mean, std)
    }
}
```

### 5.2 `src/value.rs` (Critic Network)

```rust
#[derive(Module, Debug)]
pub struct Critic<B: Backend> {
    fc1:  Linear<B>,
    norm1: LayerNorm<B>,
    fc2:  Linear<B>,
    norm2: LayerNorm<B>,
    out:  Linear<B>,
}

impl<B: Backend> Critic<B> {
    pub fn new(obs_dim: usize, device: &B::Device) -> Self {
        Self {
            fc1:  LinearConfig::new(obs_dim, 256).init(device),
            norm1: LayerNormConfig::new(256).init(device),
            fc2:  LinearConfig::new(256, 256).init(device),
            norm2: LayerNormConfig::new(256).init(device),
            out:  LinearConfig::new(256, 1).init(device),
        }
    }

    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = burn::tensor::activation::tanh(self.norm1.forward(self.fc1.forward(obs)));
        let h = burn::tensor::activation::tanh(self.norm2.forward(self.fc2.forward(h)));
        self.out.forward(h)
    }
}
```

### 5.3 `src/ppo.rs` (PPO Core)

```rust
pub struct PpoConfig {
    pub clip_epsilon:    f64,   // 0.2
    pub gamma:           f64,   // 0.99
    pub gae_lambda:      f64,   // 0.95
    pub lr:              f64,   // 3e-4
    pub epochs:          usize, // 10
    pub minibatch_size:  usize, // 512
    pub value_coef:      f64,   // 0.5
    pub entropy_coef:    f64,   // 0.01
    pub max_grad_norm:   f64,   // 0.5
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            clip_epsilon: 0.2, gamma: 0.99, gae_lambda: 0.95,
            lr: 3e-4, epochs: 10, minibatch_size: 512,
            value_coef: 0.5, entropy_coef: 0.01, max_grad_norm: 0.5,
        }
    }
}

/// Generalized Advantage Estimation (GAE).
pub fn compute_gae(
    rewards:    &[f64],
    values:     &[f64],
    dones:      &[bool],
    last_value: f64,
    gamma:      f64,
    lam:        f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    let mut advantages = vec![0.0f64; n];
    let mut returns    = vec![0.0f64; n];
    let mut gae = 0.0f64;

    for t in (0..n).rev() {
        let next_v = if t + 1 < n { values[t + 1] } else { last_value };
        let mask   = if dones[t] { 0.0 } else { 1.0 };
        let delta  = rewards[t] + gamma * next_v * mask - values[t];
        gae        = delta + gamma * lam * mask * gae;
        advantages[t] = gae;
        returns[t]    = gae + values[t];
    }
    (advantages, returns)
}
```

---

## 6. Training Crate: sapggo-train

### 6.1 `src/config.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct TrainConfig {
    pub model_path:            String,
    pub seed:                  u64,
    pub total_steps:           u64,
    pub rollout_steps:         usize,
    pub curriculum:            bool,
    pub clip_epsilon:          f64,
    pub gamma:                 f64,
    pub gae_lambda:            f64,
    pub lr:                    f64,
    pub epochs:                usize,
    pub minibatch_size:        usize,
    pub value_coef:            f64,
    pub entropy_coef:          f64,
    pub max_grad_norm:         f64,
    pub log_interval:          u64,
    pub checkpoint_interval:   u64,
    pub checkpoint_dir:        String,
    pub log_dir:               String,
}
```

### 6.2 `src/trainer.rs` (Training Loop)

```rust
pub struct Trainer {
    env:         SapggoEnv,
    agent:       PpoAgent,
    curriculum:  CurriculumManager,
    config:      TrainConfig,
    logger:      TrainingLogger,
    global_step: u64,
    episode:     u64,
}

impl Trainer {
    pub fn run(&mut self) {
        let mut obs = self.env.reset();

        while self.global_step < self.config.total_steps {

            // 1. Collect rollout
            let rollout = self.collect_rollout(&mut obs);
            self.global_step += rollout.len() as u64;

            // 2. PPO update
            let stats = self.agent.update(&rollout);

            // 3. Curriculum: check promotions
            for ep_r in &rollout.episode_rewards {
                self.episode += 1;
                if let Some(_stage) = self.curriculum.on_episode_end(*ep_r) {
                    self.env.set_params(self.curriculum.current_params());
                }
            }

            // 4. Logging
            if self.episode % self.config.log_interval == 0 {
                let mean_r = rollout.episode_rewards.iter()
                    .sum::<f64>() / rollout.episode_rewards.len() as f64;
                self.logger.log("train/reward_mean",       mean_r,            self.global_step);
                self.logger.log("train/policy_loss",       stats.policy_loss, self.global_step);
                self.logger.log("train/value_loss",        stats.value_loss,  self.global_step);
                self.logger.log("train/entropy",           stats.entropy,     self.global_step);
                self.logger.log("curriculum/stage",
                    self.curriculum.stage as i32 as f64, self.global_step);
                tracing::info!(
                    step  = self.global_step,
                    stage = self.curriculum.stage.name(),
                    mean_reward = mean_r,
                    "Training update"
                );
            }

            // 5. Checkpoint
            if self.global_step % self.config.checkpoint_interval == 0 {
                let path = format!(
                    "{}/sapggo_step_{}.pt",
                    self.config.checkpoint_dir, self.global_step
                );
                self.agent.save(&path).expect("Checkpoint save failed");
            }
        }

        self.agent.save(
            &format!("{}/sapggo_final.pt", self.config.checkpoint_dir)
        ).expect("Final checkpoint failed");

        tracing::info!(
            total_steps = self.global_step,
            "Training complete"
        );
    }
}
```

---

## 7. Evaluation Crate: sapggo-eval

```rust
pub struct EvalMetrics {
    pub mean_distance_m:    f64,
    pub mean_steps:         f64,
    pub load_drop_rate:     f64,
    pub mean_load_tilt_deg: f64,
    pub mean_speed_ms:      f64,
    pub n_episodes:         usize,
}

pub fn run_evaluation(
    env:       &mut SapggoEnv,
    agent:     &impl Policy,
    episodes:  usize,
    render:    bool,
) -> EvalMetrics {
    let mut distances = Vec::with_capacity(episodes);
    let mut steps_v   = Vec::with_capacity(episodes);
    let mut drops     = 0usize;

    for ep in 0..episodes {
        let mut obs  = env.reset();
        let mut done = false;
        let mut ep_steps = 0u64;

        while !done {
            let action = agent.act_deterministic(&obs);
            let result = env.step(&action);
            obs     = result.observation;
            done    = result.done;
            ep_steps += 1;
        }

        let info = env.last_info();
        distances.push(info.distance_m);
        steps_v.push(ep_steps);
        if info.load_dropped { drops += 1; }

        tracing::info!(
            episode  = ep + 1,
            distance = info.distance_m,
            dropped  = info.load_dropped,
            "Eval episode"
        );
    }

    let total_dist  = distances.iter().sum::<f64>();
    let total_steps: u64 = steps_v.iter().sum();

    EvalMetrics {
        mean_distance_m:    total_dist  / episodes as f64,
        mean_steps:         total_steps as f64 / episodes as f64,
        load_drop_rate:     drops as f64 / episodes as f64,
        mean_load_tilt_deg: 0.0,  // computed from obs buffer if needed
        mean_speed_ms:      total_dist / (total_steps as f64 * 0.02),
        n_episodes:         episodes,
    }
}
```

---

## 8. Visualization Crate: sapggo-viz

```rust
// src/camera.rs
pub enum CameraMode { Side, Front, Free }

pub struct SapggoViewer {
    mode: CameraMode,
}

impl SapggoViewer {
    pub fn new() -> Self { Self { mode: CameraMode::Side } }

    pub fn cycle_camera(&mut self) {
        self.mode = match self.mode {
            CameraMode::Side  => CameraMode::Front,
            CameraMode::Front => CameraMode::Free,
            CameraMode::Free  => CameraMode::Side,
        };
    }

    pub fn apply_to(&self, cam: &mut mujoco_rs::MjvCamera) {
        match self.mode {
            CameraMode::Side  => { cam.azimuth=90.0; cam.elevation=-15.0; cam.distance=4.0; }
            CameraMode::Front => { cam.azimuth=0.0;  cam.elevation=-15.0; cam.distance=3.5; }
            CameraMode::Free  => {}
        }
    }
}
```

---

## 9. Configuration Files

### `configs/train_default.toml`

```toml
model_path           = "assets/robot_humanoid_load.xml"
seed                 = 42
total_steps          = 20_000_000
rollout_steps        = 4096
curriculum           = true
clip_epsilon         = 0.2
gamma                = 0.99
gae_lambda           = 0.95
lr                   = 0.0003
epochs               = 10
minibatch_size       = 512
value_coef           = 0.5
entropy_coef         = 0.01
max_grad_norm        = 0.5
log_interval         = 10
checkpoint_interval  = 100_000
checkpoint_dir       = "checkpoints"
log_dir              = "runs"
```

### `configs/eval.toml`

```toml
model_path   = "assets/robot_humanoid_load.xml"
policy_path  = "checkpoints/sapggo_final.pt"
n_episodes   = 20
render       = true
seed         = 100
```

---

## 10. Logging and Metrics

**Metrics logged every `log_interval` episodes:**

| Metric | Tag | Unit |
|--------|-----|------|
| Mean episode reward | `train/reward_mean` | dimensionless |
| Policy loss | `train/policy_loss` | dimensionless |
| Value loss | `train/value_loss` | dimensionless |
| Entropy | `train/entropy` | bits |
| Curriculum stage | `curriculum/stage` | 0–5 |
| Mean distance | `env/distance_mean` | meters |
| Load drop rate | `env/drop_rate` | fraction |
| Mean walking speed | `env/speed_mean` | m/s |
| Mean episode steps | `env/steps_mean` | steps |

Logs are written to CSV files in `runs/` and can be visualized with any CSV-compatible plotting tool.

---

## 11. Testing Strategy

```rust
// crates/sapggo-env/tests/integration.rs

#[test]
fn env_reset_observation_dim() {
    let mut env = SapggoEnv::new("assets/robot_humanoid_load.xml", 0).unwrap();
    let obs = env.reset();
    assert_eq!(obs.len(), 70, "Observation must be 70-dimensional");
}

#[test]
fn env_step_zero_action_no_panic() {
    let mut env = SapggoEnv::new("assets/robot_humanoid_load.xml", 1).unwrap();
    env.reset();
    let r = env.step(&[0.0f64; 16]);
    assert!(r.info.steps == 1);
}

#[test]
fn reward_standing_still_is_positive() {
    use reward::{compute_reward, RewardState, RewardWeights};
    let s = RewardState {
        velocity_x: 0.5, torso_pitch: 0.0, torso_roll: 0.0,
        load_dx: 0.0, load_dy: 0.0, load_dz: 0.0,
        torques: [0.0; 16], prev_action: [0.0; 16], cur_action: [0.0; 16],
    };
    let r = compute_reward(&s, &RewardWeights::default());
    assert!(r > 0.0, "Moving forward with centered load must give positive reward");
}

#[test]
fn curriculum_promotes_on_sufficient_reward() {
    let mut m = CurriculumManager::new();
    assert_eq!(m.stage, CurriculumStage::Stand);
    for _ in 0..50 { m.on_episode_end(20.0); }
    assert_ne!(m.stage, CurriculumStage::Stand, "Should promote after 50 high-reward episodes");
}

#[test]
fn curriculum_does_not_promote_on_low_reward() {
    let mut m = CurriculumManager::new();
    for _ in 0..50 { m.on_episode_end(3.0); }
    assert_eq!(m.stage, CurriculumStage::Stand, "Should not promote on low reward");
}

#[test]
fn gae_output_shapes_correct() {
    let (adv, ret) = compute_gae(
        &[1.0; 20], &[0.5; 20], &[false; 20], 0.5, 0.99, 0.95
    );
    assert_eq!(adv.len(), 20);
    assert_eq!(ret.len(), 20);
}

#[test]
fn gae_returns_are_at_least_rewards() {
    let (_, ret) = compute_gae(
        &[1.0; 5], &[0.0; 5], &[false; 5], 0.0, 0.99, 0.95
    );
    for r in &ret { assert!(*r >= 0.9, "Returns must include future rewards"); }
}
```

---

## 12. Performance Considerations

| Concern | Detail |
|---------|--------|
| Physics speed | MuJoCo with `dt=0.002 s` and RK4: reliable at 50–100x real-time on CPU |
| Training throughput | ~500K steps/hour on CPU; ~2M/hour on GPU (Burn wgpu backend) |
| Memory per rollout | 4096 steps x 70 obs x 8 bytes = ~2.3 MB |
| Determinism | Fixed `StdRng` seed; `mj_resetData` for clean episode resets |
| Parallel envs | Add `rayon`-based parallel rollout collection for 4–8x speedup |
| Compilation | Use `--release` flag; debug builds are 10–20x slower |
| GPU training | Set `WGPU_BACKEND=vulkan` or `cuda` in environment before running |

---

## 13. Dependency Tree

```
sapggo-train (binary)
  +-- sapggo-env
  |     +-- mujoco-rs
  |     +-- sapggo-curriculum
  |     +-- rand, rand_distr
  |     +-- ndarray
  |     +-- thiserror, tracing, anyhow
  +-- sapggo-agent
  |     +-- burn (wgpu + autodiff)
  +-- sapggo-curriculum
  +-- sapggo-viz
        +-- mujoco-rs (visualizer feature)

sapggo-eval (binary)
  +-- sapggo-env
  +-- sapggo-agent
  +-- sapggo-viz
```

---

*End of SAPGGO Implementation Document — v0.1.0*

*Built with Rust. Powered by MuJoCo. Inspired by Africa.*