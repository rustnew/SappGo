use ndarray::Array1;
use rand::Rng;

use crate::error::{EnvError, EnvResult};
use crate::noise::GaussianNoise;
use crate::robot::{JOINT_NAMES, N_JOINTS, OBS_DIM};
use crate::sim::Simulation;

/// Observation vector layout (98 dimensions):
///
/// | Range     | Content                                    |
/// |-----------|--------------------------------------------|
/// | `[0..24]` | Joint angles                               |
/// | `[24..48]`| Joint velocities                           |
/// | `[48..52]`| Torso quaternion (w, x, y, z)              |
/// | `[52..55]`| Torso angular velocity (gyro)              |
/// | `[55..58]`| Torso linear acceleration                  |
/// | `[58..61]`| Foot L contact forces (3)                  |
/// | `[61..64]`| Foot R contact forces (3)                  |
/// | `[64..67]`| Load offset from head center (dx, dy, dz)  |
/// | `[67..70]`| Load angular velocity                      |
/// | `[70..73]`| Load linear acceleration                   |
/// | `[73..97]`| Previous action (24 joints)                |
/// | `[97]`    | Target forward velocity (curriculum)       |

/// Cached sensor/body/joint indices to avoid repeated name lookups on every step.
pub struct SensorIndex {
    pub torso_body_id:    usize,
    pub head_body_id:     usize,
    pub load_body_id:     usize,
    pub torso_gyro_addr:  usize,
    pub torso_acc_addr:   usize,
    pub foot_l_addr:      usize,
    pub foot_r_addr:      usize,
    pub load_gyro_addr:   usize,
    pub load_acc_addr:    usize,
    pub joint_qpos_addrs: [usize; N_JOINTS],
    pub joint_qvel_addrs: [usize; N_JOINTS],
}

impl SensorIndex {
    /// Resolves all named indices once at environment creation.
    pub fn resolve(sim: &Simulation) -> EnvResult<Self> {
        let torso_body_id = sim.body_id("torso")
            .ok_or_else(|| EnvError::BodyNotFound { name: "torso".into() })?;
        let head_body_id = sim.body_id("head")
            .ok_or_else(|| EnvError::BodyNotFound { name: "head".into() })?;
        let load_body_id = sim.body_id("load")
            .ok_or_else(|| EnvError::BodyNotFound { name: "load".into() })?;

        let torso_gyro_sid = sim.sensor_id("torso_gyro")
            .ok_or_else(|| EnvError::SensorNotFound { name: "torso_gyro".into() })?;
        let torso_acc_sid = sim.sensor_id("torso_acc")
            .ok_or_else(|| EnvError::SensorNotFound { name: "torso_acc".into() })?;
        let foot_l_sid = sim.sensor_id("foot_L_force")
            .ok_or_else(|| EnvError::SensorNotFound { name: "foot_L_force".into() })?;
        let foot_r_sid = sim.sensor_id("foot_R_force")
            .ok_or_else(|| EnvError::SensorNotFound { name: "foot_R_force".into() })?;
        let load_gyro_sid = sim.sensor_id("load_gyro")
            .ok_or_else(|| EnvError::SensorNotFound { name: "load_gyro".into() })?;
        let load_acc_sid = sim.sensor_id("load_acc")
            .ok_or_else(|| EnvError::SensorNotFound { name: "load_acc".into() })?;

        let torso_gyro_addr = sim.sensor_addr(torso_gyro_sid);
        let torso_acc_addr  = sim.sensor_addr(torso_acc_sid);
        let foot_l_addr     = sim.sensor_addr(foot_l_sid);
        let foot_r_addr     = sim.sensor_addr(foot_r_sid);
        let load_gyro_addr  = sim.sensor_addr(load_gyro_sid);
        let load_acc_addr   = sim.sensor_addr(load_acc_sid);

        let mut joint_qpos_addrs = [0usize; N_JOINTS];
        let mut joint_qvel_addrs = [0usize; N_JOINTS];
        for (i, jname) in JOINT_NAMES.iter().enumerate() {
            let jid = sim.joint_id(jname)
                .ok_or_else(|| EnvError::JointNotFound { name: (*jname).to_string() })?;
            joint_qpos_addrs[i] = sim.joint_qpos_addr(jid);
            joint_qvel_addrs[i] = sim.joint_qvel_addr(jid);
        }

        Ok(Self {
            torso_body_id,
            head_body_id,
            load_body_id,
            torso_gyro_addr,
            torso_acc_addr,
            foot_l_addr,
            foot_r_addr,
            load_gyro_addr,
            load_acc_addr,
            joint_qpos_addrs,
            joint_qvel_addrs,
        })
    }
}

/// Extracts a full observation vector from the current simulation state.
///
/// Gaussian noise with the given sigma is added to all sensor readings
/// for domain-randomization robustness.
pub fn extract_observation(
    sim:         &Simulation,
    idx:         &SensorIndex,
    prev_action: &[f64; N_JOINTS],
    target_vel:  f64,
    noise:       &GaussianNoise,
    rng:         &mut impl Rng,
) -> Array1<f64> {
    let mut obs = Array1::<f64>::zeros(OBS_DIM);

    // Joint angles and velocities [0..48]
    for i in 0..N_JOINTS {
        obs[i]            = sim.qpos(idx.joint_qpos_addrs[i]) + noise.sample(rng);
        obs[N_JOINTS + i] = sim.qvel(idx.joint_qvel_addrs[i]) + noise.sample(rng);
    }

    // Torso quaternion [48..52]
    for j in 0..4 {
        obs[48 + j] = sim.body_xquat(idx.torso_body_id, j) + noise.sample(rng);
    }

    // Torso gyro [52..55]
    for j in 0..3 {
        obs[52 + j] = sim.sensordata(idx.torso_gyro_addr + j) + noise.sample(rng);
    }

    // Torso accelerometer [55..58]
    for j in 0..3 {
        obs[55 + j] = sim.sensordata(idx.torso_acc_addr + j) + noise.sample(rng);
    }

    // Foot contact forces [58..64] (3 values per foot: fx, fy, fz)
    for j in 0..3 {
        obs[58 + j] = sim.sensordata(idx.foot_l_addr + j) + noise.sample(rng);
        obs[61 + j] = sim.sensordata(idx.foot_r_addr + j) + noise.sample(rng);
    }

    // Load offset from head center [64..67]
    for j in 0..3 {
        let head_j = sim.body_xpos(idx.head_body_id, j);
        let load_j = sim.body_xpos(idx.load_body_id, j);
        obs[64 + j] = (load_j - head_j) + noise.sample(rng);
    }

    // Load angular velocity [67..70]
    for j in 0..3 {
        obs[67 + j] = sim.sensordata(idx.load_gyro_addr + j) + noise.sample(rng);
    }

    // Load accelerometer [70..73]
    for j in 0..3 {
        obs[70 + j] = sim.sensordata(idx.load_acc_addr + j) + noise.sample(rng);
    }

    // Previous action [73..97]
    for i in 0..N_JOINTS {
        obs[73 + i] = prev_action[i];
    }

    // Target velocity [97]
    obs[97] = target_vel;

    obs
}
