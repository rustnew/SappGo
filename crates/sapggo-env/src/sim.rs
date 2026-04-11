//! Simulation abstraction over MuJoCo.
//!
//! When the `mujoco` feature is enabled, this wraps the real MuJoCo C library
//! through `mujoco-rs`. Without the feature, a stub simulation is provided
//! that stores flat arrays and allows the full RL pipeline to compile and
//! run with zeroed physics (useful for testing logic without MuJoCo installed).

use std::path::Path;

use crate::error::{EnvError, EnvResult};
use crate::robot::N_JOINTS;

// ═══════════════════════════════════════════════════════════════════════
//  REAL MuJoCo backend (feature = "mujoco")
// ═══════════════════════════════════════════════════════════════════════
#[cfg(feature = "mujoco")]
mod backend {
    use super::*;
    use std::ffi::{CString, c_int};
    use std::ptr;
    use mujoco_rs::mujoco_c::*;

    /// Safe owner of a MuJoCo model + data pair.
    pub struct Simulation {
        model_ptr: *mut mjModel,
        data_ptr:  *mut mjData,
    }

    unsafe impl Send for Simulation {}

    impl Simulation {
        pub fn from_xml(path: &Path) -> EnvResult<Self> {
            let path_str = path.to_str().ok_or_else(|| EnvError::ModelLoad {
                path: path.display().to_string(),
                source: anyhow::anyhow!("Path contains invalid UTF-8"),
            })?;
            let c_path = CString::new(path_str).map_err(|e| EnvError::ModelLoad {
                path: path_str.to_owned(),
                source: e.into(),
            })?;
            let mut err_buf = [0i8; 500];
            let model_ptr = unsafe {
                mj_loadXML(c_path.as_ptr(), ptr::null(), err_buf.as_mut_ptr(), err_buf.len() as c_int)
            };
            if model_ptr.is_null() {
                let msg = c_chars_to_string(&err_buf);
                return Err(EnvError::ModelLoad { path: path_str.to_owned(), source: anyhow::anyhow!("{}", msg) });
            }
            let data_ptr = unsafe { mj_makeData(model_ptr.as_ref().unwrap()) };
            if data_ptr.is_null() {
                unsafe { mj_deleteModel(model_ptr); }
                return Err(EnvError::ModelLoad { path: path_str.to_owned(), source: anyhow::anyhow!("mj_makeData returned null") });
            }
            Ok(Self { model_ptr, data_ptr })
        }

        #[inline] fn model(&self) -> &mjModel { unsafe { &*self.model_ptr } }
        #[inline] fn data(&self) -> &mjData   { unsafe { &*self.data_ptr } }
        #[inline] fn data_mut(&mut self) -> &mut mjData { unsafe { &mut *self.data_ptr } }

        pub fn reset_data(&mut self) { unsafe { mj_resetData(self.model(), self.data_mut()); } }
        pub fn step(&mut self)       { unsafe { mj_step(self.model(), self.data_mut()); } }
        pub fn forward(&mut self)    { unsafe { mj_forward(self.model(), self.data_mut()); } }

        pub fn body_id(&self, name: &str) -> Option<usize>     { self.name2id(mjtObj::mjOBJ_BODY, name) }
        pub fn joint_id(&self, name: &str) -> Option<usize>    { self.name2id(mjtObj::mjOBJ_JOINT, name) }
        pub fn sensor_id(&self, name: &str) -> Option<usize>   { self.name2id(mjtObj::mjOBJ_SENSOR, name) }
        pub fn actuator_id(&self, name: &str) -> Option<usize> { self.name2id(mjtObj::mjOBJ_ACTUATOR, name) }

        fn name2id(&self, obj_type: mjtObj, name: &str) -> Option<usize> {
            let c_name = CString::new(name).ok()?;
            let id = unsafe { mj_name2id(self.model(), obj_type as i32, c_name.as_ptr()) };
            if id < 0 { None } else { Some(id as usize) }
        }

        #[inline] pub fn joint_qpos_addr(&self, jid: usize) -> usize { unsafe { *self.model().jnt_qposadr.add(jid) as usize } }
        #[inline] pub fn joint_qvel_addr(&self, jid: usize) -> usize { unsafe { *self.model().jnt_dofadr.add(jid) as usize } }
        #[inline] pub fn sensor_addr(&self, sid: usize) -> usize     { unsafe { *self.model().sensor_adr.add(sid) as usize } }

        #[inline] pub fn qpos(&self, a: usize) -> f64              { unsafe { *self.data().qpos.add(a) } }
        #[inline] pub fn set_qpos(&mut self, a: usize, v: f64)     { unsafe { *self.data_mut().qpos.add(a) = v; } }
        #[inline] pub fn qvel(&self, a: usize) -> f64              { unsafe { *self.data().qvel.add(a) } }
        #[inline] pub fn body_xpos(&self, bid: usize, c: usize) -> f64  { unsafe { *self.data().xpos.add(bid * 3 + c) } }
        #[inline] pub fn body_xquat(&self, bid: usize, c: usize) -> f64 { unsafe { *self.data().xquat.add(bid * 4 + c) } }
        #[inline] pub fn sensordata(&self, a: usize) -> f64        { unsafe { *self.data().sensordata.add(a) } }
        #[inline] pub fn actuator_force(&self, i: usize) -> f64    { unsafe { *self.data().actuator_force.add(i) } }
        #[inline] pub fn set_ctrl(&mut self, i: usize, v: f64)     { unsafe { *self.data_mut().ctrl.add(i) = v; } }
        #[inline] pub fn set_xfrc_applied(&mut self, bid: usize, c: usize, v: f64) {
            unsafe { *self.data_mut().xfrc_applied.add(bid * 6 + c) = v; }
        }
    }

    impl Drop for Simulation {
        fn drop(&mut self) {
            unsafe {
                if !self.data_ptr.is_null()  { mj_deleteData(self.data_ptr); }
                if !self.model_ptr.is_null() { mj_deleteModel(self.model_ptr); }
            }
        }
    }

    fn c_chars_to_string(buf: &[i8]) -> String {
        let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  STUB backend (no MuJoCo C library required)
// ═══════════════════════════════════════════════════════════════════════
#[cfg(not(feature = "mujoco"))]
mod backend {
    use super::*;

    /// Maximum number of bodies / joints / sensors in the stub.
    const MAX_BODIES:  usize = 32;
    const MAX_QPOS:    usize = 128;
    const MAX_QVEL:    usize = 128;
    const MAX_SENSOR:  usize = 64;
    const MAX_CTRL:    usize = N_JOINTS;
    const MAX_FORCE:   usize = N_JOINTS;

    /// Stub simulation that stores flat arrays.
    /// Physics calls are no-ops — the arrays stay zeroed.
    /// This lets the full RL pipeline compile and run without MuJoCo.
    pub struct Simulation {
        qpos:          Vec<f64>,
        qvel:          Vec<f64>,
        xpos:          Vec<f64>,
        xquat:         Vec<f64>,
        sensordata:    Vec<f64>,
        ctrl:          Vec<f64>,
        actuator_force: Vec<f64>,
        xfrc_applied:  Vec<f64>,
        // Name → index maps (populated from the XML joint/body/sensor names)
        body_map:   Vec<(&'static str, usize)>,
        joint_map:  Vec<(&'static str, usize)>,
        sensor_map: Vec<(&'static str, usize)>,
    }

    impl Simulation {
        pub fn from_xml(path: &Path) -> EnvResult<Self> {
            if !path.exists() {
                return Err(EnvError::ModelLoad {
                    path: path.display().to_string(),
                    source: anyhow::anyhow!("Model file not found (stub mode — MuJoCo feature disabled)"),
                });
            }
            tracing::warn!("Running in STUB mode — MuJoCo feature is disabled. Physics is a no-op.");

            // Pre-populate name maps matching the SAPGGO XML model
            let body_map = vec![
                ("world", 0), ("torso", 1), ("neck", 2), ("head", 3), ("load", 4),
                ("upper_leg_L", 5), ("lower_leg_L", 6), ("foot_L", 7),
                ("upper_leg_R", 8), ("lower_leg_R", 9), ("foot_R", 10),
            ];
            let joint_map: Vec<(&str, usize)> = crate::robot::JOINT_NAMES.iter()
                .enumerate()
                .map(|(i, &n)| (n, i))
                .chain(std::iter::once(("load_joint", N_JOINTS)))
                .collect();
            let sensor_map = vec![
                ("torso_quat", 0), ("torso_gyro", 1), ("torso_acc", 2),
                ("head_quat", 3), ("load_pos", 4), ("load_quat", 5),
                ("load_gyro", 6), ("load_acc", 7),
                ("foot_L_force", 8), ("foot_R_force", 9),
            ];

            Ok(Self {
                qpos:           vec![0.0; MAX_QPOS],
                qvel:           vec![0.0; MAX_QVEL],
                xpos:           vec![0.0; MAX_BODIES * 3],
                xquat:          vec![0.0; MAX_BODIES * 4],
                sensordata:     vec![0.0; MAX_SENSOR],
                ctrl:           vec![0.0; MAX_CTRL],
                actuator_force: vec![0.0; MAX_FORCE],
                xfrc_applied:   vec![0.0; MAX_BODIES * 6],
                body_map,
                joint_map,
                sensor_map,
            })
        }

        pub fn reset_data(&mut self) {
            self.qpos.fill(0.0);
            self.qvel.fill(0.0);
            self.xpos.fill(0.0);
            self.xquat.fill(0.0);
            self.sensordata.fill(0.0);
            self.ctrl.fill(0.0);
            self.actuator_force.fill(0.0);
            self.xfrc_applied.fill(0.0);
            // Set identity quaternions for all bodies
            for i in 0..(self.xquat.len() / 4) {
                self.xquat[i * 4] = 1.0;
            }
            // Set default torso height
            if let Some(tid) = self.body_id("torso") {
                self.xpos[tid * 3 + 2] = 1.35; // standing height
            }
            if let Some(hid) = self.body_id("head") {
                self.xpos[hid * 3 + 2] = 1.72;
            }
        }

        pub fn step(&mut self)    { /* no-op in stub mode */ }
        pub fn forward(&mut self) { /* no-op in stub mode */ }

        pub fn body_id(&self, name: &str) -> Option<usize> {
            self.body_map.iter().find(|(n, _)| *n == name).map(|(_, id)| *id)
        }
        pub fn joint_id(&self, name: &str) -> Option<usize> {
            self.joint_map.iter().find(|(n, _)| *n == name).map(|(_, id)| *id)
        }
        pub fn sensor_id(&self, name: &str) -> Option<usize> {
            self.sensor_map.iter().find(|(n, _)| *n == name).map(|(_, id)| *id)
        }
        #[allow(dead_code)]
        pub fn actuator_id(&self, _name: &str) -> Option<usize> { Some(0) }

        #[inline] pub fn joint_qpos_addr(&self, jid: usize) -> usize { jid }
        #[inline] pub fn joint_qvel_addr(&self, jid: usize) -> usize { jid }
        #[inline] pub fn sensor_addr(&self, sid: usize) -> usize     { sid * 3 }

        #[inline] pub fn qpos(&self, a: usize) -> f64           { self.qpos.get(a).copied().unwrap_or(0.0) }
        #[inline] pub fn set_qpos(&mut self, a: usize, v: f64)  { if a < self.qpos.len() { self.qpos[a] = v; } }
        #[inline] pub fn qvel(&self, a: usize) -> f64           { self.qvel.get(a).copied().unwrap_or(0.0) }
        #[inline] pub fn body_xpos(&self, bid: usize, c: usize) -> f64  { self.xpos.get(bid * 3 + c).copied().unwrap_or(0.0) }
        #[inline] pub fn body_xquat(&self, bid: usize, c: usize) -> f64 { self.xquat.get(bid * 4 + c).copied().unwrap_or(if c == 0 { 1.0 } else { 0.0 }) }
        #[inline] pub fn sensordata(&self, a: usize) -> f64     { self.sensordata.get(a).copied().unwrap_or(0.0) }
        #[inline] pub fn actuator_force(&self, i: usize) -> f64 { self.actuator_force.get(i).copied().unwrap_or(0.0) }
        #[inline] pub fn set_ctrl(&mut self, i: usize, v: f64)  { if i < self.ctrl.len() { self.ctrl[i] = v; } }
        #[inline] pub fn set_xfrc_applied(&mut self, bid: usize, c: usize, v: f64) {
            let idx = bid * 6 + c;
            if idx < self.xfrc_applied.len() { self.xfrc_applied[idx] = v; }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Re-export the chosen backend
// ═══════════════════════════════════════════════════════════════════════
pub use backend::Simulation;
