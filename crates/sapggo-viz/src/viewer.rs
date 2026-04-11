use crate::camera::CameraMode;

/// Camera preset values for a given mode.
#[derive(Debug, Clone, Copy)]
pub struct CameraPreset {
    pub azimuth:   f64,
    pub elevation: f64,
    pub distance:  f64,
}

/// Wrapper around the MuJoCo camera for the SAPGGO visualizer.
///
/// Provides camera mode cycling and preset camera configurations
/// for observing the humanoid during training or evaluation.
pub struct SapggoViewer {
    mode: CameraMode,
}

impl SapggoViewer {
    /// Creates a new viewer starting in side-view mode.
    pub fn new() -> Self {
        Self {
            mode: CameraMode::Side,
        }
    }

    /// Returns the current camera mode.
    #[inline]
    pub fn mode(&self) -> CameraMode {
        self.mode
    }

    /// Cycles to the next camera mode.
    pub fn cycle_camera(&mut self) {
        let next = self.mode.next();
        tracing::info!(from = %self.mode, to = %next, "Camera mode changed");
        self.mode = next;
    }

    /// Returns the camera preset for the current mode, or `None` in `Free` mode.
    pub fn preset(&self) -> Option<CameraPreset> {
        match self.mode {
            CameraMode::Side => Some(CameraPreset {
                azimuth: 90.0, elevation: -15.0, distance: 4.0,
            }),
            CameraMode::Front => Some(CameraPreset {
                azimuth: 0.0, elevation: -15.0, distance: 3.5,
            }),
            CameraMode::Free => None,
        }
    }

    /// Applies the current camera preset to a MuJoCo `mjvCamera` struct.
    ///
    /// Only available when the `mujoco` feature is enabled.
    #[cfg(feature = "mujoco")]
    pub fn apply_to(&self, cam: &mut mujoco_rs::prelude::MjvCamera) {
        if let Some(p) = self.preset() {
            cam.azimuth   = p.azimuth;
            cam.elevation = p.elevation;
            cam.distance  = p.distance;
        }
    }
}

impl Default for SapggoViewer {
    fn default() -> Self {
        Self::new()
    }
}
