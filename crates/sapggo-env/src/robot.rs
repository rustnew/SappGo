/// Number of actuated joints in the humanoid model.
pub const N_JOINTS: usize = 24;

/// Observation vector dimensionality.
///
/// Layout (92 dimensions):
///   [0..24]  joint angles
///   [24..48] joint velocities
///   [48..52] torso quaternion
///   [52..55] torso gyro
///   [55..58] foot L force (3)
///   [58..61] foot R force (3)
///   [61..64] load offset
///   [64..67] load angular velocity
///   [67..91] previous action
///   [91]     target velocity
pub const OBS_DIM: usize = 92;

/// Action vector dimensionality (one per actuated joint).
pub const ACT_DIM: usize = 24;

/// Ordered list of actuated joint names.
/// The index matches the position in the action vector.
pub const JOINT_NAMES: [&str; N_JOINTS] = [
    "hip_flex_L",   "hip_add_L",    "hip_rot_L",
    "knee_L",
    "ankle_flex_L", "ankle_inv_L",
    "hip_flex_R",   "hip_add_R",    "hip_rot_R",
    "knee_R",
    "ankle_flex_R", "ankle_inv_R",
    "torso_flex",   "torso_lat",
    "neck_tilt",    "neck_rot",
    "shoulder_flex_L", "shoulder_add_L", "shoulder_rot_L",
    "elbow_flex_L",
    "shoulder_flex_R", "shoulder_add_R", "shoulder_rot_R",
    "elbow_flex_R",
];

/// Maximum torque (Nm) per actuator, same order as [`JOINT_NAMES`].
pub const MAX_TORQUE: [f64; N_JOINTS] = [
    150.0, 150.0, 150.0,   // hip L
    200.0,                  // knee L
     80.0,  80.0,           // ankle L
    150.0, 150.0, 150.0,   // hip R
    200.0,                  // knee R
     80.0,  80.0,           // ankle R
    120.0, 120.0,           // torso
     40.0,  40.0,           // neck
     80.0,  80.0,  60.0,   // shoulder L
     60.0,                  // elbow L
     80.0,  80.0,  60.0,   // shoulder R
     60.0,                  // elbow R
];

/// Action smoothing coefficient (low-pass filter alpha for previous value).
pub const ACTION_SMOOTH_ALPHA: f64 = 0.8;

/// Action smoothing coefficient (low-pass filter beta for new value).
pub const ACTION_SMOOTH_BETA: f64 = 0.2;
