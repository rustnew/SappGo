use rand::Rng;

use crate::sensor::SensorIndex;
use crate::sim::Simulation;

/// Maximum random placement offset (meters) applied to the load at episode
/// reset for domain randomization.
const MAX_PLACEMENT_OFFSET: f64 = 0.015;

/// Drop detection threshold: if the load's z-position drops more than this
/// distance below the head, the load is considered fallen.
pub const LOAD_DROP_THRESHOLD: f64 = -0.05;

/// Places the load on the robot's head with a small random XY offset.
///
/// The MuJoCo XML already positions the load body at `pos="0 0 0.22"` relative
/// to the head. This function applies a small perturbation to the load's qpos
/// for domain randomization.
pub fn place_load_on_head(
    sim: &mut Simulation,
    idx: &SensorIndex,
    rng: &mut impl Rng,
) {
    let dx = (rng.gen::<f64>() - 0.5) * 2.0 * MAX_PLACEMENT_OFFSET;
    let dy = (rng.gen::<f64>() - 0.5) * 2.0 * MAX_PLACEMENT_OFFSET;

    // The load body has a freejoint ("load_joint"); its qpos occupies 7 values
    // (3 pos + 4 quat). We need to run mj_forward first so xpos is valid.
    sim.forward();

    if let Some(load_jnt_id) = sim.joint_id("load_joint") {
        let addr = sim.joint_qpos_addr(load_jnt_id);

        let head_x = sim.body_xpos(idx.head_body_id, 0);
        let head_y = sim.body_xpos(idx.head_body_id, 1);
        let head_z = sim.body_xpos(idx.head_body_id, 2);

        sim.set_qpos(addr,     head_x + dx);
        sim.set_qpos(addr + 1, head_y + dy);
        // z: head top + offset from XML (≈ 0.22 m)
        sim.set_qpos(addr + 2, head_z + 0.22);
        // Identity quaternion (w, x, y, z)
        sim.set_qpos(addr + 3, 1.0);
        sim.set_qpos(addr + 4, 0.0);
        sim.set_qpos(addr + 5, 0.0);
        sim.set_qpos(addr + 6, 0.0);
    }
}

/// Returns `true` if the load has dropped below the head.
#[inline]
pub fn is_dropped(sim: &Simulation, idx: &SensorIndex) -> bool {
    let head_z = sim.body_xpos(idx.head_body_id, 2);
    let load_z = sim.body_xpos(idx.load_body_id, 2);
    (load_z - head_z) < LOAD_DROP_THRESHOLD
}
