//! SAPGGO Visual Viewer — opens a 3D window showing the humanoid robot.
//!
//! Usage:
//!   cargo run --bin sapggo-viewer --features sapggo-viz/mujoco
//!
//! Controls:
//!   - Left-click + drag: rotate camera
//!   - Right-click + drag: pan camera
//!   - Scroll: zoom in/out
//!   - Ctrl + double-click: track a body
//!   - Esc: free camera
//!   - Ctrl+Q: quit

use mujoco_rs::prelude::MjModel;
use mujoco_rs::viewer::MjViewer;
use std::time::{Duration, Instant};

const MODEL_PATH: &str = "assets/robot_humanoid_load.xml";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════╗");
    println!("║       SAPGGO — MuJoCo 3D Viewer         ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  Left-click + drag  → rotate camera     ║");
    println!("║  Right-click + drag → pan camera        ║");
    println!("║  Scroll             → zoom              ║");
    println!("║  Ctrl+double-click  → track body        ║");
    println!("║  Esc                → free camera       ║");
    println!("║  Ctrl+Q             → quit              ║");
    println!("╚══════════════════════════════════════════╝");

    // Load MuJoCo model
    let model = MjModel::from_xml(MODEL_PATH)?;
    let mut data = model.make_data();

    let m = model.ffi();
    println!("Model loaded: {} bodies, {} joints, {} actuators",
        m.nbody, m.njnt, m.nu);

    // Launch the passive viewer (OpenGL window)
    let mut viewer = MjViewer::launch_passive(&model, 1000)
        .map_err(|e| format!("Failed to launch viewer: {:?}", e))?;

    println!("Viewer launched! Running simulation...");

    let _timestep = m.opt.timestep;
    let mut step_count = 0u64;

    // Main simulation + rendering loop
    while viewer.running() {
        let start = Instant::now();

        // Step physics (10 sub-steps per frame for smooth rendering)
        for _ in 0..10 {
            data.step();
            step_count += 1;
        }

        // Sync viewer with simulation state
        viewer.sync(&mut data);

        // Print status every 5000 steps
        if step_count % 5000 == 0 {
            // Read torso z from raw xpos (body 1 = root_body)
            let torso_z = unsafe { *data.ffi().xpos.add(1 * 3 + 2) };
            println!("Step {step_count:>8} | torso_z = {torso_z:.3} m");
        }

        // Cap to ~60 FPS
        let elapsed = start.elapsed();
        if elapsed < Duration::from_millis(16) {
            std::thread::sleep(Duration::from_millis(16) - elapsed);
        }
    }

    println!("Viewer closed after {step_count} steps.");
    Ok(())
}
