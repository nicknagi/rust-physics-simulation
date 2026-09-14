//! Verifies the cursor well actually attracts and repels, on the GPU path.
//!
//! The well sits off-centre so attraction is unambiguous -- a well at the centre
//! would be indistinguishable from the ring simply falling inward.
use simulation::gpu::{GpuContext, GpuSim};
use simulation::{Config, DiscSpawn, Sim};

fn main() {
    let ctx = GpuContext::new().expect("no GPU");
    ctx.device
        .on_uncaptured_error(std::sync::Arc::new(|e| eprintln!("WGPU ERROR: {e}")));

    let cfg_for = || Config {
        width: 1600.0,
        height: 900.0,
        radius: 0.4,
        gravity: true,
        parallel: true,
        mass: 1e16 * 1.0e-5,
        softening: 8.0,
        central_gm: 2.0e7,
        ..Config::default()
    };
    // Well sits off-centre so attraction is unambiguous (not just infall).
    let (wx, wy) = (1150.0f32, 450.0f32);

    let mean_dist = |p: &[[f32; 2]]| -> f32 {
        p.iter()
            .map(|q| ((q[0] - wx).powi(2) + (q[1] - wy).powi(2)).sqrt())
            .sum::<f32>()
            / p.len() as f32
    };

    println!("well at ({wx}, {wy}); mean particle distance to it after 3s\n");
    for &(label, gm) in &[("off", 0.0f32), ("attract", 1.0e8), ("repel", -1.0e8)] {
        let cfg = cfg_for();
        let mut seed = Sim::new(cfg);
        let fastest = seed.spawn_orbital_disc(DiscSpawn {
            count: 4000,
            seed: 12345,
            inner_radius: 101.0,
            outer_radius: 297.0,
            clockwise: false,
        });
        let mut cfg = cfg;
        cfg.max_speed = fastest * 4.0;
        let mut gpu = GpuSim::new(
            &ctx.device,
            cfg,
            &seed.px,
            &seed.py,
            &seed.vx,
            &seed.vy,
            &seed.color,
        );

        let d0 = mean_dist(&gpu.read_positions(&ctx.device, &ctx.queue));
        gpu.set_well(gm, wx, wy);
        gpu.run_steps(&ctx.device, &ctx.queue, 1.0 / 240.0, 720);
        let p = gpu.read_positions(&ctx.device, &ctx.queue);
        let d1 = mean_dist(&p);
        let nan = p
            .iter()
            .filter(|q| !q[0].is_finite() || !q[1].is_finite())
            .count();
        println!(
            "  {:>8} (gm {:>9.1e}): {:.1} -> {:.1} px  ({:+.1}%)   nan={}",
            label,
            gm,
            d0,
            d1,
            (d1 - d0) / d0 * 100.0,
            nan
        );
    }
}
