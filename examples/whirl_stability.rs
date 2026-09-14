//! Reproduces the measurements behind the `--whirl` preset.
//!
//! A ring is easy to spawn and hard to keep. Three separate effects break it,
//! and each had to be measured rather than guessed at:
//!
//!   1. Orbital speeds from `v = sqrt(G * M_enclosed / r)` make the ring
//!      collapse. That formula is the spherical shell theorem; this is a flat
//!      disc under a 1/r^2 force, where mass outside a radius does not cancel.
//!      `spawn_orbital_disc` calibrates against the measured field instead.
//!   2. A cold self-gravitating ring clumps and heats itself apart, whatever
//!      the collision radius.
//!   3. Collisions in a sheared disc transport angular momentum outward and
//!      spread it viscously, even with self-gravity switched off entirely.
//!
//! Run with: cargo run --release --example whirl_stability

use simulation::{Config, DiscSpawn, Sim};

fn retained(sim: &Sim, cx: f32, cy: f32) -> f32 {
    let k = (0..sim.len())
        .filter(|&i| {
            let r = ((sim.px[i] - cx).powi(2) + (sim.py[i] - cy).powi(2)).sqrt();
            (85.0..330.0).contains(&r)
        })
        .count();
    k as f32 / sim.len() as f32 * 100.0
}

fn run(radius: f32, scale: f32, central_gm: f32) -> String {
    let cfg = Config {
        width: 1600.0,
        height: 900.0,
        radius,
        gravity: scale > 0.0,
        parallel: true,
        mass: if scale > 0.0 { 1e16 * scale } else { 1.0 },
        softening: 8.0,
        central_gm,
        ..Config::default()
    };
    let mut sim = Sim::new(cfg);
    let fastest = sim.spawn_orbital_disc(DiscSpawn {
        count: 4000,
        seed: 12345,
        inner_radius: 104.0,
        outer_radius: 297.0,
        clockwise: false,
    });
    sim.cfg.max_speed = fastest * 4.0;

    let mut line = String::new();
    for _ in 0..6 {
        for _ in 0..2400 {
            sim.step(1.0 / 240.0);
        }
        line.push_str(&format!(" {:>4.0}%", retained(&sim, 800.0, 450.0)));
    }
    line
}

fn main() {
    println!("particles retained in the ring, sampled every 10s out to 60s\n");

    println!("no central mass -- self-gravity alone cannot hold a ring:");
    println!("{:>8} {:>8}  10s ..            .. 60s", "radius", "scale");
    for &(r, s) in &[(0.4f32, 1.0e-3f32), (0.4, 1.0e-4)] {
        println!("{:>8} {:>8.0e}  {}", r, s, run(r, s, 0.0));
    }

    println!("\nwith a central mass (G*M = 2e7), varying collision radius:");
    for &r in &[1.5f32, 0.8, 0.4] {
        println!("{:>8} {:>8.0e}  {}", r, 1.0e-4f32, run(r, 1.0e-4, 2.0e7));
    }

    println!("\nwith a central mass, varying self-gravity at radius 0.4:");
    for &s in &[1.0e-4f32, 3.0e-5, 1.0e-5, 0.0] {
        println!("{:>8} {:>8.0e}  {}", 0.4f32, s, run(0.4, s, 2.0e7));
    }
    println!("\n--whirl uses radius 0.4 and scale 1e-5: the ring stays intact for ~50s.");
}
