//! Two whirl recipes, and the trade-off between them.
//!
//! Suppressing the collisional spreading of a ring by lowering the central mass
//! and the particle count works -- the ring becomes permanently stable -- but it
//! spends the two properties that *are* the visual: rotation rate and density.
//! Decoupling the collision radius from the drawn size escapes the trade-off:
//! near-point masses barely collide, so a fast, dense ring stays coherent.
use simulation::{Config, DiscSpawn, Sim};

struct Recipe {
    name: &'static str,
    n: usize,
    radius: f32,
    gm: f32,
    inner: f32,
    outer: f32,
    scale: f32,
    dot: f32,
}

fn main() {
    let recipes = [
        Recipe {
            name: "stable, slow",
            n: 800,
            radius: 2.0,
            gm: 5.0e5,
            inner: 120.0,
            outer: 350.0,
            scale: 0.0,
            dot: 2.0,
        },
        Recipe {
            name: "--whirl",
            n: 4000,
            radius: 0.4,
            gm: 2.0e7,
            inner: 101.0,
            outer: 297.0,
            scale: 1.0e-5,
            dot: 1.6,
        },
    ];

    for r in &recipes {
        let cfg = Config {
            width: 1600.0,
            height: 1600.0,
            radius: r.radius,
            gravity: r.scale > 0.0,
            parallel: true,
            mass: if r.scale > 0.0 { 1e16 * r.scale } else { 1.0 },
            softening: if r.scale > 0.0 { 8.0 } else { 4.0 },
            central_gm: r.gm,
            ..Config::default()
        };
        let mut sim = Sim::new(cfg);
        let fastest = sim.spawn_orbital_disc(DiscSpawn {
            count: r.n,
            seed: 4242,
            inner_radius: r.inner,
            outer_radius: r.outer,
            clockwise: false,
        });
        sim.cfg.max_speed = fastest * 4.0;
        let (cx, cy) = (800.0f32, 800.0f32);

        // Mean angular velocity of the ensemble: sum(L) / sum(m r^2).
        let omega = |s: &Sim| -> f64 {
            let (mut l, mut i) = (0.0f64, 0.0f64);
            for k in 0..s.len() {
                let dx = (s.px[k] - cx) as f64;
                let dy = (s.py[k] - cy) as f64;
                l += dx * s.vy[k] as f64 - dy * s.vx[k] as f64;
                i += dx * dx + dy * dy;
            }
            l / i
        };
        let period_at =
            |rad: f32| 2.0 * std::f64::consts::PI * rad as f64 / (r.gm as f64 / rad as f64).sqrt();
        let retained = |s: &Sim| -> f32 {
            let lo = r.inner * 0.82;
            let hi = r.outer * 1.11;
            let k = (0..s.len())
                .filter(|&i| {
                    let d = ((s.px[i] - cx).powi(2) + (s.py[i] - cy).powi(2)).sqrt();
                    d >= lo && d < hi
                })
                .count();
            k as f32 / s.len() as f32 * 100.0
        };

        let w0 = omega(&sim);
        println!("=== {} ===", r.name);
        println!(
            "  {} particles, collision radius {}, drawn radius {}",
            r.n, r.radius, r.dot
        );
        println!(
            "  central G*M {:.1e}, annulus {:.0}..{:.0}",
            r.gm, r.inner, r.outer
        );
        println!(
            "  orbital period: inner {:.1}s, outer {:.1}s",
            period_at(r.inner),
            period_at(r.outer)
        );
        println!(
            "  revolutions in 30s: inner {:.1}, outer {:.1}",
            30.0 / period_at(r.inner),
            30.0 / period_at(r.outer)
        );
        println!(
            "  mean angular speed {:.4} rad/s  ({:.2} rev per 30s)",
            w0,
            w0.abs() * 30.0 / (2.0 * std::f64::consts::PI)
        );
        println!(
            "  drawn area (total ink) {:.0} px^2",
            r.n as f32 * std::f32::consts::PI * r.dot * r.dot
        );
        let mut line = String::new();
        for _ in 0..6 {
            for _ in 0..2400 {
                sim.step(1.0 / 240.0);
            }
            line.push_str(&format!(" {:>4.0}%", retained(&sim)));
        }
        println!("  retained 10s..60s:{}\n", line);
    }
}
