//! Headless benchmark. No window, no vsync -- measures the physics only.
//!
//! Usage:
//!   bench [--particles N] [--steps S] [--gravity] [--no-parallel]
//!         [--threads T] [--radius R] [--width W] [--height H] [--sweep]

use std::time::Instant;

use simulation::{Config, Sim};

struct Args {
    particles: usize,
    steps: usize,
    gravity: bool,
    parallel: bool,
    threads: usize,
    radius: f32,
    width: f32,
    height: f32,
    sweep: bool,
    auto_domain: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            particles: 10_000,
            steps: 300,
            gravity: false,
            parallel: true,
            threads: 0,
            radius: 4.0,
            width: 1920.0,
            height: 1080.0,
            sweep: false,
            auto_domain: true,
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let next = |i: &mut usize| -> String {
            *i += 1;
            argv.get(*i).cloned().unwrap_or_default()
        };
        match argv[i].as_str() {
            "--particles" | "-n" => a.particles = next(&mut i).parse().unwrap_or(a.particles),
            "--steps" | "-s" => a.steps = next(&mut i).parse().unwrap_or(a.steps),
            "--threads" | "-t" => a.threads = next(&mut i).parse().unwrap_or(0),
            "--radius" | "-r" => a.radius = next(&mut i).parse().unwrap_or(a.radius),
            "--width" => a.width = next(&mut i).parse().unwrap_or(a.width),
            "--height" => a.height = next(&mut i).parse().unwrap_or(a.height),
            "--gravity" | "-g" => a.gravity = true,
            "--no-parallel" => a.parallel = false,
            "--sweep" => a.sweep = true,
            "--fixed-domain" => a.auto_domain = false,
            other => eprintln!("warning: ignoring unknown argument {other:?}"),
        }
        i += 1;
    }
    a
}

struct Result_ {
    ms_per_step: f64,
    steps_per_sec: f64,
    finite: bool,
    oob: usize,
}

/// Scale the domain with the particle count so every size runs at the same
/// packing fraction. With a fixed window, 100k particles of radius 4 would need
/// more than the available area, and the benchmark would be measuring a jammed
/// solver rather than the simulation.
fn domain_for(n: usize, radius: f32, packing: f32, aspect: f32) -> (f32, f32) {
    let area = (n as f32 * std::f32::consts::PI * radius * radius / packing).max(1.0);
    let h = (area / aspect).sqrt();
    (h * aspect, h)
}

fn run(a: &Args, particles: usize, gravity: bool, parallel: bool) -> Result_ {
    let (width, height) = if a.auto_domain {
        domain_for(particles, a.radius, 0.18, 16.0 / 9.0)
    } else {
        (a.width, a.height)
    };
    let cfg = Config {
        width,
        height,
        radius: a.radius,
        gravity,
        parallel,
        ..Config::default()
    };
    let mut sim = Sim::new(cfg);
    sim.spawn_random(particles, 0xC0FFEE);

    let dt = 1.0 / 120.0;
    // Warm up: first steps pay for cold caches and the initial tree build.
    for _ in 0..10 {
        sim.step(dt);
    }

    let t0 = Instant::now();
    for _ in 0..a.steps {
        sim.step(dt);
    }
    let elapsed = t0.elapsed().as_secs_f64();

    Result_ {
        ms_per_step: elapsed / a.steps as f64 * 1e3,
        steps_per_sec: a.steps as f64 / elapsed,
        finite: !sim.has_non_finite(),
        oob: sim.out_of_bounds_count(),
    }
}

fn main() {
    let a = parse_args();
    if a.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(a.threads)
            .build_global()
            .ok();
    }
    let threads = rayon::current_num_threads();

    if a.sweep {
        println!(
            "sweep: radius {}, {} steps, {} rayon threads, domain auto-scaled to 18% packing",
            a.radius, a.steps, threads
        );
        println!();
        println!(
            "{:>9}  {:>9}  {:>12}  {:>12}  {:>12}  {:>12}",
            "particles", "mode", "serial ms", "parallel ms", "speedup", "steps/s"
        );
        println!("{}", "-".repeat(78));
        for &n in &[1_000usize, 5_000, 20_000, 50_000, 100_000] {
            for &grav in &[false, true] {
                let serial = run(&a, n, grav, false);
                let par = run(&a, n, grav, true);
                println!(
                    "{:>9}  {:>9}  {:>12.3}  {:>12.3}  {:>11.2}x  {:>12.1}",
                    n,
                    if grav { "gravity" } else { "collide" },
                    serial.ms_per_step,
                    par.ms_per_step,
                    serial.ms_per_step / par.ms_per_step,
                    par.steps_per_sec
                );
            }
        }
        return;
    }

    let r = run(&a, a.particles, a.gravity, a.parallel);
    let (w, h) = if a.auto_domain {
        domain_for(a.particles, a.radius, 0.18, 16.0 / 9.0)
    } else {
        (a.width, a.height)
    };
    println!("domain {:.0}x{:.0}, radius {}", w, h, a.radius);
    println!(
        "particles={} gravity={} parallel={} threads={}",
        a.particles,
        a.gravity,
        a.parallel,
        if a.parallel { threads } else { 1 }
    );
    println!(
        "  {:.3} ms/step   {:.1} steps/s",
        r.ms_per_step, r.steps_per_sec
    );
    println!("  all finite: {}   out of bounds: {}", r.finite, r.oob);
}
