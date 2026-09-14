//! Headless GPU benchmark and validation.
//!
//! Runs the same configuration through the CPU and GPU backends from identical
//! initial conditions, reports throughput for both, and checks the GPU result
//! against the invariants the CPU path is tested for.
//!
//! The two backends are not expected to agree bit-for-bit: the CPU resolves
//! collision pairs in sequence, while the GPU gathers, so each particle sees its
//! neighbours' pre-collision state. Both are valid; they diverge like any two
//! integrators of a chaotic system.

use std::time::Instant;

use simulation::gpu::{GpuContext, GpuSim};
use simulation::{Config, Sim};

fn domain_for(n: usize, radius: f32, packing: f32, aspect: f32) -> (f32, f32) {
    let area = (n as f32 * std::f32::consts::PI * radius * radius / packing).max(1.0);
    let h = (area / aspect).sqrt();
    (h * aspect, h)
}

struct Args {
    particles: usize,
    steps: usize,
    gravity: bool,
    radius: f32,
    sweep: bool,
    skip_cpu: bool,
}

fn parse() -> Args {
    let mut a = Args {
        particles: 100_000,
        steps: 200,
        gravity: false,
        radius: 4.0,
        sweep: false,
        skip_cpu: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let next = |i: &mut usize| -> String {
            *i += 1;
            argv.get(*i).cloned().unwrap_or_default()
        };
        match argv[i].as_str() {
            "-n" | "--particles" => a.particles = next(&mut i).parse().unwrap_or(a.particles),
            "-s" | "--steps" => a.steps = next(&mut i).parse().unwrap_or(a.steps),
            "-r" | "--radius" => a.radius = next(&mut i).parse().unwrap_or(a.radius),
            "-g" | "--gravity" => a.gravity = true,
            "--sweep" => a.sweep = true,
            "--skip-cpu" => a.skip_cpu = true,
            other => eprintln!("warning: ignoring {other:?}"),
        }
        i += 1;
    }
    a
}

struct Outcome {
    gpu_ms: f64,
    cpu_ms: Option<f64>,
    finite: bool,
    oob: usize,
    max_overlap: Option<f32>,
    gpu_ke: f64,
    cpu_ke: f64,
    over_clamp: usize,
}

fn run(ctx: &GpuContext, a: &Args, n: usize, gravity: bool, check_overlap: bool) -> Outcome {
    let (width, height) = domain_for(n, a.radius, 0.18, 16.0 / 9.0);
    let cfg = Config {
        width,
        height,
        radius: a.radius,
        gravity,
        parallel: true,
        ..Config::default()
    };

    // One spawn, shared as the starting state for both backends.
    let mut cpu = Sim::new(cfg);
    cpu.spawn_random(n, 0xC0FFEE);

    let gpu = GpuSim::new(
        &ctx.device,
        cfg,
        &cpu.px,
        &cpu.py,
        &cpu.vx,
        &cpu.vy,
        &cpu.color,
    );

    let dt = 1.0 / 120.0;

    // Warm up: shader compilation and first-touch allocation land here.
    gpu.run_steps(&ctx.device, &ctx.queue, dt, 10);

    let t0 = Instant::now();
    gpu.run_steps(&ctx.device, &ctx.queue, dt, a.steps);
    let gpu_ms = t0.elapsed().as_secs_f64() / a.steps as f64 * 1e3;

    let positions = gpu.read_positions(&ctx.device, &ctx.queue);
    let velocities = gpu.read_velocities(&ctx.device, &ctx.queue);

    let r = cfg.radius;
    let (lo_x, hi_x) = (r - 0.01, cfg.width - r + 0.01);
    let (lo_y, hi_y) = (r - 0.01, cfg.height - r + 0.01);
    let finite = positions
        .iter()
        .all(|p| p[0].is_finite() && p[1].is_finite());
    let oob = positions
        .iter()
        .filter(|p| p[0] < lo_x || p[0] > hi_x || p[1] < lo_y || p[1] > hi_y)
        .count();

    // Collisions are elastic, so total kinetic energy should be conserved up to
    // wall interactions and the speed clamp. A large drift would mean the
    // collision response is losing or injecting energy.
    let gpu_ke: f64 = velocities
        .iter()
        .map(|v| 0.5 * cfg.mass as f64 * (v[0] as f64 * v[0] as f64 + v[1] as f64 * v[1] as f64))
        .sum();
    let over_clamp = velocities
        .iter()
        .filter(|v| (v[0] * v[0] + v[1] * v[1]).sqrt() > cfg.max_speed * 1.01)
        .count();

    let max_overlap = if check_overlap {
        let min_d = r * 2.0;
        let mut worst = 0.0f32;
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let dx = positions[j][0] - positions[i][0];
                let dy = positions[j][1] - positions[i][1];
                let d = (dx * dx + dy * dy).sqrt();
                if d < min_d {
                    worst = worst.max(min_d - d);
                }
            }
        }
        Some(worst)
    } else {
        None
    };

    let cpu_ms = if a.skip_cpu {
        None
    } else {
        for _ in 0..5 {
            cpu.step(dt);
        }
        let t0 = Instant::now();
        for _ in 0..a.steps {
            cpu.step(dt);
        }
        Some(t0.elapsed().as_secs_f64() / a.steps as f64 * 1e3)
    };

    Outcome {
        gpu_ms,
        cpu_ms,
        finite,
        oob,
        max_overlap,
        gpu_ke,
        cpu_ke: cpu.kinetic_energy(),
        over_clamp,
    }
}

fn main() {
    let a = parse();
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            std::process::exit(1);
        }
    };
    println!("adapter: {}", ctx.describe());
    println!("cpu threads: {}", rayon::current_num_threads());
    println!();

    if a.sweep {
        println!(
            "{:>9}  {:>8}  {:>11}  {:>11}  {:>9}  {:>10}",
            "particles", "mode", "cpu ms", "gpu ms", "speedup", "gpu steps/s"
        );
        println!("{}", "-".repeat(70));
        for &n in &[10_000usize, 50_000, 200_000, 1_000_000] {
            for &g in &[false, true] {
                // Gravity is an exact O(n^2) sum on the GPU; past a few hundred
                // thousand particles that is no longer a sensible thing to time.
                if g && n > 200_000 {
                    continue;
                }
                let o = run(&ctx, &a, n, g, false);
                let cpu = o.cpu_ms.unwrap_or(f64::NAN);
                println!(
                    "{:>9}  {:>8}  {:>11.3}  {:>11.3}  {:>8.1}x  {:>10.1}{}",
                    n,
                    if g { "gravity" } else { "collide" },
                    cpu,
                    o.gpu_ms,
                    cpu / o.gpu_ms,
                    1000.0 / o.gpu_ms,
                    if o.finite && o.oob == 0 {
                        ""
                    } else {
                        "  INVALID"
                    }
                );
            }
        }
        return;
    }

    let o = run(&ctx, &a, a.particles, a.gravity, a.particles <= 5_000);
    println!(
        "particles={} gravity={} steps={}",
        a.particles, a.gravity, a.steps
    );
    if let Some(cpu) = o.cpu_ms {
        println!("  cpu: {:>9.3} ms/step", cpu);
        println!(
            "  gpu: {:>9.3} ms/step   ({:.1}x, {:.0} steps/s)",
            o.gpu_ms,
            cpu / o.gpu_ms,
            1000.0 / o.gpu_ms
        );
    } else {
        println!("  gpu: {:>9.3} ms/step", o.gpu_ms);
    }
    println!(
        "  validation: finite={} out_of_bounds={} over_speed_clamp={}",
        o.finite, o.oob, o.over_clamp
    );
    println!(
        "  kinetic energy: gpu {:.4e}  cpu {:.4e}  ({:+.1}%)",
        o.gpu_ke,
        o.cpu_ke,
        (o.gpu_ke - o.cpu_ke) / o.cpu_ke * 100.0
    );
    if let Some(ov) = o.max_overlap {
        println!(
            "  max overlap: {:.4} px (diameter {:.1})",
            ov,
            a.radius * 2.0
        );
    }
}
