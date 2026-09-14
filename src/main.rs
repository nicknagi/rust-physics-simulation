//! Windowed front-end for the particle simulation.
//!
//! Physics runs on a fixed timestep with an accumulator, decoupled from the
//! render rate. The original drove physics straight off the frame delta, so the
//! simulation slowed down as the particle count rose; here a heavy frame costs
//! frames, not simulated time.

use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{Button, Key, PressEvent, RenderEvent, UpdateEvent};
use piston::window::WindowSettings;

use simulation::render::ParticleRenderer;
use simulation::{Config, Sim};

/// Physics timestep. Fixed, so behaviour does not depend on frame rate.
const FIXED_DT: f32 = 1.0 / 240.0;
/// Ceiling on catch-up steps, so a stall cannot spiral.
const MAX_SUBSTEPS: u32 = 8;

struct Options {
    particles: usize,
    cfg: Config,
    seed: u64,
    vsync: bool,
    max_fps: u64,
    naive_render: bool,
}

fn print_help() {
    println!(
        "\
particle simulation

USAGE:
    simulation [OPTIONS]

OPTIONS:
    -n, --particles <N>   number of particles        [default: 10000]
    -r, --radius <R>      particle radius in pixels  [default: 3]
    -g, --gravity         enable gravitational attraction
        --width <W>       window width               [default: 1600]
        --height <H>      window height              [default: 900]
        --theta <T>       Barnes-Hut opening angle   [default: 0.7]
        --seed <S>        RNG seed                   [default: 12345]
        --no-parallel     run the physics on one thread
        --no-vsync        uncap the frame rate
        --max-fps <F>     frame rate ceiling         [default: 10000]
        --naive-render    draw one ellipse per particle (the original path)
    -i, --interactive     prompt for settings on stdin (original behaviour)
    -h, --help            show this message

CONTROLS:
    G      toggle gravity        Space  pause
    R      respawn particles     Esc    quit"
    );
}

fn parse_options() -> Option<Options> {
    let mut cfg = Config {
        width: 1600.0,
        height: 900.0,
        radius: 3.0,
        ..Config::default()
    };
    let mut opts = Options {
        particles: 10_000,
        cfg,
        seed: 12345,
        vsync: true,
        max_fps: 10_000,
        naive_render: false,
    };

    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        return Some(opts);
    }
    if argv.iter().any(|a| a == "-i" || a == "--interactive") {
        return interactive(opts);
    }

    let mut i = 0;
    while i < argv.len() {
        let next = |i: &mut usize| -> String {
            *i += 1;
            argv.get(*i).cloned().unwrap_or_default()
        };
        match argv[i].as_str() {
            "-n" | "--particles" => opts.particles = next(&mut i).parse().unwrap_or(opts.particles),
            "-r" | "--radius" => cfg.radius = next(&mut i).parse().unwrap_or(cfg.radius),
            "--width" => cfg.width = next(&mut i).parse().unwrap_or(cfg.width),
            "--height" => cfg.height = next(&mut i).parse().unwrap_or(cfg.height),
            "--theta" => cfg.theta = next(&mut i).parse().unwrap_or(cfg.theta),
            "--seed" => opts.seed = next(&mut i).parse().unwrap_or(opts.seed),
            "-g" | "--gravity" => cfg.gravity = true,
            "--no-parallel" => cfg.parallel = false,
            "--no-vsync" => opts.vsync = false,
            "--max-fps" => opts.max_fps = next(&mut i).parse().unwrap_or(opts.max_fps),
            "--naive-render" => opts.naive_render = true,
            "-h" | "--help" => {
                print_help();
                return None;
            }
            other => eprintln!("warning: ignoring unknown argument {other:?}"),
        }
        i += 1;
    }
    opts.cfg = cfg;
    Some(opts)
}

/// The original stdin prompts, kept for anyone used to them.
fn interactive(mut opts: Options) -> Option<Options> {
    use std::io::{self, Write};

    let ask = |prompt: &str| -> String {
        print!("{prompt}");
        io::stdout().flush().ok();
        let mut s = String::new();
        io::stdin().read_line(&mut s).ok();
        s.trim().to_string()
    };

    match ask("Enter number of particles: ").parse::<usize>() {
        Ok(n) => opts.particles = n,
        Err(_) => eprintln!("not a number; keeping {}", opts.particles),
    }
    let g = ask("Gravitational attraction? (0 = no, anything else = yes): ");
    opts.cfg.gravity = !(g.is_empty() || g == "0");
    Some(opts)
}

fn main() {
    let Some(opts) = parse_options() else {
        return;
    };
    let cfg = opts.cfg;

    if !cfg.parallel {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .ok();
    }

    let mut sim = Sim::new(cfg);
    sim.spawn_random(opts.particles, opts.seed);

    println!(
        "{} particles, radius {}, gravity {}, {} threads",
        sim.len(),
        cfg.radius,
        if cfg.gravity { "on" } else { "off" },
        if cfg.parallel {
            rayon::current_num_threads()
        } else {
            1
        }
    );
    println!("G toggles gravity, Space pauses, R respawns, Esc quits.");

    let opengl = OpenGL::V3_2;
    let mut window: Window =
        WindowSettings::new("particle simulation", [cfg.width as f64, cfg.height as f64])
            .graphics_api(opengl)
            .vsync(opts.vsync)
            .exit_on_esc(true)
            .build()
            .expect("failed to create window");

    let mut gl = GlGraphics::new(opengl);
    let mut renderer = ParticleRenderer::for_radius(cfg.radius);
    // Piston's event loop interleaves updates and renders against its own ups /
    // max_fps schedule, and its defaults pin the frame rate near 30 regardless
    // of how little work there is. Physics runs on its own fixed-step
    // accumulator below, so ups only needs to be high enough to sample time --
    // the frame rate is what we actually want off the leash.
    let mut event_settings = EventSettings::new();
    event_settings.ups = 240;
    event_settings.max_fps = opts.max_fps;
    let mut events = Events::new(event_settings);

    let mut accumulator = 0.0f32;
    let mut paused = false;
    let mut frames = 0u32;
    let mut physics_ms = 0.0f64;
    let mut render_ms = 0.0f64;
    let mut last_report = std::time::Instant::now();

    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            let px = &sim.px;
            let py = &sim.py;
            let colors = &sim.color;
            let radius = sim.cfg.radius;
            let naive = opts.naive_render;
            let t_draw = std::time::Instant::now();
            gl.draw(args.viewport(), |c, g| {
                graphics::clear([0.02, 0.02, 0.04, 1.0], g);
                if naive {
                    simulation::render::draw_per_particle(
                        c.transform,
                        &c.draw_state,
                        px,
                        py,
                        colors,
                        radius,
                        g,
                    );
                } else {
                    renderer.draw(c.transform, &c.draw_state, px, py, colors, radius, g);
                }
            });
            render_ms += t_draw.elapsed().as_secs_f64() * 1e3;
            frames += 1;
        }

        if let Some(args) = e.update_args() {
            if !paused {
                accumulator += args.dt as f32;
                let mut sub = 0;
                let t0 = std::time::Instant::now();
                while accumulator >= FIXED_DT && sub < MAX_SUBSTEPS {
                    sim.step(FIXED_DT);
                    accumulator -= FIXED_DT;
                    sub += 1;
                }
                if sub == MAX_SUBSTEPS {
                    // Too far behind to catch up; drop the backlog rather than
                    // spiral into ever-longer frames.
                    accumulator = 0.0;
                }
                physics_ms += t0.elapsed().as_secs_f64() * 1e3;
            }

            if last_report.elapsed().as_secs_f32() >= 1.0 {
                let secs = last_report.elapsed().as_secs_f64();
                println!(
                    "{:.0} fps   physics {:.2} ms/frame   render {:.2} ms/frame   {} particles",
                    frames as f64 / secs,
                    physics_ms / frames.max(1) as f64,
                    render_ms / frames.max(1) as f64,
                    sim.len()
                );
                frames = 0;
                physics_ms = 0.0;
                render_ms = 0.0;
                last_report = std::time::Instant::now();
            }
        }

        if let Some(Button::Keyboard(key)) = e.press_args() {
            match key {
                Key::G => {
                    sim.cfg.gravity = !sim.cfg.gravity;
                    println!("gravity {}", if sim.cfg.gravity { "on" } else { "off" });
                }
                Key::Space => {
                    paused = !paused;
                    println!("{}", if paused { "paused" } else { "running" });
                }
                Key::R => {
                    sim.spawn_random(opts.particles, rand::random::<u64>());
                    println!("respawned");
                }
                _ => {}
            }
        }
    }
}
