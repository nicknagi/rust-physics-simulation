//! GPU-backed windowed simulation (wgpu -> Metal on Apple Silicon).
//!
//! Particle state is uploaded once and then stays in GPU memory: compute passes
//! advance it and the render pipeline instances directly off the same position
//! buffer, so no particle data crosses the bus per frame.

use std::sync::Arc;
use std::time::Instant;

use simulation::gpu::{GpuRenderer, GpuSim};
use simulation::{Config, DiscSpawn, Sim};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const FIXED_DT: f32 = 1.0 / 240.0;
const MAX_SUBSTEPS: u32 = 8;

struct Options {
    particles: usize,
    radius: f32,
    width: f32,
    height: f32,
    gravity: bool,
    gravity_scale: f32,
    speed: f32,
    whirl: bool,
    dot_size: f32,
    central_gm: f32,
    seed: u64,
    vsync: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            particles: 50_000,
            radius: 1.5,
            width: 1600.0,
            height: 900.0,
            gravity: false,
            gravity_scale: 1.0,
            speed: 1.0,
            whirl: false,
            dot_size: 0.0,
            central_gm: 2.0e7,
            seed: 12345,
            vsync: true,
        }
    }
}

fn parse() -> Option<Options> {
    let mut o = Options::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let next = |i: &mut usize| -> String {
            *i += 1;
            argv.get(*i).cloned().unwrap_or_default()
        };
        match argv[i].as_str() {
            "-n" | "--particles" => o.particles = next(&mut i).parse().unwrap_or(o.particles),
            "-r" | "--radius" => o.radius = next(&mut i).parse().unwrap_or(o.radius),
            "--width" => o.width = next(&mut i).parse().unwrap_or(o.width),
            "--height" => o.height = next(&mut i).parse().unwrap_or(o.height),
            "--seed" => o.seed = next(&mut i).parse().unwrap_or(o.seed),
            "-g" | "--gravity" => o.gravity = true,
            "--gravity-scale" => {
                o.gravity_scale = next(&mut i).parse().unwrap_or(o.gravity_scale);
                o.gravity = true;
            }
            "--speed" => o.speed = next(&mut i).parse().unwrap_or(o.speed),
            "--whirl" => o.whirl = true,
            "--dot-size" => o.dot_size = next(&mut i).parse().unwrap_or(o.dot_size),
            "--central-gm" => o.central_gm = next(&mut i).parse().unwrap_or(o.central_gm),
            "--no-vsync" => o.vsync = false,
            "-h" | "--help" => {
                println!(
                    "\
GPU particle simulation

USAGE:
    gpu [OPTIONS]

OPTIONS:
    -n, --particles <N>   particle count            [default: 50000]
    -r, --radius <R>      particle radius           [default: 1.5]
        --width <W>       window width              [default: 1600]
        --height <H>      window height             [default: 900]
    -g, --gravity         enable gravity (exact O(n^2) on GPU; see README)
        --gravity-scale <S>  scale gravity strength, 1 = default (implies -g)
        --speed <S>       scale initial velocities, 1 = default. Weak gravity is
                          only visible if the particles start slow enough for it
                          to dominate.
        --seed <S>        RNG seed
        --whirl           orbiting ring around a central mass (see README)
        --dot-size <D>    drawn particle size, independent of collision radius
        --central-gm <G>  strength of the central attractor  [default: 2e7]
        --no-vsync        uncap the frame rate
    -h, --help            show this message

CONTROLS:
    G  toggle gravity     Space  pause     Esc  quit"
                );
                return None;
            }
            other => eprintln!("warning: ignoring {other:?}"),
        }
        i += 1;
    }
    Some(o)
}

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    sim: GpuSim,
    renderer: GpuRenderer,
}

struct App {
    opts: Options,
    gfx: Option<Gfx>,
    last_frame: Instant,
    accumulator: f32,
    paused: bool,
    frames: u32,
    last_report: Instant,
}

impl App {
    fn new(opts: Options) -> Self {
        App {
            opts,
            gfx: None,
            last_frame: Instant::now(),
            accumulator: 0.0,
            paused: false,
            frames: 0,
            last_report: Instant::now(),
        }
    }

    fn init(&mut self, event_loop: &ActiveEventLoop) {
        let o = &self.opts;
        // The simulation domain can be far larger than any window: scale the
        // window down to fit, preserving aspect. Retina doubles the physical
        // surface, and Metal caps textures at 16384, so a 1:1 mapping breaks
        // well before the GPU runs out of particles.
        const MAX_WINDOW: f32 = 1600.0;
        let scale = (MAX_WINDOW / o.width)
            .min(MAX_WINDOW * 9.0 / 16.0 / o.height)
            .min(1.0);
        let (win_w, win_h) = (o.width * scale, o.height * scale);
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("gpu particle simulation")
                        .with_inner_size(winit::dpi::LogicalSize::new(win_w, win_h)),
                )
                .expect("failed to create window"),
        );

        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("no suitable GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpu sim device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("failed to create device");

        println!("adapter: {}", adapter.get_info().name);

        let size = window.inner_size();
        // Let wgpu fill in the format/colour-space/alpha defaults for this
        // surface, then override only what we care about.
        let max_dim = adapter.limits().max_texture_dimension_2d;
        let mut config = surface
            .get_default_config(
                &adapter,
                size.width.clamp(1, max_dim),
                size.height.clamp(1, max_dim),
            )
            .expect("surface is not supported by this adapter");
        config.present_mode = if self.opts.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let format = config.format;
        surface.configure(&device, &config);

        // The simulation domain is the window in logical pixels; the render pass
        // maps world space onto whatever the surface currently is.
        // The whirl preset needs a very different regime from the default
        // scatter: a dominant central mass for Keplerian orbits, a small
        // collision radius (collisions in a sheared disc spread it viscously),
        // and weak mutual gravity (a cold self-gravitating ring clumps apart).
        // See the README for the measurements behind these numbers.
        let (radius, gravity_scale, central_gm) = if self.opts.whirl {
            (
                if self.opts.radius == Options::default().radius {
                    0.4
                } else {
                    self.opts.radius
                },
                if self.opts.gravity_scale == 1.0 {
                    1.0e-5
                } else {
                    self.opts.gravity_scale
                },
                self.opts.central_gm,
            )
        } else {
            (self.opts.radius, self.opts.gravity_scale, 0.0)
        };

        let cfg = Config {
            width: self.opts.width,
            height: self.opts.height,
            radius,
            gravity: self.opts.gravity || self.opts.whirl,
            central_gm,
            // Gravity enters the shader only as G * mass, so scaling mass is
            // exactly a strength dial. Collisions are equal-mass and unaffected.
            mass: Config::default().mass * gravity_scale,
            ..Config::default()
        };

        // Spawn on the CPU once, purely to produce the initial state.
        let mut seed_sim = Sim::new(cfg);
        let mut cfg = cfg;
        if self.opts.whirl {
            // Leave margin to the walls: a ring sized right up to the edge
            // turns any outward drift into a wall bounce, which wrecks it.
            let outer = (cfg.width.min(cfg.height) * 0.33).max(40.0);
            let fastest = seed_sim.spawn_orbital_disc(DiscSpawn {
                count: self.opts.particles,
                seed: self.opts.seed,
                inner_radius: outer * 0.35,
                outer_radius: outer,
                clockwise: false,
            });
            // Clamping an orbit destroys the angular momentum holding the ring
            // up, so lift the cap clear of the fastest orbit.
            cfg.max_speed = fastest * 4.0;
            seed_sim.cfg.max_speed = cfg.max_speed;
            println!(
                "whirl: ring {:.0}..{:.0} px, central_gm {:.1e}, fastest orbit {:.0} px/s",
                outer * 0.34,
                outer,
                cfg.central_gm,
                fastest
            );
        } else {
            seed_sim.spawn_random(self.opts.particles, self.opts.seed);
        }
        if self.opts.speed != 1.0 {
            for v in seed_sim.vx.iter_mut().chain(seed_sim.vy.iter_mut()) {
                *v *= self.opts.speed;
            }
        }

        let sim = GpuSim::new(
            &device,
            cfg,
            &seed_sim.px,
            &seed_sim.py,
            &seed_sim.vx,
            &seed_sim.vy,
            &seed_sim.color,
        );
        let renderer = GpuRenderer::new(&device, format, &sim);
        // Drawn size is independent of the collision radius: a whirl wants
        // near-point masses for the physics but visible dots on screen.
        let dot = if self.opts.dot_size > 0.0 {
            self.opts.dot_size
        } else if self.opts.whirl {
            (cfg.radius * 4.0).max(1.6)
        } else {
            cfg.radius
        };
        renderer.set_viewport(&queue, cfg.width, cfg.height, dot);

        // Particles occupy area; past roughly 60% the domain cannot hold them
        // and the run degenerates into a jammed solid that also overflows the
        // per-cell bin capacity, so the numbers stop meaning anything.
        let packing = self.opts.particles as f32 * std::f32::consts::PI * cfg.radius * cfg.radius
            / (cfg.width * cfg.height);
        let _ = &packing;
        println!(
            "{} particles, radius {}, gravity {}, packing {:.0}%",
            sim.len(),
            cfg.radius,
            if cfg.gravity {
                format!("on (scale {gravity_scale:e})")
            } else {
                "off".to_string()
            },
            packing * 100.0
        );
        // GPU gravity is an exact pairwise sum. That is more accurate than the
        // CPU's Barnes-Hut, but it is O(n^2) against the CPU's O(n log n), and
        // no amount of parallelism wins that argument past ~20k particles.
        if cfg.gravity && self.opts.particles > 20_000 {
            eprintln!(
                "warning: GPU gravity is an exact O(n^2) sum and is slower than the CPU's \
                 Barnes-Hut above ~20k particles. For gravity at this scale use the CPU \
                 binary: cargo run --release --bin simulation -- -n {} --gravity",
                self.opts.particles
            );
        }
        if packing > 0.6 {
            eprintln!(
                "warning: {:.0}% packing -- the particles do not fit in {:.0}x{:.0}. \
                 Lower --particles or --radius, or raise --width/--height.",
                packing * 100.0,
                cfg.width,
                cfg.height
            );
        }
        println!("G toggles gravity, Space pauses, Esc quits.");

        self.gfx = Some(Gfx {
            window,
            surface,
            device,
            queue,
            config,
            sim,
            renderer,
        });
        self.last_frame = Instant::now();
        self.last_report = Instant::now();
    }

    fn frame(&mut self) {
        let Some(gfx) = self.gfx.as_mut() else {
            return;
        };

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.25);
        self.last_frame = now;

        if !self.paused {
            self.accumulator += dt;
            let mut sub = 0;
            while self.accumulator >= FIXED_DT && sub < MAX_SUBSTEPS {
                gfx.sim.step(&gfx.device, &gfx.queue, FIXED_DT);
                self.accumulator -= FIXED_DT;
                sub += 1;
            }
            if sub == MAX_SUBSTEPS {
                self.accumulator = 0.0;
            }
        }

        use wgpu::CurrentSurfaceTexture as Cst;
        let frame = match gfx.surface.get_current_texture() {
            Cst::Success(f) | Cst::Suboptimal(f) => f,
            Cst::Outdated | Cst::Lost => {
                gfx.surface.configure(&gfx.device, &gfx.config);
                return;
            }
            // Transient: skip this frame and try again on the next one.
            Cst::Timeout | Cst::Occluded => return,
            other => {
                eprintln!("surface error: {other:?}");
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        gfx.renderer.draw(&mut encoder, &view, gfx.sim.len() as u32);
        gfx.queue.submit(Some(encoder.finish()));
        drop(view);
        gfx.queue.present(frame);

        self.frames += 1;
        if self.last_report.elapsed().as_secs_f32() >= 1.0 {
            println!(
                "{:.0} fps   {} particles   gravity {}",
                self.frames as f64 / self.last_report.elapsed().as_secs_f64(),
                gfx.sim.len(),
                if gfx.sim.cfg.gravity { "on" } else { "off" }
            );
            self.frames = 0;
            self.last_report = Instant::now();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_none() {
            self.init(event_loop);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gfx) = self.gfx.as_mut() {
                    let max_dim = gfx.device.limits().max_texture_dimension_2d;
                    gfx.config.width = size.width.clamp(1, max_dim);
                    gfx.config.height = size.height.clamp(1, max_dim);
                    gfx.surface.configure(&gfx.device, &gfx.config);
                }
            }
            WindowEvent::RedrawRequested => self.frame(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::Space) => {
                        self.paused = !self.paused;
                        println!("{}", if self.paused { "paused" } else { "running" });
                    }
                    Key::Character(ref c) if c.eq_ignore_ascii_case("g") => {
                        if let Some(gfx) = self.gfx.as_mut() {
                            gfx.sim.cfg.gravity = !gfx.sim.cfg.gravity;
                            println!("gravity {}", if gfx.sim.cfg.gravity { "on" } else { "off" });
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gfx) = self.gfx.as_ref() {
            gfx.window.request_redraw();
        }
    }
}

fn main() {
    let Some(opts) = parse() else {
        return;
    };
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(opts);
    event_loop.run_app(&mut app).expect("event loop failed");
}
