extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;

use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, RenderEvent, UpdateEvent};
use piston::window::WindowSettings;
use rand::Rng;
use std::io;

// Ball object with physics properties
struct Ball {
    velocity: Vector2D,
    location: Vector2D,
    radius: f64,
    color: [f32; 4],
}

// 2D vector representation
struct Vector2D {
    x: f64,
    y: f64,
}

pub struct Simulation {
    gl: GlGraphics,   // OpenGL drawing backend.
    balls: Vec<Ball>, // Collection of objects in scene
    resolution: (f64, f64),
    simulation_factor: u32,
}

impl Simulation {
    fn render(&mut self, args: &RenderArgs) {
        use graphics::*;
        const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

        let balls = &self.balls;
        self.gl.draw(args.viewport(), |c, gl| {
            // Clear the screen.
            clear(BLACK, gl);
            for ball in balls.iter() {
                let transform1 = c.transform.trans(ball.location.x, ball.location.y);

                let circle = rectangle::centered_square(0.0, 0.0, ball.radius);
                ellipse(ball.color, circle, transform1, gl);
            }
        });
    }

    fn update(&mut self) {
        for ball in self.balls.iter_mut() {
            // Update ball location
            ball.location = Vector2D {
                x: ball.location.x + ball.velocity.x * self.simulation_factor as f64,
                y: ball.location.y + ball.velocity.y * self.simulation_factor as f64,
            };

            // Check for collisions with window boundaries
            if ball.location.y + ball.radius >= self.resolution.1 {
                ball.velocity = Vector2D {
                    x: ball.velocity.x,
                    y: -1.0 * ball.velocity.y,
                };
            } else if ball.location.y - ball.radius < 0.0 {
                ball.velocity = Vector2D {
                    x: ball.velocity.x,
                    y: -1.0 * ball.velocity.y,
                };
            } else if ball.location.x + ball.radius >= self.resolution.0 {
                ball.velocity = Vector2D {
                    x: -1.0 * ball.velocity.x,
                    y: ball.velocity.y,
                };
            } else if ball.location.x - ball.radius < 0.0 {
                ball.velocity = Vector2D {
                    x: -1.0 * ball.velocity.x,
                    y: ball.velocity.y,
                };
            }
        }
    }
}

fn main() {
    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

    let mut num_balls = String::new();

    println!("Enter number of balls in simulation: ");

    io::stdin()
        .read_line(&mut num_balls)
        .ok()
        .expect("Couldn't read line");

    let num_balls: u32 = num_balls.trim().parse().expect("Wanted a number");

    // Window resolution
    let width = 1920.0;
    let height = 1080.0;

    // Create an Glutin window.
    let mut window: Window = WindowSettings::new("simulation", [width, height])
        .graphics_api(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();

    // Create objects in simulation
    let mut balls = Vec::new();
    let radius = 10.0;
    let mut rng = rand::thread_rng();

    for _ in 0..num_balls {
        balls.push(Ball {
            velocity: Vector2D {
                x: rng.gen_range(-1.0..1.0),
                y: rng.gen_range(-1.0..1.0),
            },
            location: Vector2D {
                x: rng.gen_range(radius..50.0 - radius),
                y: rng.gen_range(radius..50.0 - radius),
            },
            radius,
            color: [rng.gen_range(0.2..1.0), rng.gen_range(0.2..1.0), rng.gen_range(0.2..1.0), rng.gen_range(0.5..1.0)],
        });
    }

    // Create a new simulation and run it.
    let mut simulation = Simulation {
        gl: GlGraphics::new(opengl),
        balls: balls,
        resolution: (width, height),
        simulation_factor: 1,
    };

    let mut events = Events::new(EventSettings::new());
    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            simulation.render(&args);
        }

        if let Some(_) = e.update_args() {
            simulation.update();
        }
    }
}
