extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;

use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;
use rand::Rng;
use std::collections::HashMap;
use std::io;

// Ball object with physics properties
#[derive(Copy, Clone, Debug)]
struct Ball {
    acceleration: Vector2D,
    velocity: Vector2D,
    location: Vector2D,
    radius: f64,
    color: [f32; 4],
    mass: f64, // Mass in KG
}

// 2D vector representation
#[derive(Copy, Clone, Debug)]
struct Vector2D {
    x: f64,
    y: f64,
}

impl Vector2D {
    fn dot(&self, other: &Vector2D) -> f64 {
        return self.x * other.x + self.y * other.y;
    }

    fn norm(&self) -> f64 {
        return (self.dot(self)).sqrt();
    }

    fn normalize(&self) -> Vector2D{
        Vector2D {
            x: self.x / self.norm(),
            y: self.y / self.norm(),
        }
    }

    fn subtract(&self, other: &Vector2D) -> Vector2D {
        Vector2D {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    fn add(&self, other: &Vector2D) -> Vector2D {
        Vector2D {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    fn scale(&self, scale: f64) -> Vector2D {
        Vector2D {
            x: scale * self.x,
            y: scale * self.y,
        }
    }
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

    fn update(&mut self, args: &UpdateArgs) {
        const G: f64 = 6.67408e-11; // Gravitation Constant

        let mut acc_updates: HashMap<u32, Vector2D> = HashMap::new();
        // Calculate Forces on each ball
        for i in 0..self.balls.len() {
            let mut acc = Vector2D { x: 0.0, y: 0.0 };
            let current_ball = &self.balls[i];
            for j in 0..self.balls.len() {
                if i == j {
                    continue;
                }
                // Gravitation force
                let mut gravitation_dir = self.balls[j].location.subtract(&current_ball.location);
                let magnitude =
                    G * self.balls[j].mass / (gravitation_dir.norm() * gravitation_dir.norm());
                gravitation_dir = gravitation_dir.normalize();
                let gravitation_force = gravitation_dir.scale(magnitude * 0.0);
                acc = acc.add(&gravitation_force);
            }
            // Adding constant acceleration
            acc = acc.add(&Vector2D { x: 0.0, y: 0.0 });
            acc_updates.insert(i as u32, acc);
        }

        // Check for collisions with other particles and update each ball
        self.balls = self.check_for_collisions_and_update_velocity();

        for (i, ball) in self.balls.iter_mut().enumerate() {
            // Update Ball Acceleration
            ball.acceleration = *acc_updates.get(&(i as u32)).expect("Did not want this");

            // Update Ball velocity
            ball.velocity = Vector2D {
                x: ball.velocity.x + ball.acceleration.x * args.dt * self.simulation_factor as f64,
                y: ball.velocity.y + ball.acceleration.y * args.dt * self.simulation_factor as f64,
            };

            // Update ball location
            ball.location = Vector2D {
                x: ball.location.x + ball.velocity.x * self.simulation_factor as f64 * args.dt,
                y: ball.location.y + ball.velocity.y * self.simulation_factor as f64 * args.dt,
            };

            // Check for collisions with window boundaries
            if ball.location.y + ball.radius > self.resolution.1 {
                ball.velocity = Vector2D {
                    x: ball.velocity.x,
                    y: -1.0 * ball.velocity.y,
                };
                ball.location.y = self.resolution.1 - ball.radius;
            }
            if ball.location.y - ball.radius < 0.0 {
                ball.velocity = Vector2D {
                    x: ball.velocity.x,
                    y: -1.0 * ball.velocity.y,
                };
                ball.location.y = ball.radius;
            }
            if ball.location.x + ball.radius > self.resolution.0 {
                ball.velocity = Vector2D {
                    x: -1.0 * ball.velocity.x,
                    y: ball.velocity.y,
                };
                ball.location.x = self.resolution.0 - ball.radius;
            }
            if ball.location.x - ball.radius < 0.0 {
                ball.velocity = Vector2D {
                    x: -1.0 * ball.velocity.x,
                    y: ball.velocity.y,
                };
                ball.location.x = ball.radius;
            }
        }
    }

    fn check_for_collisions_and_update_velocity(&mut self) -> Vec<Ball> {
        let mut sorted_balls = self.balls.to_vec();
        sorted_balls.sort_by(|a, b| a.location.x.partial_cmp(&b.location.x).unwrap());
        let mut collision_updates: Vec<Vector2D> = Vec::new();
        let mut location_updates: Vec<Vector2D> = Vec::new();

        for i in 0..sorted_balls.len() {
            collision_updates.push(sorted_balls[i].velocity);
            location_updates.push(sorted_balls[i].location);
        }

        for i in 0..sorted_balls.len() {
            for j in (i + 1)..sorted_balls.len() {
                let ball1 = &sorted_balls[i];
                let ball2 = &sorted_balls[j];
                let is_collision_x = (location_updates[i].x - location_updates[j].x).abs()
                    <= (ball1.radius + ball2.radius);
                let is_collision_y = (location_updates[i].y - location_updates[j].y).abs()
                    <= (ball1.radius + ball2.radius);

                let is_collision = is_collision_x && is_collision_y;
                if !is_collision {
                    break;
                }
                // Resolve weird collision
                let loc_update = location_updates[j].subtract(&location_updates[i]).normalize().scale(ball1.radius + ball2.radius);
                location_updates[j] = location_updates[j].add(&loc_update);

                // Update the particle velocities
                let v1_minus_v2 = collision_updates[i].subtract(&collision_updates[j]);
                let x1_minus_x2 = location_updates[i].subtract(&location_updates[j]);
                let distance = x1_minus_x2.norm();
                
                let mass_term_1 = (2.0 * ball2.mass) / (ball1.mass + ball2.mass);
                let dot_product_term_1 = v1_minus_v2.dot(&x1_minus_x2) / (distance * distance);
                let velocity_ball1 = collision_updates[i].subtract(&x1_minus_x2.scale(dot_product_term_1 * mass_term_1));

                let mass_term_2 = (2.0 * ball1.mass) / (ball1.mass + ball2.mass);
                let v2_minus_v1 = v1_minus_v2.scale(-1.0);
                let x2_minus_x1 = x1_minus_x2.scale(-1.0);
                let dot_product_term_2 = v2_minus_v1.dot(&x2_minus_x1) / (distance * distance);
                let velocity_ball2 = collision_updates[j].subtract(&x2_minus_x1.scale(dot_product_term_2 * mass_term_2));
                
                collision_updates[i] = velocity_ball1;
                collision_updates[j] = velocity_ball2;
            }
        }

        for i in 0..sorted_balls.len() {
            sorted_balls[i].velocity = collision_updates[i];
            sorted_balls[i].location = location_updates[i];
        }

        return sorted_balls;
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
    let width = 1200.0;
    let height = 600.0;

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
            acceleration: Vector2D { x: 0.0, y: 0.0 },
            velocity: Vector2D {
                x: rng.gen_range(-100.0..100.0),
                y: rng.gen_range(-100.0..100.0),
            },
            location: Vector2D {
                x: rng.gen_range(0.0..(width)),
                y: rng.gen_range(0.0..(height)),
            },
            radius,
            color: [
                rng.gen_range(0.2..1.0),
                rng.gen_range(0.2..1.0),
                rng.gen_range(0.2..1.0),
                rng.gen_range(0.5..1.0),
            ],
            mass: 1.0,
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

        if let Some(args) = e.update_args() {
            simulation.update(&args);
        }
    }
}

// TODO:
// 1. Play with forces: force on ball attracting towards random ball, gravitational attraction, electric charge simulation etc.
// 2. Implement particle collisions
