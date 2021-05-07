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
    prev_location: Vector2D, // Location info for t-1 for collision calculations
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
    simulation_factor: f64,
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

        for (i, ball) in self.balls.iter_mut().enumerate() {
            // Update Ball Acceleration
            ball.acceleration = *acc_updates.get(&(i as u32)).expect("Did not want this");

            // Update Ball velocity
            ball.velocity = Vector2D {
                x: ball.velocity.x + ball.acceleration.x * args.dt * self.simulation_factor as f64,
                y: ball.velocity.y + ball.acceleration.y * args.dt * self.simulation_factor as f64,
            };

            ball.prev_location = ball.location;

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

        // Check for collisions with other particles and update each ball
        self.balls = self.check_for_collisions_and_update_velocity();
    }

    fn check_for_collisions_and_update_velocity(&mut self) -> Vec<Ball> {
        let mut sorted_balls = self.balls.to_vec();
        sorted_balls.sort_by(|a, b| a.location.x.partial_cmp(&b.location.x).unwrap());
        let mut velocity_updates: Vec<Vector2D> = Vec::new();
        let mut location_updates: Vec<Vector2D> = Vec::new();

        for i in 0..sorted_balls.len() {
            velocity_updates.push(sorted_balls[i].velocity);
            location_updates.push(sorted_balls[i].location);
        }

        for i in 0..sorted_balls.len() {
            for j in (i + 1)..sorted_balls.len() {
                let ball1 = &sorted_balls[i];
                let ball2 = &sorted_balls[j];
            
                let is_collision = location_updates[i].subtract(&location_updates[j]).norm() <= ball1.radius + ball2.radius;

                if !is_collision {
                    break;
                }

                // Resolve weird collision 2
                // TODO: Solve parametric equation solution to find right intersection point, then backtrack delta_t
                //       Then, compute collision response and compensate for time

                // Solving parametric equations for backtracking
                // source: http://people.scs.carleton.ca/~nussbaum/courses/COMP3501/notes/collision_2012.pdf
                let v = ball1.prev_location.subtract(&ball2.prev_location);
                let u = (location_updates[i].subtract(&ball1.prev_location)).subtract(&location_updates[j].subtract(&ball2.prev_location));
                let uv = u.dot(&v);
                let u_squared = u.norm().powi(2);
                let v_squared = v.norm().powi(2);

                let determinant = (uv.powi(2) - (u_squared * (v_squared - (ball1.radius + ball2.radius).powi(2)))).sqrt();
                let t2 = (-uv - determinant) / u_squared;
                let mut backtrack_time = 0.0;                
                if t2 < 1.0 {
                    location_updates[i] = ball1.prev_location.add(&(location_updates[i].subtract(&ball1.prev_location)).scale(t2));
                    location_updates[j] = ball2.prev_location.add(&(location_updates[j].subtract(&ball2.prev_location)).scale(t2));
                    backtrack_time = (location_updates[i].subtract(&ball1.prev_location)).scale(1.0 - t2).norm() / velocity_updates[i].norm();
                    let backtrack_time2 = (location_updates[j].subtract(&ball2.prev_location)).scale(1.0 - t2).norm() / velocity_updates[j].norm();
                    if (backtrack_time - backtrack_time2).abs() > 1e-6 {
                        println!("Wow this should not be happening {}, {}", backtrack_time, backtrack_time2);
                    }
                }

                // Update the particle velocities
                let v1_minus_v2 = velocity_updates[i].subtract(&velocity_updates[j]);
                let x1_minus_x2 = location_updates[i].subtract(&location_updates[j]);
                let distance = x1_minus_x2.norm();
                
                let mass_term_1 = (2.0 * ball2.mass) / (ball1.mass + ball2.mass);
                let dot_product_term_1 = v1_minus_v2.dot(&x1_minus_x2) / (distance * distance);
                let velocity_ball1 = velocity_updates[i].subtract(&x1_minus_x2.scale(dot_product_term_1 * mass_term_1));

                let mass_term_2 = (2.0 * ball1.mass) / (ball1.mass + ball2.mass);
                let v2_minus_v1 = v1_minus_v2.scale(-1.0);
                let x2_minus_x1 = x1_minus_x2.scale(-1.0);
                let dot_product_term_2 = v2_minus_v1.dot(&x2_minus_x1) / (distance * distance);
                let velocity_ball2 = velocity_updates[j].subtract(&x2_minus_x1.scale(dot_product_term_2 * mass_term_2));
                
                velocity_updates[i] = velocity_ball1;
                velocity_updates[j] = velocity_ball2;

                if t2 < 1.0 {
                    location_updates[i] = location_updates[i].add(&velocity_updates[i].scale(backtrack_time));
                    location_updates[j] = location_updates[j].add(&velocity_updates[j].scale(backtrack_time));
                }
            }
        }

        for i in 0..sorted_balls.len() {
            sorted_balls[i].velocity = velocity_updates[i];
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
            prev_location: Vector2D { x: 0.0, y: 0.0 },
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
        simulation_factor: 1.0,
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
