extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;

use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;

struct Ball {
    velocity: Vector2D,
    location: Vector2D
}

struct Vector2D {
    x: f64, 
    y: f64
}

pub struct Simulation {
    gl: GlGraphics, // OpenGL drawing backend.
    balls: Vec<Ball>,  // Collection of objects in scene
    resolution: (f64,f64),
    simulation_factor: u32
}

impl Simulation {
    fn render(&mut self, args: &RenderArgs) {
        use graphics::*;

        const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
        const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        // const YELLOW: [f32; 4] = [1.0, 1.0, 0.0, 1.0];

        let circle = rectangle::centered_square(0.0, 0.0, 50.0);

        let balls = &self.balls;

        self.gl.draw(args.viewport(), |c, gl| {
            // Clear the screen.
            clear(BLACK, gl);
            
            for ball in balls.iter() {
                let transform1 = c
                .transform
                .trans(ball.location.x, ball.location.y);

                ellipse(RED, circle, transform1, gl);
            }
        });
    }

    fn update(&mut self, args: &UpdateArgs) {
        // Rotate 2 radians per second.
        // self.location.0 = 100.0;
        // self.location.1 += 1.0;

        // self.location2.0 += args.dt*50.0;

        for ball in self.balls.iter_mut() {
            ball.location = Vector2D {
                x: ball.location.x + ball.velocity.x * self.simulation_factor as f64,
                y: ball.location.y + ball.velocity.y * self.simulation_factor as f64
            };
            if ball.location.y + 50.0 >= self.resolution.1 {
                ball.velocity = Vector2D {x: ball.velocity.x, y: -1.0 * ball.velocity.y};
            } else if ball.location.y - 50.0 < 0.0 {
                ball.velocity = Vector2D {x: ball.velocity.x, y: -1.0 * ball.velocity.y};
            } else if ball.location.x + 50.0 >= self.resolution.0 {
                ball.velocity = Vector2D {x: -1.0 * ball.velocity.x, y: ball.velocity.y};
            } else if ball.location.x - 50.0 < 0.0 {
                ball.velocity = Vector2D {x: -1.0 * ball.velocity.x, y: ball.velocity.y};
            }
        }


    }
}

fn main() {
    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

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
    balls.push(Ball {
        velocity: Vector2D{x: 1.0, y: 1.0},
        location: Vector2D{x: 50.0, y: 50.0}
    });

    // Create a new game and run it.
    let mut simulation = Simulation {
        gl: GlGraphics::new(opengl),
        balls: balls,
        resolution: (width, height),
        simulation_factor: 1
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