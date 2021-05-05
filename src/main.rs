extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;

use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;

struct Trajectory2D {
    m: f64, // slope of the line
    b: f64, // y-intercept of the line
    x_update: f64, // x update direction and scale factor
    current_location: Point2D
}

struct Point2D {
    x: f64, 
    y: f64
}

pub struct Simulation {
    gl: GlGraphics, // OpenGL drawing backend.
    trajectory: Trajectory2D,  // Rotation for the circle.
    resolution: (f64,f64),
}

impl Simulation {
    fn calculate_new_location(&self, traj: &Trajectory2D) -> Point2D {
        let x = traj.current_location.x + traj.x_update;
        Point2D {
            x: x,
            y: traj.m * x + traj.b
        }
    }

    fn render(&mut self, args: &RenderArgs) {
        use graphics::*;

        const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
        const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        // const YELLOW: [f32; 4] = [1.0, 1.0, 0.0, 1.0];

        let circle = rectangle::centered_square(0.0, 0.0, 50.0);

        self.trajectory.current_location = self.calculate_new_location(&self.trajectory);
        let (x1, y1) = (self.trajectory.current_location.x, self.trajectory.current_location.y);

        self.gl.draw(args.viewport(), |c, gl| {
            // Clear the screen.
            clear(BLACK, gl);

            let transform1 = c
                .transform
                .trans(x1, y1);

            ellipse(RED, circle, transform1, gl);
        });
    }

    fn update(&mut self, args: &UpdateArgs) {
        if self.trajectory.current_location.y + 50.0 > self.resolution.1 {
            self.trajectory.m = self.trajectory.m * -1.0;
            self.trajectory.b = 2.0*(self.resolution.1-50.0);
        }

        // Rotate 2 radians per second.
        // self.location.0 = 100.0;
        // self.location.1 += 1.0;

        // self.location2.0 += args.dt*50.0;
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

    // Create a new game and run it.
    let mut simulation = Simulation {
        gl: GlGraphics::new(opengl),
        trajectory: Trajectory2D {
            m: 1.0,
            b: 0.0,
            x_update: 2.0,
            current_location: Point2D {
                x: 0.0, y: 0.0
            }
        },
        resolution: (width, height)
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