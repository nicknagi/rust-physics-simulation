//! Batched particle renderer.
//!
//! The original issued one `ellipse()` call per particle; each one re-tessellated
//! a circle and pushed its own tiny draw, so the render cost was dominated by
//! per-call overhead. Here every particle is expanded against a shared unit-circle
//! triangle template and streamed through a single `tri_list_c`, which the OpenGL
//! backend coalesces into a handful of draw calls regardless of particle count.

use graphics::math::Matrix2d;
use graphics::{DrawState, Graphics};

/// The backend's per-chunk vertex ceiling (`graphics::BACK_END_MAX_VERTEX_COUNT`).
const MAX_CHUNK_VERTS: usize = 1023;

pub struct ParticleRenderer {
    /// Unit-radius circle, pre-expanded from a fan into a flat triangle list.
    template: Vec<[f32; 2]>,
    verts: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
}

impl ParticleRenderer {
    pub fn new(segments: usize) -> Self {
        let segments = segments.clamp(3, 64);
        let mut template = Vec::with_capacity(segments * 3);
        for s in 0..segments {
            let a0 = s as f32 / segments as f32 * std::f32::consts::TAU;
            let a1 = (s + 1) as f32 / segments as f32 * std::f32::consts::TAU;
            template.push([0.0, 0.0]);
            template.push([a0.cos(), a0.sin()]);
            template.push([a1.cos(), a1.sin()]);
        }
        ParticleRenderer {
            template,
            verts: Vec::with_capacity(MAX_CHUNK_VERTS),
            colors: Vec::with_capacity(MAX_CHUNK_VERTS),
        }
    }

    /// Pick a circle resolution that stays smooth without wasting vertices on
    /// particles only a few pixels across.
    pub fn for_radius(radius: f32) -> Self {
        let segments = if radius < 2.0 {
            3
        } else if radius < 5.0 {
            6
        } else if radius < 12.0 {
            10
        } else {
            16
        };
        Self::new(segments)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw<G: Graphics>(
        &mut self,
        transform: Matrix2d,
        draw_state: &DrawState,
        px: &[f32],
        py: &[f32],
        colors: &[[f32; 4]],
        radius: f32,
        g: &mut G,
    ) {
        let per_particle = self.template.len();
        if per_particle == 0 || px.is_empty() {
            return;
        }

        // Flatten the 2x3 affine transform once; applying it per vertex in f32
        // is cheaper than going through the f64 matrix helpers.
        let m00 = transform[0][0] as f32;
        let m01 = transform[0][1] as f32;
        let m02 = transform[0][2] as f32;
        let m10 = transform[1][0] as f32;
        let m11 = transform[1][1] as f32;
        let m12 = transform[1][2] as f32;

        let mut verts = std::mem::take(&mut self.verts);
        let mut colbuf = std::mem::take(&mut self.colors);
        verts.clear();
        colbuf.clear();
        let template = &self.template;

        g.tri_list_c(draw_state, |flush| {
            for i in 0..px.len() {
                let cx = px[i];
                let cy = py[i];
                let col = colors[i];
                for t in template.iter() {
                    let x = cx + t[0] * radius;
                    let y = cy + t[1] * radius;
                    verts.push([m00 * x + m01 * y + m02, m10 * x + m11 * y + m12]);
                    colbuf.push(col);
                }
                if verts.len() + per_particle > MAX_CHUNK_VERTS {
                    flush(&verts, &colbuf);
                    verts.clear();
                    colbuf.clear();
                }
            }
            if !verts.is_empty() {
                flush(&verts, &colbuf);
                verts.clear();
                colbuf.clear();
            }
        });

        self.verts = verts;
        self.colors = colbuf;
    }
}

/// The original drawing strategy: one `ellipse()` call per particle. Kept so the
/// batching win can be measured directly rather than asserted -- select it with
/// `--naive-render`.
#[allow(clippy::too_many_arguments)]
pub fn draw_per_particle<G: Graphics>(
    transform: Matrix2d,
    draw_state: &DrawState,
    px: &[f32],
    py: &[f32],
    colors: &[[f32; 4]],
    radius: f32,
    g: &mut G,
) {
    use graphics::{Ellipse, Transformed};
    let r = radius as f64;
    let circle = [-r, -r, r * 2.0, r * 2.0];
    for i in 0..px.len() {
        let t = transform.trans(px[i] as f64, py[i] as f64);
        Ellipse::new(colors[i]).draw(circle, draw_state, t, g);
    }
}
