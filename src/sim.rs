//! Simulation core: state, integration, collision resolution.
//!
//! State is structure-of-arrays (`px`, `py`, `vx`, `vy` as separate `f32`
//! vectors) rather than a `Vec<Ball>`. The hot loops only touch position and
//! velocity, so keeping colour out of those arrays roughly triples the number
//! of particles per cache line.
//!
//! Particle index is stable for the lifetime of the sim. The original sorted
//! its ball array by x every step, which destroyed identity; here ordering
//! lives in side tables (`grid`, `bh`) and never moves the particles themselves.

use std::marker::PhantomData;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::bh::BarnesHut;
use crate::grid::{Grid, NEIGHBOURS};

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub width: f32,
    pub height: f32,
    pub radius: f32,
    pub mass: f32,
    pub gravity: bool,
    /// Gravitational constant.
    pub g: f32,
    /// Speed ceiling. Load-bearing: it is what keeps gravity mode stable.
    pub max_speed: f32,
    /// Per-interaction acceleration ceiling, matching the original's clamp.
    pub max_accel: f32,
    /// 1.0 = perfectly elastic.
    pub restitution: f32,
    /// Barnes-Hut opening angle. Larger = faster and coarser. 0 = exact O(n^2).
    pub theta: f32,
    /// Plummer softening, stops close pairs producing huge accelerations.
    pub softening: f32,
    pub parallel: bool,
    pub solver_iterations: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            width: 1200.0,
            height: 600.0,
            radius: 10.0,
            mass: 1e16,
            gravity: false,
            g: 6.674_08e-11,
            max_speed: 100.0,
            max_accel: 100.0,
            restitution: 1.0,
            theta: 0.7,
            softening: 4.0,
            parallel: true,
            solver_iterations: 1,
        }
    }
}

pub struct Sim {
    pub cfg: Config,
    pub px: Vec<f32>,
    pub py: Vec<f32>,
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
    /// Render-only, kept out of the physics arrays on purpose.
    pub color: Vec<[f32; 4]>,
    ax: Vec<f32>,
    ay: Vec<f32>,
    grid: Grid,
    bh: BarnesHut,
}

impl Sim {
    pub fn new(cfg: Config) -> Self {
        // One diameter per cell: a particle can then only touch its own cell
        // and the ring around it.
        let grid = Grid::new(cfg.radius * 2.0, cfg.width, cfg.height);
        Sim {
            grid,
            bh: BarnesHut::new(cfg.theta),
            cfg,
            px: Vec::new(),
            py: Vec::new(),
            vx: Vec::new(),
            vy: Vec::new(),
            color: Vec::new(),
            ax: Vec::new(),
            ay: Vec::new(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.px.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.px.is_empty()
    }

    pub fn tree_nodes(&self) -> usize {
        self.bh.node_count()
    }

    /// Fill the sim with `n` randomly placed particles. Deterministic for a
    /// given seed so benchmarks and tests are reproducible.
    pub fn spawn_random(&mut self, n: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let r = self.cfg.radius;
        let w = self.cfg.width;
        let h = self.cfg.height;

        self.px.clear();
        self.py.clear();
        self.vx.clear();
        self.vy.clear();
        self.color.clear();
        for _ in 0..n {
            self.px.push(rng.random_range(r..(w - r).max(r + 1.0)));
            self.py.push(rng.random_range(r..(h - r).max(r + 1.0)));
            self.vx.push(rng.random_range(-100.0..100.0));
            self.vy.push(rng.random_range(-100.0..100.0));
            self.color.push([
                rng.random_range(0.2..1.0),
                rng.random_range(0.2..1.0),
                rng.random_range(0.2..1.0),
                rng.random_range(0.5..1.0),
            ]);
        }
        self.ax.clear();
        self.ax.resize(n, 0.0);
        self.ay.clear();
        self.ay.resize(n, 0.0);
    }

    /// Install an exact particle set. Used by tests to build deterministic
    /// scenarios; panics if the component vectors disagree in length.
    pub fn set_particles(&mut self, px: Vec<f32>, py: Vec<f32>, vx: Vec<f32>, vy: Vec<f32>) {
        let n = px.len();
        assert!(
            py.len() == n && vx.len() == n && vy.len() == n,
            "particle component vectors must all have the same length"
        );
        self.px = px;
        self.py = py;
        self.vx = vx;
        self.vy = vy;
        self.color.clear();
        self.color.resize(n, [1.0, 1.0, 1.0, 1.0]);
        self.ax.clear();
        self.ax.resize(n, 0.0);
        self.ay.clear();
        self.ay.resize(n, 0.0);
    }

    pub fn step(&mut self, dt: f32) {
        if self.is_empty() {
            return;
        }
        self.compute_accelerations();
        self.integrate_and_bound(dt);
        self.grid.rebuild(&self.px, &self.py);
        for _ in 0..self.cfg.solver_iterations {
            self.resolve_collisions();
        }
        self.finalize();
    }

    fn compute_accelerations(&mut self) {
        if !self.cfg.gravity {
            return;
        }
        let half = self.cfg.width.max(self.cfg.height) * 0.5;
        self.bh.set_theta(self.cfg.theta);
        self.bh.build(
            &self.px,
            &self.py,
            self.cfg.mass,
            self.cfg.width * 0.5,
            self.cfg.height * 0.5,
            half,
        );

        let Sim {
            px,
            py,
            ax,
            ay,
            bh,
            cfg,
            ..
        } = self;
        let soft_sq = cfg.softening * cfg.softening;
        const CHUNK: usize = 256;

        let eval = |chunk_index: usize, axc: &mut [f32], ayc: &mut [f32]| {
            let mut stack: Vec<u32> = Vec::with_capacity(128);
            let base = chunk_index * CHUNK;
            for k in 0..axc.len() {
                let i = base + k;
                let (a, b) = bh.accel(
                    px[i],
                    py[i],
                    i as u32,
                    px,
                    py,
                    cfg.g,
                    cfg.mass,
                    cfg.max_accel,
                    soft_sq,
                    &mut stack,
                );
                axc[k] = a;
                ayc[k] = b;
            }
        };

        if cfg.parallel {
            ax.par_chunks_mut(CHUNK)
                .zip(ay.par_chunks_mut(CHUNK))
                .enumerate()
                .for_each(|(ci, (axc, ayc))| eval(ci, axc, ayc));
        } else {
            ax.chunks_mut(CHUNK)
                .zip(ay.chunks_mut(CHUNK))
                .enumerate()
                .for_each(|(ci, (axc, ayc))| eval(ci, axc, ayc));
        }
    }

    fn integrate_and_bound(&mut self, dt: f32) {
        let Sim {
            px,
            py,
            vx,
            vy,
            ax,
            ay,
            cfg,
            ..
        } = self;
        const CHUNK: usize = 2048;

        if cfg.parallel {
            px.par_chunks_mut(CHUNK)
                .zip(py.par_chunks_mut(CHUNK))
                .zip(vx.par_chunks_mut(CHUNK))
                .zip(vy.par_chunks_mut(CHUNK))
                .zip(ax.par_chunks(CHUNK))
                .zip(ay.par_chunks(CHUNK))
                .for_each(|(((((p_x, p_y), v_x), v_y), a_x), a_y)| {
                    integrate_chunk(p_x, p_y, v_x, v_y, a_x, a_y, dt, cfg);
                });
        } else {
            px.chunks_mut(CHUNK)
                .zip(py.chunks_mut(CHUNK))
                .zip(vx.chunks_mut(CHUNK))
                .zip(vy.chunks_mut(CHUNK))
                .zip(ax.chunks(CHUNK))
                .zip(ay.chunks(CHUNK))
                .for_each(|(((((p_x, p_y), v_x), v_y), a_x), a_y)| {
                    integrate_chunk(p_x, p_y, v_x, v_y, a_x, a_y, dt, cfg);
                });
        }
    }

    /// Re-containment plus speed clamp, run after collision resolution.
    ///
    /// Both are needed: positional correction can nudge a particle through a
    /// wall, and an elastic impulse can push one particle's speed above the cap
    /// even though the pair's energy is conserved. The original clamped inside
    /// its collision routine for the same reason -- the cap is what keeps
    /// gravity mode stable.
    fn finalize(&mut self) {
        let Sim {
            px,
            py,
            vx,
            vy,
            cfg,
            ..
        } = self;
        let (lo_x, hi_x, lo_y, hi_y) = bounds(cfg);
        let e = cfg.restitution;
        let max_speed = cfg.max_speed;
        let max_speed_sq = max_speed * max_speed;
        const CHUNK: usize = 2048;

        let apply = |px: &mut [f32], py: &mut [f32], vx: &mut [f32], vy: &mut [f32]| {
            for k in 0..px.len() {
                bound_one(&mut px[k], &mut vx[k], lo_x, hi_x, e);
                bound_one(&mut py[k], &mut vy[k], lo_y, hi_y, e);
                let speed_sq = vx[k] * vx[k] + vy[k] * vy[k];
                if speed_sq > max_speed_sq {
                    let s = max_speed / speed_sq.sqrt();
                    vx[k] *= s;
                    vy[k] *= s;
                }
            }
        };

        if cfg.parallel {
            px.par_chunks_mut(CHUNK)
                .zip(py.par_chunks_mut(CHUNK))
                .zip(vx.par_chunks_mut(CHUNK))
                .zip(vy.par_chunks_mut(CHUNK))
                .for_each(|(((p_x, p_y), v_x), v_y)| apply(p_x, p_y, v_x, v_y));
        } else {
            px.chunks_mut(CHUNK)
                .zip(py.chunks_mut(CHUNK))
                .zip(vx.chunks_mut(CHUNK))
                .zip(vy.chunks_mut(CHUNK))
                .for_each(|(((p_x, p_y), v_x), v_y)| apply(p_x, p_y, v_x, v_y));
        }
    }

    /// Recompute the acceleration field for the current positions without
    /// advancing time. Exposed so tests can compare Barnes-Hut against the
    /// exact pairwise sum directly, rather than through a chaotic trajectory.
    pub fn refresh_accelerations(&mut self) {
        self.compute_accelerations();
    }

    pub fn accelerations(&self) -> (&[f32], &[f32]) {
        (&self.ax, &self.ay)
    }

    /// Grid-accelerated narrow phase.
    ///
    /// Parallelism uses an even/odd row split. `grid::NEIGHBOURS` only reaches
    /// row `+1`, so the work for row `r` reads and writes particles in rows `r`
    /// and `r + 1` only. Same-parity rows are 2 apart, so their footprints
    /// (`{r, r+1}` and `{r+2, r+3}`) are disjoint -- no races, and the result is
    /// independent of thread scheduling.
    fn resolve_collisions(&mut self) {
        let Sim {
            px,
            py,
            vx,
            vy,
            grid,
            cfg,
            ..
        } = self;
        let rows = grid.rows();
        let cols = grid.cols();

        if cfg.parallel && rows >= 4 {
            let s_px = SyncSlice::new(px);
            let s_py = SyncSlice::new(py);
            let s_vx = SyncSlice::new(vx);
            let s_vy = SyncSlice::new(vy);
            for parity in 0..2usize {
                (0..rows)
                    .into_par_iter()
                    .filter(|r| r % 2 == parity)
                    .for_each(|row| {
                        // SAFETY: see the doc comment above -- same-parity rows
                        // touch disjoint particle sets.
                        unsafe { resolve_row(row, cols, grid, cfg, &s_px, &s_py, &s_vx, &s_vy) }
                    });
            }
        } else {
            let s_px = SyncSlice::new(px);
            let s_py = SyncSlice::new(py);
            let s_vx = SyncSlice::new(vx);
            let s_vy = SyncSlice::new(vy);
            for row in 0..rows {
                // SAFETY: single-threaded here.
                unsafe { resolve_row(row, cols, grid, cfg, &s_px, &s_py, &s_vx, &s_vy) }
            }
        }
    }

    // -- diagnostics, used by tests and the benchmark ------------------------

    pub fn kinetic_energy(&self) -> f64 {
        let m = self.cfg.mass as f64;
        (0..self.len())
            .map(|i| {
                let vx = self.vx[i] as f64;
                let vy = self.vy[i] as f64;
                0.5 * m * (vx * vx + vy * vy)
            })
            .sum()
    }

    pub fn momentum(&self) -> (f64, f64) {
        let m = self.cfg.mass as f64;
        let mut sx = 0.0;
        let mut sy = 0.0;
        for i in 0..self.len() {
            sx += m * self.vx[i] as f64;
            sy += m * self.vy[i] as f64;
        }
        (sx, sy)
    }

    pub fn has_non_finite(&self) -> bool {
        self.px
            .iter()
            .chain(&self.py)
            .chain(&self.vx)
            .chain(&self.vy)
            .any(|v| !v.is_finite())
    }

    pub fn out_of_bounds_count(&self) -> usize {
        let (lo_x, hi_x, lo_y, hi_y) = bounds(&self.cfg);
        let eps = 1e-2;
        (0..self.len())
            .filter(|&i| {
                self.px[i] < lo_x - eps
                    || self.px[i] > hi_x + eps
                    || self.py[i] < lo_y - eps
                    || self.py[i] > hi_y + eps
            })
            .count()
    }

    /// Deepest particle-particle overlap in the current state, in pixels.
    /// O(n^2) -- diagnostics only.
    pub fn max_overlap(&self) -> f32 {
        let min_d = self.cfg.radius * 2.0;
        let mut worst = 0.0f32;
        for i in 0..self.len() {
            for j in (i + 1)..self.len() {
                let dx = self.px[j] - self.px[i];
                let dy = self.py[j] - self.py[i];
                let d = (dx * dx + dy * dy).sqrt();
                if d < min_d {
                    worst = worst.max(min_d - d);
                }
            }
        }
        worst
    }
}

#[inline]
fn bounds(cfg: &Config) -> (f32, f32, f32, f32) {
    let r = cfg.radius;
    (r, (cfg.width - r).max(r), r, (cfg.height - r).max(r))
}

#[inline]
fn bound_one(p: &mut f32, v: &mut f32, lo: f32, hi: f32, e: f32) {
    if *p < lo {
        *p = lo;
        if *v < 0.0 {
            *v = -*v * e;
        }
    } else if *p > hi {
        *p = hi;
        if *v > 0.0 {
            *v = -*v * e;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn integrate_chunk(
    px: &mut [f32],
    py: &mut [f32],
    vx: &mut [f32],
    vy: &mut [f32],
    ax: &[f32],
    ay: &[f32],
    dt: f32,
    cfg: &Config,
) {
    let (lo_x, hi_x, lo_y, hi_y) = bounds(cfg);
    let max_speed_sq = cfg.max_speed * cfg.max_speed;
    let e = cfg.restitution;
    let gravity = cfg.gravity;

    for k in 0..px.len() {
        let mut v_x = vx[k];
        let mut v_y = vy[k];
        if gravity {
            v_x += ax[k] * dt;
            v_y += ay[k] * dt;
        }

        let speed_sq = v_x * v_x + v_y * v_y;
        if speed_sq > max_speed_sq {
            let s = cfg.max_speed / speed_sq.sqrt();
            v_x *= s;
            v_y *= s;
        }

        let mut x = px[k] + v_x * dt;
        let mut y = py[k] + v_y * dt;
        bound_one(&mut x, &mut v_x, lo_x, hi_x, e);
        bound_one(&mut y, &mut v_y, lo_y, hi_y, e);

        px[k] = x;
        py[k] = y;
        vx[k] = v_x;
        vy[k] = v_y;
    }
}

/// Resolve every collision pair owned by one grid row.
///
/// # Safety
/// Caller must guarantee no other thread is touching rows `row` or `row + 1`.
#[allow(clippy::too_many_arguments)]
unsafe fn resolve_row(
    row: usize,
    cols: usize,
    grid: &Grid,
    cfg: &Config,
    px: &SyncSlice<f32>,
    py: &SyncSlice<f32>,
    vx: &SyncSlice<f32>,
    vy: &SyncSlice<f32>,
) {
    for col in 0..cols {
        let items = grid.cell_items(row * cols + col);
        if items.is_empty() {
            continue;
        }

        // Pairs inside this cell.
        for a in 0..items.len() {
            for b in (a + 1)..items.len() {
                resolve_pair(items[a] as usize, items[b] as usize, cfg, px, py, vx, vy);
            }
        }

        // Pairs with the forward half of the neighbourhood, so each unordered
        // pair is considered exactly once.
        for (dx, dy) in NEIGHBOURS {
            let other = grid.cell_items_at(col as isize + dx, row as isize + dy);
            for &i in items {
                for &j in other {
                    resolve_pair(i as usize, j as usize, cfg, px, py, vx, vy);
                }
            }
        }
    }
}

/// Equal-mass elastic response plus analytic positional correction.
///
/// This replaces the original's unbounded `loop` that backed particles apart by
/// `-SMALL_T * velocity` until they separated: that had no termination proof and
/// stalled for slow or coincident pairs. Here the overlap is removed in one
/// step, and a zero-distance pair gets a deterministic fallback normal instead
/// of dividing by zero.
#[inline]
unsafe fn resolve_pair(
    i: usize,
    j: usize,
    cfg: &Config,
    px: &SyncSlice<f32>,
    py: &SyncSlice<f32>,
    vx: &SyncSlice<f32>,
    vy: &SyncSlice<f32>,
) {
    let pxi = px.at(i);
    let pyi = py.at(i);
    let pxj = px.at(j);
    let pyj = py.at(j);

    let dx = *pxj - *pxi;
    let dy = *pyj - *pyi;
    let dist_sq = dx * dx + dy * dy;
    let min_d = cfg.radius * 2.0;

    if dist_sq >= min_d * min_d {
        return;
    }

    let (nx, ny, dist) = if dist_sq > 1e-12 {
        let d = dist_sq.sqrt();
        (dx / d, dy / d, d)
    } else {
        // Exactly coincident: pick a fixed axis so the result stays
        // deterministic and finite.
        (1.0, 0.0, 0.0)
    };

    let correction = (min_d - dist) * 0.5;
    *pxi -= nx * correction;
    *pyi -= ny * correction;
    *pxj += nx * correction;
    *pyj += ny * correction;

    let vxi = vx.at(i);
    let vyi = vy.at(i);
    let vxj = vx.at(j);
    let vyj = vy.at(j);

    let rel_normal = (*vxj - *vxi) * nx + (*vyj - *vyi) * ny;
    if rel_normal < 0.0 {
        // Equal masses: the normal components swap (scaled by restitution).
        let impulse = rel_normal * (1.0 + cfg.restitution) * 0.5;
        *vxi += nx * impulse;
        *vyi += ny * impulse;
        *vxj -= nx * impulse;
        *vyj -= ny * impulse;
    }
}

/// Shared mutable view over a slice, for the disjoint-write pattern above.
struct SyncSlice<'a, T> {
    ptr: *mut T,
    _marker: PhantomData<&'a mut [T]>,
}

// SAFETY: callers are responsible for ensuring concurrent accesses target
// disjoint indices; see `Sim::resolve_collisions`.
unsafe impl<T: Send> Send for SyncSlice<'_, T> {}
unsafe impl<T: Send> Sync for SyncSlice<'_, T> {}

impl<'a, T> SyncSlice<'a, T> {
    fn new(s: &'a mut [T]) -> Self {
        SyncSlice {
            ptr: s.as_mut_ptr(),
            _marker: PhantomData,
        }
    }

    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn at(&self, i: usize) -> &mut T {
        &mut *self.ptr.add(i)
    }
}
