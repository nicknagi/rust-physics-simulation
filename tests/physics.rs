//! Behavioural tests for the simulation core.
//!
//! These pin the properties the rewrite is supposed to preserve (conservation,
//! containment, finiteness) and the specific defects it set out to fix
//! (coincident-particle NaN, the unbounded separation loop, races in the
//! parallel collision pass).

use simulation::{Config, DiscSpawn, Sim};

fn base_cfg() -> Config {
    Config {
        width: 400.0,
        height: 400.0,
        radius: 10.0,
        gravity: false,
        parallel: false,
        ..Config::default()
    }
}

/// Two equal masses meeting head-on, far from any wall. With restitution 1 the
/// pair must conserve both momentum and kinetic energy.
#[test]
fn head_on_collision_conserves_momentum_and_energy() {
    let mut sim = Sim::new(base_cfg());
    sim.set_particles(
        vec![180.0, 220.0],
        vec![200.0, 200.0],
        vec![30.0, -30.0],
        vec![0.0, 0.0],
    );

    let p0 = sim.momentum();
    let e0 = sim.kinetic_energy();
    for _ in 0..200 {
        sim.step(1.0 / 240.0);
    }
    // Guard against a vacuous pass: the pair must actually have collided.
    assert!(
        sim.vx[0] < 0.0 && sim.vx[1] > 0.0,
        "particles never collided, so conservation was not exercised"
    );
    let p1 = sim.momentum();
    let e1 = sim.kinetic_energy();

    assert!(
        (p1.0 - p0.0).abs() / e0.max(1.0) < 1e-9,
        "momentum x drifted: {p0:?} -> {p1:?}"
    );
    assert!(
        (p1.1 - p0.1).abs() / e0.max(1.0) < 1e-9,
        "momentum y drifted: {p0:?} -> {p1:?}"
    );
    assert!(
        ((e1 - e0) / e0).abs() < 1e-5,
        "energy drifted: {e0:e} -> {e1:e}"
    );
}

/// A head-on pair must actually reverse, not pass through each other.
#[test]
fn head_on_collision_reverses_velocities() {
    let mut sim = Sim::new(base_cfg());
    sim.set_particles(
        vec![180.0, 220.0],
        vec![200.0, 200.0],
        vec![30.0, -30.0],
        vec![0.0, 0.0],
    );
    for _ in 0..200 {
        sim.step(1.0 / 240.0);
    }
    assert!(sim.vx[0] < 0.0, "left particle should have bounced back");
    assert!(sim.vx[1] > 0.0, "right particle should have bounced back");
}

/// The original divided by an unchecked norm, so exactly coincident particles
/// produced NaN, and its separation loop could never push them apart.
#[test]
fn coincident_particles_separate_without_nan() {
    let mut sim = Sim::new(base_cfg());
    sim.set_particles(
        vec![200.0, 200.0],
        vec![200.0, 200.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
    );

    for _ in 0..10 {
        sim.step(1.0 / 240.0);
    }

    assert!(!sim.has_non_finite(), "coincident pair produced NaN/inf");
    let dx = sim.px[1] - sim.px[0];
    let dy = sim.py[1] - sim.py[0];
    assert!(
        (dx * dx + dy * dy).sqrt() > 1.0,
        "coincident particles never separated"
    );
}

/// Zero-velocity overlapping particles: the original's loop advanced position by
/// `-SMALL_T * velocity`, so with no velocity it could not terminate. This must
/// finish and separate them.
#[test]
fn stationary_overlap_resolves() {
    let mut sim = Sim::new(base_cfg());
    sim.set_particles(
        vec![200.0, 205.0],
        vec![200.0, 200.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
    );
    for _ in 0..40 {
        sim.step(1.0 / 240.0);
    }
    let dx = sim.px[1] - sim.px[0];
    let dy = sim.py[1] - sim.py[0];
    let dist = (dx * dx + dy * dy).sqrt();
    assert!(
        dist > 19.0,
        "overlap not resolved: separation {dist} for diameter 20"
    );
    assert!(!sim.has_non_finite());
}

/// Collisions must be found across grid cell boundaries, not just within a cell.
#[test]
fn collisions_detected_across_cell_boundaries() {
    let mut sim = Sim::new(base_cfg());
    // Cells are one diameter (20px) wide, so these two sit in adjacent cells.
    sim.set_particles(
        vec![199.0, 214.0],
        vec![200.0, 200.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
    );
    sim.step(1.0 / 240.0);
    let dist = (sim.px[1] - sim.px[0]).abs();
    assert!(
        dist > 15.0,
        "cross-cell overlap was missed: separation {dist}"
    );
}

#[test]
fn particles_stay_in_bounds_and_finite() {
    for gravity in [false, true] {
        let cfg = Config {
            width: 800.0,
            height: 600.0,
            radius: 5.0,
            gravity,
            parallel: true,
            ..Config::default()
        };
        let mut sim = Sim::new(cfg);
        sim.spawn_random(3_000, 7);
        for _ in 0..300 {
            sim.step(1.0 / 240.0);
        }
        assert!(
            !sim.has_non_finite(),
            "non-finite state (gravity={gravity})"
        );
        assert_eq!(
            sim.out_of_bounds_count(),
            0,
            "particles escaped (gravity={gravity})"
        );
    }
}

/// Gravity mode must stay bounded: the speed clamp is what keeps it stable.
#[test]
fn gravity_stays_bounded() {
    let cfg = Config {
        width: 600.0,
        height: 600.0,
        radius: 4.0,
        gravity: true,
        parallel: true,
        ..Config::default()
    };
    let mut sim = Sim::new(cfg);
    sim.spawn_random(2_000, 99);
    for _ in 0..400 {
        sim.step(1.0 / 240.0);
    }
    assert!(!sim.has_non_finite());
    let max_speed = (0..sim.len())
        .map(|i| (sim.vx[i] * sim.vx[i] + sim.vy[i] * sim.vy[i]).sqrt())
        .fold(0.0f32, f32::max);
    assert!(
        max_speed <= cfg.max_speed * 1.01,
        "speed clamp breached: {max_speed}"
    );
}

/// Repeating a parallel run must give bit-identical results. The parallel
/// collision pass writes through raw pointers; a genuine data race would show up
/// here as run-to-run divergence.
#[test]
fn parallel_runs_are_deterministic() {
    let run = || {
        let cfg = Config {
            width: 800.0,
            height: 600.0,
            radius: 5.0,
            gravity: true,
            parallel: true,
            ..Config::default()
        };
        let mut sim = Sim::new(cfg);
        sim.spawn_random(4_000, 4242);
        for _ in 0..120 {
            sim.step(1.0 / 240.0);
        }
        (
            sim.px.clone(),
            sim.py.clone(),
            sim.vx.clone(),
            sim.vy.clone(),
        )
    };

    let a = run();
    let b = run();
    assert_eq!(a.0, b.0, "x positions diverged between identical runs");
    assert_eq!(a.1, b.1, "y positions diverged between identical runs");
    assert_eq!(a.2, b.2, "x velocities diverged between identical runs");
    assert_eq!(a.3, b.3, "y velocities diverged between identical runs");
}

/// Serial and parallel apply pairs in a different order, so they are not
/// bit-identical -- but they must agree closely and both stay physical.
#[test]
fn serial_and_parallel_agree_statistically() {
    let run = |parallel: bool| {
        let cfg = Config {
            width: 800.0,
            height: 600.0,
            radius: 5.0,
            gravity: false,
            parallel,
            ..Config::default()
        };
        let mut sim = Sim::new(cfg);
        sim.spawn_random(3_000, 31337);
        for _ in 0..60 {
            sim.step(1.0 / 240.0);
        }
        sim
    };

    let s = run(false);
    let p = run(true);
    assert!(!s.has_non_finite() && !p.has_non_finite());

    let ke_s = s.kinetic_energy();
    let ke_p = p.kinetic_energy();
    assert!(
        ((ke_s - ke_p) / ke_s).abs() < 0.02,
        "serial/parallel energy mismatch: {ke_s:e} vs {ke_p:e}"
    );
}

/// Barnes-Hut with theta=0 never accepts an approximation, so it degenerates to
/// the exact pairwise sum. Compare the acceleration *field* directly: an N-body
/// system is chaotic, so comparing evolved trajectories measures Lyapunov
/// divergence rather than approximation quality.
#[test]
fn barnes_hut_matches_exact_acceleration_field() {
    let field = |theta: f32| {
        let cfg = Config {
            width: 500.0,
            height: 500.0,
            radius: 3.0,
            gravity: true,
            parallel: false,
            theta,
            ..Config::default()
        };
        let mut sim = Sim::new(cfg);
        sim.spawn_random(800, 2024);
        sim.refresh_accelerations();
        let (ax, ay) = sim.accelerations();
        (ax.to_vec(), ay.to_vec())
    };

    let (ex, ey) = field(0.0);
    let mean_rel_err = |theta: f32| {
        let (ax, ay) = field(theta);
        let mut total = 0.0f64;
        for i in 0..ex.len() {
            let exact_mag = (ex[i] * ex[i] + ey[i] * ey[i]).sqrt();
            let dx = ax[i] - ex[i];
            let dy = ay[i] - ey[i];
            let err = (dx * dx + dy * dy).sqrt();
            total += (err / (exact_mag + 1e-6)) as f64;
        }
        total / ex.len() as f64
    };

    let coarse = mean_rel_err(1.0);
    let normal = mean_rel_err(0.7);
    let fine = mean_rel_err(0.3);

    assert!(
        normal < 0.05,
        "theta=0.7 mean relative error too high: {normal:.4}"
    );
    // Tightening the opening angle must monotonically improve accuracy; if it
    // does not, the traversal or the opening test is wrong.
    assert!(
        fine < normal && normal < coarse,
        "error should shrink with theta: 0.3 -> {fine:.4}, 0.7 -> {normal:.4}, 1.0 -> {coarse:.4}"
    );
}

#[test]
fn empty_and_single_particle_sims_do_not_panic() {
    let mut sim = Sim::new(base_cfg());
    sim.spawn_random(0, 1);
    sim.step(1.0 / 240.0);

    let mut sim = Sim::new(base_cfg());
    sim.spawn_random(1, 1);
    for _ in 0..50 {
        sim.step(1.0 / 240.0);
    }
    assert!(!sim.has_non_finite());
}

/// A whirl must actually persist: an orbiting ring should keep almost all of
/// its particles in the annulus it was launched in.
///
/// The configuration here is not arbitrary -- it is what the measurements
/// settled on. A ring needs a dominant central mass (self-gravity alone clumps
/// and heats it apart), a small collision radius (collisions in a sheared disc
/// spread it viscously), and only weak mutual gravity.
#[test]
fn whirl_persists() {
    let n = 2000;
    let cfg = Config {
        width: 1600.0,
        height: 900.0,
        radius: 0.4,
        gravity: true,
        parallel: true,
        mass: 1e16 * 1.0e-5,
        softening: 8.0,
        central_gm: 2.0e7,
        ..Config::default()
    };
    let mut sim = Sim::new(cfg);
    let fastest = sim.spawn_orbital_disc(DiscSpawn {
        count: n,
        seed: 12345,
        inner_radius: 104.0,
        outer_radius: 297.0,
        clockwise: false,
    });
    // Clamping an orbit destroys the angular momentum holding the ring up.
    sim.cfg.max_speed = fastest * 4.0;

    let (cx, cy) = (800.0f32, 450.0f32);
    let retained = |s: &Sim| -> f32 {
        let k = (0..s.len())
            .filter(|&i| {
                let r = ((s.px[i] - cx).powi(2) + (s.py[i] - cy).powi(2)).sqrt();
                (85.0..330.0).contains(&r)
            })
            .count();
        k as f32 / s.len() as f32
    };
    let spins = |s: &Sim| -> f64 {
        (0..s.len())
            .map(|i| {
                let dx = (s.px[i] - cx) as f64;
                let dy = (s.py[i] - cy) as f64;
                dx * s.vy[i] as f64 - dy * s.vx[i] as f64
            })
            .sum()
    };

    assert!(fastest > 100.0, "orbits are implausibly slow: {fastest}");
    let l0 = spins(&sim);

    // 30 seconds of simulated time.
    for _ in 0..7200 {
        sim.step(1.0 / 240.0);
    }

    assert!(!sim.has_non_finite());
    let kept = retained(&sim);
    assert!(
        kept > 0.9,
        "whirl fell apart: only {:.0}% of particles still in the ring",
        kept * 100.0
    );
    assert!(
        spins(&sim).signum() == l0.signum(),
        "ring reversed direction"
    );
}
