# Rust Physics Simulation

A 2D particle physics simulation in Rust. Started as a project to learn the
language; since rewritten for throughput.

## How To Run

```
cargo run --release -- --particles 20000
cargo run --release -- --particles 5000 --gravity
cargo run --release -- --help
```

Controls: `G` toggles gravity, `Space` pauses, `R` respawns, `Esc` quits.

`--interactive` restores the original stdin prompts.

Headless benchmark, which measures the physics with no window or vsync in the way:

```
cargo run --release --bin bench -- --sweep
cargo run --release --bin bench -- --particles 100000 --gravity
```

## Implementation

**State layout.** Particles live in structure-of-arrays form (`px`, `py`, `vx`,
`vy` as separate `f32` vectors) rather than a `Vec<Ball>`. The hot loops only
touch position and velocity, so keeping colour out of them fits far more
particles per cache line. Particle index is stable for the life of the sim.

**Collision broad phase.** A uniform spatial-hash grid with cells one diameter
wide, rebuilt each step with a counting sort (O(n), no per-cell allocation). A
particle can only touch its own cell and the ring around it, so the narrow phase
visits a bounded number of candidates instead of all `n`. The previous version
sorted by x and described itself as sweep-and-prune, but never used the sort to
prune, so it still compared every pair.

**Collision response.** Equal-mass elastic impulse plus a one-step analytic
positional correction.

**Gravity.** A Barnes-Hut quadtree: any group of distant particles is collapsed
to its centre of mass once it subtends a small enough angle, taking gravity from
O(n²) to O(n log n). Nodes are stored flat with contiguous children. `--theta`
controls the accuracy/speed tradeoff; `--theta 0` degenerates to the exact
pairwise sum and is used in tests as the reference.

**Parallelism.** Rayon across all cores. Force evaluation is read-only against
the tree and parallelises trivially. Collision resolution uses an even/odd grid
row split: the neighbour offsets only reach row `+1`, so same-parity rows touch
disjoint particle sets. That makes the parallel pass race-free *and*
deterministic — a property the test suite asserts directly, since a real race
would show up as run-to-run divergence.

**Timestep.** Fixed, with an accumulator, so behaviour no longer depends on frame
rate. The original integrated straight off the frame delta.

**Rendering.** All particles are expanded against a shared unit-circle triangle
template and streamed through a single batched `tri_list_c`. The original issued
one `ellipse()` call per particle.

## Benchmarks

Apple M1 Pro (8 cores), release build, radius 4, domain scaled to hold packing
fraction constant across sizes. "original" is the 2021 implementation run
headless on identical initial conditions.

Physics, milliseconds per step:

| particles | collisions: before | after | | gravity: before | after | |
|---|---|---|---|---|---|---|
| 1,000 | 6.29 | **0.16** | 39× | 2.13 | **0.43** | 5× |
| 5,000 | 42.2 | **0.34** | 124× | 54.0 | **1.36** | 40× |
| 20,000 | 379 | **0.75** | 506× | 863 | **5.57** | 155× |
| 50,000 | 1,689 | **1.27** | 1,329× | 5,443 | **14.1** | 385× |

100,000 particles run at 2.3 ms/step (collisions) and 29 ms/step (gravity); the
original was not measurable at that size in reasonable time.

Rendering, milliseconds per frame (`--naive-render` selects the old path):

| particles | batched | per-particle `ellipse()` | |
|---|---|---|---|
| 1,000 | 0.34 | 1.70 | 5.0× |
| 10,000 | 2.78 | 18.9 | 6.8× |
| 50,000 | 13.9 | 96.4 | 7.0× |

Two caveats, stated plainly:

- Rendering is now the bottleneck, not physics. At 50k particles the physics
  costs ~3 ms/frame and the draw ~14 ms.
- Frame rates measured from an unfocused window are pinned near 30 fps by macOS
  throttling, which is why the tables above report per-operation cost rather
  than fps.

## Fixes carried in the rewrite

- **Unbounded loop.** Overlap was resolved by repeatedly stepping particles back
  by `-1e-5 * velocity` with no iteration cap and no termination proof. Slow or
  coincident pairs could spin for millions of iterations. It is now solved in
  one step. This was also why the original's gravity mode benchmarked *faster*
  than its collision mode: gravity took an early `break` that skipped the loop
  entirely.
- **Division by zero.** `normalize()` divided by an unchecked norm, so exactly
  coincident particles produced NaN. Degenerate cases now get a deterministic
  fallback normal.
- **Unstable particle identity.** The ball array was sorted by x every step, so
  an index did not refer to the same particle across frames. Ordering now lives
  in side tables and particles never move.
- **Aggregate clamping (introduced and caught during this rewrite).** Porting the
  original's per-pair acceleration cap to Barnes-Hut naively clamps a whole node
  to one pair's limit, understating a distant cluster by its particle count. The
  correct bound is `count × max_accel`.

## Tests

`cargo test` covers momentum and energy conservation in elastic collisions,
cross-cell collision detection, containment and finiteness under load, the
gravity speed clamp, parallel determinism, and Barnes-Hut accuracy against the
exact pairwise sum (including that error shrinks monotonically as `theta`
tightens).

## Dependencies

`cargo audit` reports no vulnerabilities. Four transitive crates are flagged
unmaintained (`rusttype`, `ttf-parser` ×2, `paste`); all arrive through piston's
font and image-codec stack, which this project does not use and cannot opt out
of from here.

`.cargo/config.toml` sets `target-cpu=native`. Delete it when building portable
binaries.
