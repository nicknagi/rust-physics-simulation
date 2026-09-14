# Rust Physics Simulation

A 2D particle physics simulation in Rust. Started as a project to learn the
language; since rewritten for throughput, with both a multithreaded CPU backend
and a GPU backend (wgpu, running on Metal here).

## How To Run

```
cargo run --release -- --particles 20000
cargo run --release -- --particles 5000 --gravity
cargo run --release -- --help
```

Controls: `G` toggles gravity, `Space` pauses, `R` respawns, `Esc` quits.

`--interactive` restores the original stdin prompts.

GPU backend:

```
cargo run --release --bin gpu -- --particles 200000 --radius 1
cargo run --release --bin gpu -- --help
```

Headless benchmarks, which measure the physics with no window or vsync in the way:

```
cargo run --release --bin bench     -- --sweep          # CPU
cargo run --release --bin gpubench  -- --sweep          # GPU vs CPU
cargo run --release --bin gpubench  -- -n 5000 -s 200   # GPU, with validation
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

## GPU backend

`src/gpu.rs` plus `src/shaders/*.wgsl` run the whole simulation on the GPU
through wgpu (Metal on Apple Silicon). Particle state is uploaded once and then
never leaves GPU memory: the compute passes write the position buffer and the
render pipeline instances directly off that same buffer.

One step is four dispatches — `clear_bins`, `integrate`, `bin`, `collide` —
inside a single compute pass, which the WebGPU spec already orders and
memory-synchronises, so no explicit barriers are needed.

**Collision response gathers rather than scatters.** Each thread reads its
neighbours and writes only its own particle, so the pass needs no atomics and
has no races. The response is antisymmetric, so the partner thread derives
exactly the opposite impulse from the same inputs and momentum is conserved.
(The one case needing care is exactly coincident particles: the fallback
separation axis flips with index order, otherwise both threads would push the
same way and the pair would never separate.)

**Rendering** generates six vertices per particle in the vertex shader and
carves a disc out of them in the fragment shader. There is no vertex buffer and
no per-frame geometry upload.

### GPU vs CPU, milliseconds per step

Headless, M1 Pro, radius 4, domain scaled to constant packing:

| particles | mode | CPU | GPU | |
|---|---|---|---|---|
| 10,000 | collisions | 0.39 | **0.20** | 2.0× |
| 50,000 | collisions | 1.26 | **0.36** | 3.5× |
| 200,000 | collisions | 4.55 | **0.72** | 6.3× |
| 500,000 | collisions | 11.8 | **4.51** | 2.6× |
| 1,000,000 | collisions | 30.3 | **15.7** | 1.9× |
| 10,000 | gravity | 2.67 | 2.67 | 1.0× |
| 50,000 | gravity | **14.2** | 47.0 | 0.3× |
| 200,000 | gravity | **60.1** | 690 | 0.09× |

Interactively, the GPU build holds a vsync-locked 60 fps to at least 200k
particles with the CPU essentially idle; uncapped it runs 220–250 fps at
50k–200k and ~45 fps at 500k.

### Two honest limitations

**GPU gravity loses to CPU gravity above ~20k particles**, and the gap widens
fast. This is not a hardware result, it is an algorithmic one: the GPU does an
exact O(n²) pairwise sum while the CPU uses a Barnes-Hut tree at O(n log n).
A fast processor running the wrong complexity class still loses. The GPU answer
is the *more accurate* one — no opening-angle approximation — but for gravity at
scale, use the CPU binary. The GPU app prints a warning when you ask for this.
Fixing it properly means a tree or a multipole scheme on the GPU.

**Long unbroken GPU compute runs lose the device.** Roughly 2.2 seconds of
continuous compute on macOS drops the device and fails the next buffer map,
regardless of how the work is split across command buffers or how it is polled.
`GpuSim::run_steps` slices long runs with a hard fence between slices. The
windowed app never approaches this, since it runs at most 8 substeps per frame.

**GPU collision speedup peaks near 200k and then decays.** Particles are stored
in spawn order, so spatially adjacent particles are scattered through memory and
the neighbour gather degenerates into random access across a bin table that is
hundreds of megabytes at a million particles. The fix is to sort particles by
cell so the gather reads contiguously, which needs a GPU radix sort.

### GPU validation

`gpubench` checks the GPU result against the invariants the CPU path is tested
for. At 5,000 particles over 200 steps: all finite, zero out of bounds, zero
breaches of the speed clamp, maximum particle overlap 0.0001 px against an 8 px
diameter, and total kinetic energy within 2.2% of the CPU run. The two backends
are not expected to match bit-for-bit — the CPU resolves collision pairs in
sequence while the GPU gathers, so each particle sees its neighbours'
pre-collision state.


## The whirl preset

`gpu --whirl` spawns an orbiting ring -- particles circling a common centre.
`Sim::spawn_orbital_disc` builds it, and getting one that actually persists took
three separate corrections, each found by measurement rather than assumption.
`cargo run --release --example whirl_stability` reproduces the numbers.

**Orbital speeds cannot come from `v = sqrt(G * M_enclosed / r)`.** That is the
spherical shell theorem. This is a flat disc under a 1/r^2 force, where mass
outside a given radius does not cancel, and using it made the ring collapse
immediately (rms radius 248 -> 61 in five seconds). The spawner instead lays
down positions, evaluates the *real* acceleration field, and gives each particle
the speed `v = sqrt(a_inward * r)` that balances the pull it genuinely feels.

**Self-gravity alone cannot hold a ring together.** A cold self-gravitating disc
clumps, heats through close encounters, and spreads -- at every concentration,
softening and particle count tried. Stable orbits need a dominant central mass,
so `Config::central_gm` adds a fixed attractor at the centre of the domain,
implemented in both backends. With it the orbits are Keplerian; a single
particle holds its radius to within 0.3% over 50 seconds.

**Collisions spread a sheared ring even with gravity off.** Neighbouring
particles orbit at different speeds, so every collision transports angular
momentum outward -- the same viscous spreading that real accretion discs show.
At collision radius 1.5 the ring is gone in 20 seconds with self-gravity
completely disabled. The preset therefore uses near-point masses (radius 0.4)
and draws them larger than they collide, which is how particle discs are
normally rendered; `--dot-size` controls the drawn size independently.

Fraction of particles still in the ring, with a central mass at collision
radius 0.4:

| self-gravity | 10s | 20s | 30s | 40s | 50s | 60s |
|---|---|---|---|---|---|---|
| 1e-4 | 100% | 96% | 45% | 21% | 13% | 10% |
| 3e-5 | 100% | 100% | 100% | 78% | 29% | 17% |
| **1e-5** | **100%** | **100%** | **100%** | **100%** | **98%** | **51%** |
| off | 100% | 100% | 100% | 100% | 83% | 29% |

`--whirl` uses 1e-5. Note that a little mutual gravity beats none: weak cohesion
resists the collisional spreading. Angular momentum is conserved to within 2.3%
once walls and the speed clamp are kept out of the way -- both are external
forces and either one silently drains it, which is why the ring is sized with
margin to the walls and the speed clamp is lifted clear of the fastest orbit.

```
cargo run --release --bin gpu -- --whirl -n 4000
cargo run --release --bin gpu -- --whirl -n 4000 --gravity-scale 1e-4   # breaks up sooner
```


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
of from here. The wgpu and winit stack adds no advisories. Dropping piston in
favour of the wgpu front-end would clear all four.

`.cargo/config.toml` sets `target-cpu=native`. Delete it when building portable
binaries.
