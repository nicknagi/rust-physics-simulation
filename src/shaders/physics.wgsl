// GPU physics. One step is four dispatches: clear_bins -> integrate -> bin -> collide.
//
// Buffers are used as a fixed pair rather than a per-frame ping-pong:
//   integrate: pos0/vel0 -> pos1/vel1
//   bin:       reads pos1
//   collide:   pos1/vel1 -> pos0/vel0
// so a completed step always leaves the current state in pos0.

struct Params {
    width: f32,
    height: f32,
    radius: f32,
    dt: f32,

    count: u32,
    cols: u32,
    rows: u32,
    bin_capacity: u32,

    max_speed: f32,
    restitution: f32,
    gm: f32,
    max_accel: f32,

    softening_sq: f32,
    gravity: u32,
    cell_size: f32,
    /// G * M of a fixed attractor at the centre of the domain. Independent of
    /// `gravity`: a dominant central mass is what makes orbits Keplerian and
    /// therefore stable.
    central_gm: f32,

    /// Cursor-driven well: positive attracts, negative repels, 0 is off.
    /// Position is in domain space, not window space.
    well_gm: f32,
    well_x: f32,
    well_y: f32,
    _pad: f32,
};

/// The well is deliberately broad -- it should stir the field rather than spear
/// individual particles, so it softens over 40px against the physics' 4-8px.
const WELL_SOFTENING_SQ: f32 = 1600.0;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> bin_counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> bin_items: array<u32>;

@group(1) @binding(0) var<storage, read> pos_in: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read> vel_in: array<vec2<f32>>;
@group(1) @binding(2) var<storage, read_write> pos_out: array<vec2<f32>>;
@group(1) @binding(3) var<storage, read_write> vel_out: array<vec2<f32>>;

const WG: u32 = 256u;

fn cell_of(p: vec2<f32>) -> u32 {
    let cx = u32(clamp(floor(p.x / P.cell_size), 0.0, f32(P.cols - 1u)));
    let cy = u32(clamp(floor(p.y / P.cell_size), 0.0, f32(P.rows - 1u)));
    return cy * P.cols + cx;
}

// Reflect off the walls and cap speed. Mirrors the CPU path exactly.
fn contain(p: ptr<function, vec2<f32>>, v: ptr<function, vec2<f32>>) {
    let r = P.radius;
    let hi_x = max(P.width - r, r);
    let hi_y = max(P.height - r, r);
    let e = P.restitution;

    if ((*p).x < r) {
        (*p).x = r;
        if ((*v).x < 0.0) { (*v).x = -(*v).x * e; }
    } else if ((*p).x > hi_x) {
        (*p).x = hi_x;
        if ((*v).x > 0.0) { (*v).x = -(*v).x * e; }
    }
    if ((*p).y < r) {
        (*p).y = r;
        if ((*v).y < 0.0) { (*v).y = -(*v).y * e; }
    } else if ((*p).y > hi_y) {
        (*p).y = hi_y;
        if ((*v).y > 0.0) { (*v).y = -(*v).y * e; }
    }

    let speed_sq = dot(*v, *v);
    if (speed_sq > P.max_speed * P.max_speed) {
        *v = *v * (P.max_speed * inverseSqrt(speed_sq));
    }
}

@compute @workgroup_size(WG)
fn clear_bins(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= P.cols * P.rows) { return; }
    atomicStore(&bin_counts[c], 0u);
}

// Shared tile for the gravity pass: each workgroup stages 256 positions in
// workgroup memory so the O(n^2) sum reads mostly from fast local storage.
var<workgroup> tile: array<vec2<f32>, 256>;

@compute @workgroup_size(WG)
fn integrate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = gid.x;
    // Threads past the end must still reach every barrier below, so clamp the
    // index instead of returning early.
    let me = min(i, max(P.count, 1u) - 1u);
    let p0 = pos_in[me];
    var v = vel_in[me];
    var acc = vec2<f32>(0.0, 0.0);

    if (P.gravity != 0u) {
        // Exact pairwise gravity -- no Barnes-Hut approximation. The GPU can
        // afford the full sum at these sizes, so this is strictly more accurate
        // than the CPU path.
        let tiles = (P.count + WG - 1u) / WG;
        for (var t = 0u; t < tiles; t = t + 1u) {
            let src = t * WG + lid.x;
            if (src < P.count) {
                tile[lid.x] = pos_in[src];
            } else {
                tile[lid.x] = vec2<f32>(0.0, 0.0);
            }
            workgroupBarrier();

            let limit = min(WG, P.count - t * WG);
            for (var k = 0u; k < limit; k = k + 1u) {
                let idx = t * WG + k;
                if (idx != me) {
                    let d = tile[k] - p0;
                    let d2 = dot(d, d) + P.softening_sq;
                    let a = min(P.gm / d2, P.max_accel);
                    acc = acc + d * (a * inverseSqrt(d2));
                }
            }
            workgroupBarrier();
        }
    }

    if (i >= P.count) { return; }

    if (P.central_gm > 0.0) {
        let c = vec2<f32>(P.width * 0.5, P.height * 0.5);
        let d = c - p0;
        let d2 = dot(d, d) + P.softening_sq;
        acc = acc + d * (P.central_gm / d2 * inverseSqrt(d2));
    }

    if (P.well_gm != 0.0) {
        let w = vec2<f32>(P.well_x, P.well_y);
        let d = w - p0;
        let d2 = dot(d, d) + WELL_SOFTENING_SQ;
        acc = acc + d * (P.well_gm / d2 * inverseSqrt(d2));
    }

    if (P.gravity != 0u || P.central_gm > 0.0 || P.well_gm != 0.0) {
        v = v + acc * P.dt;
        let speed_sq = dot(v, v);
        if (speed_sq > P.max_speed * P.max_speed) {
            v = v * (P.max_speed * inverseSqrt(speed_sq));
        }
    }

    var p = p0 + v * P.dt;
    contain(&p, &v);
    pos_out[i] = p;
    vel_out[i] = v;
}

@compute @workgroup_size(WG)
fn bin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.count) { return; }
    let c = cell_of(pos_out[i]);
    let slot = atomicAdd(&bin_counts[c], 1u);
    // Cells beyond capacity drop their overflow; capacity is sized so this only
    // happens under extreme local density.
    if (slot < P.bin_capacity) {
        bin_items[c * P.bin_capacity + slot] = i;
    }
}

@compute @workgroup_size(WG)
fn collide(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.count) { return; }

    var p = pos_in[i];
    var v = vel_in[i];
    let min_d = P.radius * 2.0;
    let min_d_sq = min_d * min_d;

    let cx = i32(clamp(floor(p.x / P.cell_size), 0.0, f32(P.cols - 1u)));
    let cy = i32(clamp(floor(p.y / P.cell_size), 0.0, f32(P.rows - 1u)));

    // Gather, never scatter: this thread only ever writes its own particle, so
    // the pass needs no atomics and has no races. The response is antisymmetric,
    // so the partner thread derives exactly the opposite impulse from the same
    // inputs and momentum is conserved.
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        let ny = cy + dy;
        if (ny < 0 || ny >= i32(P.rows)) { continue; }
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let nx = cx + dx;
            if (nx < 0 || nx >= i32(P.cols)) { continue; }

            let c = u32(ny) * P.cols + u32(nx);
            let n = min(atomicLoad(&bin_counts[c]), P.bin_capacity);
            for (var k = 0u; k < n; k = k + 1u) {
                let j = bin_items[c * P.bin_capacity + k];
                if (j == i) { continue; }

                let d = pos_in[j] - p;
                let dist_sq = dot(d, d);
                if (dist_sq >= min_d_sq) { continue; }

                var nrm: vec2<f32>;
                var dist: f32;
                if (dist_sq > 1e-12) {
                    dist = sqrt(dist_sq);
                    nrm = d / dist;
                } else {
                    // Exactly coincident. The fallback axis must flip with index
                    // order, otherwise both threads would push the same way and
                    // the pair would never separate.
                    dist = 0.0;
                    if (i < j) { nrm = vec2<f32>(1.0, 0.0); } else { nrm = vec2<f32>(-1.0, 0.0); }
                }

                p = p - nrm * ((min_d - dist) * 0.5);

                let rel_normal = dot(vel_in[j] - v, nrm);
                if (rel_normal < 0.0) {
                    v = v + nrm * (rel_normal * (1.0 + P.restitution) * 0.5);
                }
            }
        }
    }

    contain(&p, &v);
    pos_out[i] = p;
    vel_out[i] = v;
}
