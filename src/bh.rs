//! Barnes-Hut quadtree for the N-body gravity term.
//!
//! The original summed every pair, so gravity cost O(n^2) per step and fell off
//! a cliff past a few hundred particles. This approximates any group of distant
//! particles by its centre of mass when the group subtends a small enough angle
//! (`size^2 < theta^2 * distance^2`), which brings a step to O(n log n).
//!
//! Layout notes: nodes live in one flat `Vec` and a node's four children are
//! always contiguous, so a traversal step touches one cache line rather than
//! chasing four pointers. Empty quadrants still get a node (mass 0) to keep that
//! contiguity -- traversal skips them on the `count == 0` test.

const NO_CHILD: u32 = u32::MAX;
const LEAF_CAPACITY: usize = 8;
const MAX_DEPTH: u32 = 24;

#[derive(Clone, Copy)]
struct Node {
    com_x: f32,
    com_y: f32,
    mass: f32,
    /// Full side length of this node's square, cached for the opening test.
    size: f32,
    first_child: u32,
    start: u32,
    count: u32,
}

impl Node {
    const EMPTY: Node = Node {
        com_x: 0.0,
        com_y: 0.0,
        mass: 0.0,
        size: 0.0,
        first_child: NO_CHILD,
        start: 0,
        count: 0,
    };
}

pub struct BarnesHut {
    nodes: Vec<Node>,
    /// Particle indices, recursively partitioned into quadrant-contiguous runs.
    /// Kept across frames: particles move little per step, so last frame's order
    /// is already nearly sorted and the partition stays cache-friendly.
    order: Vec<u32>,
    scratch: Vec<u32>,
    theta_sq: f32,
}

impl BarnesHut {
    pub fn new(theta: f32) -> Self {
        BarnesHut {
            nodes: Vec::new(),
            order: Vec::new(),
            scratch: Vec::new(),
            theta_sq: theta * theta,
        }
    }

    pub fn set_theta(&mut self, theta: f32) {
        self.theta_sq = theta * theta;
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn build(&mut self, px: &[f32], py: &[f32], mass_per: f32, cx: f32, cy: f32, half: f32) {
        let n = px.len();
        if self.order.len() != n {
            self.order.clear();
            self.order.extend(0..n as u32);
        }
        self.scratch.clear();
        self.scratch.resize(n, 0);
        self.nodes.clear();

        if n == 0 {
            return;
        }

        self.nodes.push(Node::EMPTY);
        build_into(
            0,
            &mut self.nodes,
            &mut self.order,
            &mut self.scratch,
            px,
            py,
            mass_per,
            0,
            n,
            cx,
            cy,
            half,
            0,
        );
    }

    /// Gravitational acceleration at `(x, y)`, skipping particle `me`.
    ///
    /// `stack` is caller-owned so a hot loop allocates nothing; each rayon
    /// worker keeps its own.
    #[allow(clippy::too_many_arguments)]
    pub fn accel(
        &self,
        x: f32,
        y: f32,
        me: u32,
        px: &[f32],
        py: &[f32],
        g_mass: f32,
        mass_per: f32,
        max_accel: f32,
        softening_sq: f32,
        stack: &mut Vec<u32>,
    ) -> (f32, f32) {
        let mut ax = 0.0f32;
        let mut ay = 0.0f32;
        if self.nodes.is_empty() {
            return (ax, ay);
        }

        stack.clear();
        stack.push(0);
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id as usize];
            if node.count == 0 {
                continue;
            }

            let dx = node.com_x - x;
            let dy = node.com_y - y;
            let dist_sq = dx * dx + dy * dy;

            if node.first_child == NO_CHILD {
                // Leaf: a handful of particles, sum them directly.
                let s = node.start as usize;
                let e = s + node.count as usize;
                for &j in &self.order[s..e] {
                    if j == me {
                        continue;
                    }
                    let ju = j as usize;
                    let dx = px[ju] - x;
                    let dy = py[ju] - y;
                    let d2 = dx * dx + dy * dy + softening_sq;
                    let inv_d = d2.sqrt().recip();
                    // Per-interaction clamp, preserving the original's cap.
                    let a = (g_mass * mass_per / d2).min(max_accel);
                    ax += a * dx * inv_d;
                    ay += a * dy * inv_d;
                }
            } else if node.size * node.size < self.theta_sq * dist_sq {
                // Far enough away: treat the whole node as one body.
                //
                // The clamp bound is `count * max_accel`, not `max_accel`: the
                // original capped each *pair*, so `count` pairs may legitimately
                // sum to `count` times the cap. Clamping the aggregate to a
                // single pair's limit would understate a distant cluster by its
                // particle count (a 100-body node at 300px: 741 -> 100).
                let d2 = dist_sq + softening_sq;
                let inv_d = d2.sqrt().recip();
                let a = (g_mass * node.mass / d2).min(max_accel * node.count as f32);
                ax += a * dx * inv_d;
                ay += a * dy * inv_d;
            } else {
                let first = node.first_child;
                stack.push(first);
                stack.push(first + 1);
                stack.push(first + 2);
                stack.push(first + 3);
            }
        }
        (ax, ay)
    }
}

#[inline]
fn quadrant(x: f32, y: f32, cx: f32, cy: f32) -> usize {
    (x >= cx) as usize | (((y >= cy) as usize) << 1)
}

#[allow(clippy::too_many_arguments)]
fn build_into(
    node_id: usize,
    nodes: &mut Vec<Node>,
    order: &mut [u32],
    scratch: &mut [u32],
    px: &[f32],
    py: &[f32],
    mass_per: f32,
    start: usize,
    count: usize,
    cx: f32,
    cy: f32,
    half: f32,
    depth: u32,
) {
    if count <= LEAF_CAPACITY || depth >= MAX_DEPTH {
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        for &i in &order[start..start + count] {
            sx += px[i as usize];
            sy += py[i as usize];
        }
        let inv = if count > 0 {
            (count as f32).recip()
        } else {
            0.0
        };
        nodes[node_id] = Node {
            com_x: sx * inv,
            com_y: sy * inv,
            mass: mass_per * count as f32,
            size: half * 2.0,
            first_child: NO_CHILD,
            start: start as u32,
            count: count as u32,
        };
        return;
    }

    // Four-way partition of order[start..start+count] by quadrant.
    let mut counts = [0usize; 4];
    for &i in &order[start..start + count] {
        counts[quadrant(px[i as usize], py[i as usize], cx, cy)] += 1;
    }
    let mut offsets = [0usize; 4];
    let mut acc = 0usize;
    for q in 0..4 {
        offsets[q] = acc;
        acc += counts[q];
    }
    let mut cursor = offsets;
    for &i in &order[start..start + count] {
        let q = quadrant(px[i as usize], py[i as usize], cx, cy);
        scratch[start + cursor[q]] = i;
        cursor[q] += 1;
    }
    order[start..start + count].copy_from_slice(&scratch[start..start + count]);

    let first_child = nodes.len() as u32;
    nodes.resize(nodes.len() + 4, Node::EMPTY);

    let qh = half * 0.5;
    for q in 0..4 {
        let child_cx = if q & 1 != 0 { cx + qh } else { cx - qh };
        let child_cy = if q & 2 != 0 { cy + qh } else { cy - qh };
        build_into(
            first_child as usize + q,
            nodes,
            order,
            scratch,
            px,
            py,
            mass_per,
            start + offsets[q],
            counts[q],
            child_cx,
            child_cy,
            qh,
            depth + 1,
        );
    }

    let mut mass = 0.0f32;
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    for q in 0..4 {
        let c = &nodes[first_child as usize + q];
        mass += c.mass;
        sx += c.com_x * c.mass;
        sy += c.com_y * c.mass;
    }
    let inv = if mass > 0.0 { 1.0 / mass } else { 0.0 };
    nodes[node_id] = Node {
        com_x: sx * inv,
        com_y: sy * inv,
        mass,
        size: half * 2.0,
        first_child,
        start: start as u32,
        count: count as u32,
    };
}
