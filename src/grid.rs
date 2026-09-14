//! Uniform spatial-hash grid used as the collision broad phase.
//!
//! Every particle shares one radius, so a grid with cells exactly one diameter
//! wide is the ideal accelerator: a particle can only touch particles in its
//! own cell or the eight around it. Cells are stored CSR-style (a start offset
//! per cell plus one flat index array) and filled with a counting sort, so a
//! rebuild is O(n) with no per-cell allocation.

/// Neighbour cells scanned for each cell, chosen so every unordered pair is
/// visited exactly once. Each entry is `(dx, dy)`.
///
/// Because the offsets only ever reach row `+1`, a row's work touches rows
/// `y` and `y + 1` and nothing else. That is what makes the even/odd row
/// split in `sim` safe to run in parallel.
pub const NEIGHBOURS: [(isize, isize); 4] = [(1, 0), (-1, 1), (0, 1), (1, 1)];

pub struct Grid {
    cell: f32,
    inv_cell: f32,
    cols: usize,
    rows: usize,
    starts: Vec<u32>,
    items: Vec<u32>,
    cursor: Vec<u32>,
}

impl Grid {
    pub fn new(cell: f32, width: f32, height: f32) -> Self {
        let cell = cell.max(1e-3);
        let cols = ((width / cell).ceil() as usize).max(1);
        let rows = ((height / cell).ceil() as usize).max(1);
        Grid {
            cell,
            inv_cell: 1.0 / cell,
            cols,
            rows,
            starts: vec![0; cols * rows + 1],
            items: Vec::new(),
            cursor: vec![0; cols * rows],
        }
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cell_size(&self) -> f32 {
        self.cell
    }

    #[inline]
    pub fn cell_index(&self, x: f32, y: f32) -> usize {
        let cx = (x * self.inv_cell) as isize;
        let cy = (y * self.inv_cell) as isize;
        let cx = cx.clamp(0, self.cols as isize - 1) as usize;
        let cy = cy.clamp(0, self.rows as isize - 1) as usize;
        cy * self.cols + cx
    }

    /// Particle indices sitting in cell `c`.
    #[inline]
    pub fn cell_items(&self, c: usize) -> &[u32] {
        let s = self.starts[c] as usize;
        let e = self.starts[c + 1] as usize;
        &self.items[s..e]
    }

    /// Same, addressed by column/row. Returns empty when out of bounds.
    #[inline]
    pub fn cell_items_at(&self, cx: isize, cy: isize) -> &[u32] {
        if cx < 0 || cy < 0 || cx >= self.cols as isize || cy >= self.rows as isize {
            return &[];
        }
        self.cell_items(cy as usize * self.cols + cx as usize)
    }

    /// Counting-sort every particle into its cell. O(n), reuses all buffers.
    pub fn rebuild(&mut self, px: &[f32], py: &[f32]) {
        let n = px.len();
        let ncells = self.cols * self.rows;

        self.items.clear();
        self.items.resize(n, 0);
        for s in self.starts.iter_mut() {
            *s = 0;
        }

        // Histogram, offset by one so the prefix sum lands directly in `starts`.
        for i in 0..n {
            let c = self.cell_index(px[i], py[i]);
            self.starts[c + 1] += 1;
        }
        for c in 0..ncells {
            self.starts[c + 1] += self.starts[c];
        }

        self.cursor.clear();
        self.cursor.extend_from_slice(&self.starts[..ncells]);
        for i in 0..n {
            let c = self.cell_index(px[i], py[i]);
            let slot = self.cursor[c] as usize;
            self.items[slot] = i as u32;
            self.cursor[c] += 1;
        }
    }
}
