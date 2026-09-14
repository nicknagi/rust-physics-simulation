//! A 2D particle physics simulation.
//!
//! The core is renderer-agnostic so it can be driven by the windowed binary,
//! the headless benchmark, or tests. State is stored structure-of-arrays in
//! `f32` so the hot loops stay cache-dense and auto-vectorize.

pub mod bh;
pub mod grid;
pub mod render;
pub mod sim;
pub mod vec2;

pub use sim::{Config, Sim};
pub use vec2::{v2, Vec2};
