
# Rust Physics Simulation

A simple physics simulation built with Rust. Goal of the project was to learn basic rust while doing something fun.

## How To Run

Simply download the binary from the releases tab and run!

If you would rather compile yourself, then run ```cargo build --release``` from the project root.

## Implementation

Implementation is not the most efficient but works well up to 10000 particles.

Collision detection: a simple collision detection algorithm is used to find and resolve collisions. Approach similar to sweep and prune is used for speed.

Forces: Arbitrary forces can be applied and resolved on each particles, feel free to edit the code to experiment. The current implementation includes gravitational attraction as an example. A constant force such as gravity (g) can also be applied by modifying line 125.

## Demo


https://user-images.githubusercontent.com/14340000/117508795-27d1ab80-af57-11eb-9566-2863e8106ed3.mov


https://user-images.githubusercontent.com/14340000/117508803-2b653280-af57-11eb-9474-56d47e0c2ee9.mov


https://user-images.githubusercontent.com/14340000/117508807-2d2ef600-af57-11eb-890d-2ac97d90f9f7.mov


