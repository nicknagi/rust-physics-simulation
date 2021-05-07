
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


https://user-images.githubusercontent.com/14340000/117506070-feaf1c00-af52-11eb-9a01-975626356f65.mov

https://user-images.githubusercontent.com/14340000/117506148-1ededb00-af53-11eb-9a87-280a40dbf966.mov

https://user-images.githubusercontent.com/14340000/117507623-4c2c8880-af55-11eb-9745-2f5d5d828287.mov
