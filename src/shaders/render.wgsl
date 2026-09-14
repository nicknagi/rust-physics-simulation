// Instanced particle rendering. Six vertices per particle, generated in the
// shader -- there is no vertex buffer and no per-frame geometry upload. The
// position buffer is the one the compute passes already wrote, so particle data
// never leaves the GPU.

struct RenderParams {
    width: f32,
    height: f32,
    radius: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> colors: array<vec4<f32>>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    let uv = corners[vi];
    let world = positions[ii] + uv * R.radius;

    var out: VsOut;
    out.clip = vec4<f32>(
        world.x / R.width * 2.0 - 1.0,
        1.0 - world.y / R.height * 2.0,
        0.0,
        1.0,
    );
    out.uv = uv;
    out.color = colors[ii];
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Carve a disc out of the quad, with a little edge smoothing.
    let d = dot(in.uv, in.uv);
    if (d > 1.0) { discard; }
    let alpha = 1.0 - smoothstep(0.75, 1.0, d);
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
