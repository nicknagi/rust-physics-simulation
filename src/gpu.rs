//! GPU backend (wgpu -> Metal on this machine).
//!
//! Particle state lives in GPU buffers for the whole run: the compute passes
//! write the position buffer and the render pipeline reads that same buffer as
//! an instance source, so nothing crosses the bus per frame. The CPU only
//! uploads once at startup and reads back on demand for validation.

use wgpu::util::DeviceExt;

use crate::sim::Config;

/// Particles per workgroup. Must match `WG` in physics.wgsl.
const WORKGROUP: u32 = 256;
/// Maximum particles tracked per grid cell. Cells are one diameter wide, so at
/// the packing fractions this runs at the mean occupancy is ~0.23 particles per
/// cell; 8 is a very wide margin. Overflow beyond this is dropped by `bin`.
const BIN_CAPACITY: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
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
    central_gm: f32,

    well_gm: f32,
    well_x: f32,
    well_y: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderParams {
    width: f32,
    height: f32,
    radius: f32,
    well_gm: f32,
    well_x: f32,
    well_y: f32,
    _pad0: f32,
    _pad1: f32,
}

pub struct GpuContext {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub instance: wgpu::Instance,
}

impl GpuContext {
    pub fn new() -> Result<Self, String> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| format!("no suitable GPU adapter: {e}"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("simulation device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| format!("failed to create device: {e}"))?;

        Ok(GpuContext {
            adapter,
            device,
            queue,
            instance,
        })
    }

    pub fn describe(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type)
    }
}

pub struct GpuSim {
    pub cfg: Config,
    count: u32,
    cols: u32,
    rows: u32,

    params: wgpu::Buffer,
    pos: [wgpu::Buffer; 2],
    vel: [wgpu::Buffer; 2],
    colors: wgpu::Buffer,
    readback: wgpu::Buffer,

    bg_common: wgpu::BindGroup,
    /// pos0/vel0 -> pos1/vel1
    bg_forward: wgpu::BindGroup,
    /// pos1/vel1 -> pos0/vel0
    bg_back: wgpu::BindGroup,

    pipe_clear: wgpu::ComputePipeline,
    pipe_integrate: wgpu::ComputePipeline,
    pipe_bin: wgpu::ComputePipeline,
    pipe_collide: wgpu::ComputePipeline,

    /// Cursor well, in domain space. `gm` of 0 disables it.
    well: (f32, f32, f32),
}

impl GpuSim {
    pub fn new(
        device: &wgpu::Device,
        cfg: Config,
        px: &[f32],
        py: &[f32],
        vx: &[f32],
        vy: &[f32],
        colors: &[[f32; 4]],
    ) -> Self {
        let count = px.len() as u32;
        let cell_size = cfg.radius * 2.0;
        let cols = ((cfg.width / cell_size).ceil() as u32).max(1);
        let rows = ((cfg.height / cell_size).ceil() as u32).max(1);

        let interleave = |a: &[f32], b: &[f32]| -> Vec<[f32; 2]> {
            a.iter().zip(b).map(|(&x, &y)| [x, y]).collect()
        };
        let pos_data = interleave(px, py);
        let vel_data = interleave(vx, vy);

        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let mk = |label: &str, data: &[[f32; 2]]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: storage,
            })
        };

        let pos = [mk("pos0", &pos_data), mk("pos1", &pos_data)];
        let vel = [mk("vel0", &vel_data), mk("vel1", &vel_data)];
        let color_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("colors"),
            contents: bytemuck::cast_slice(colors),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let ncells = (cols * rows) as u64;
        let bin_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bin counts"),
            size: (ncells * 4).max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bin_items = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bin items"),
            size: (ncells * BIN_CAPACITY as u64 * 4).max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (count.max(1) as u64) * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl_common = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("common"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, false),
                storage_entry(2, false),
            ],
        });

        let bgl_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("state"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });

        let bg_common = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("common"),
            layout: &bgl_common,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bin_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bin_items.as_entire_binding(),
                },
            ],
        });

        let state_group = |label: &str, a: usize, b: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl_state,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pos[a].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vel[a].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pos[b].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: vel[b].as_entire_binding(),
                    },
                ],
            })
        };
        let bg_forward = state_group("forward", 0, 1);
        let bg_back = state_group("back", 1, 0);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("physics"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/physics.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("physics layout"),
            bind_group_layouts: &[Some(&bgl_common), Some(&bgl_state)],
            immediate_size: 0,
        });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        GpuSim {
            cfg,
            count,
            cols,
            rows,
            params,
            pos,
            vel,
            colors: color_buf,
            readback,
            bg_common,
            bg_forward,
            bg_back,
            pipe_clear: pipe("clear_bins"),
            pipe_integrate: pipe("integrate"),
            pipe_bin: pipe("bin"),
            pipe_collide: pipe("collide"),
            well: (0.0, 0.0, 0.0),
        }
    }

    pub fn len(&self) -> usize {
        self.count as usize
    }

    /// Point the cursor well at `(x, y)` in **domain** space. `gm` is positive
    /// to attract, negative to repel, zero to switch it off. Same
    /// `gm / d^2 * d_hat` form as the central attractor, and likewise unclamped
    /// -- `max_speed` already bounds any fling.
    pub fn set_well(&mut self, gm: f32, x: f32, y: f32) {
        self.well = (gm, x, y);
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The buffer holding current positions -- also the render instance source.
    pub fn position_buffer(&self) -> &wgpu::Buffer {
        &self.pos[0]
    }

    pub fn color_buffer(&self) -> &wgpu::Buffer {
        &self.colors
    }

    fn write_params(&self, queue: &wgpu::Queue, dt: f32) {
        let p = Params {
            width: self.cfg.width,
            height: self.cfg.height,
            radius: self.cfg.radius,
            dt,
            count: self.count,
            cols: self.cols,
            rows: self.rows,
            bin_capacity: BIN_CAPACITY,
            max_speed: self.cfg.max_speed,
            restitution: self.cfg.restitution,
            gm: self.cfg.g * self.cfg.mass,
            max_accel: self.cfg.max_accel,
            softening_sq: self.cfg.softening * self.cfg.softening,
            gravity: self.cfg.gravity as u32,
            cell_size: self.cfg.radius * 2.0,
            central_gm: self.cfg.central_gm,
            well_gm: self.well.0,
            well_x: self.well.1,
            well_y: self.well.2,
            _pad: 0.0,
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
    }

    /// Encode one physics step. Dispatches inside a single compute pass are
    /// ordered and memory-visible to each other per the WebGPU spec, so the four
    /// stages need no explicit barriers.
    pub fn encode_step(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.count == 0 {
            return;
        }
        let particle_groups = self.count.div_ceil(WORKGROUP);
        let cell_groups = (self.cols * self.rows).div_ceil(WORKGROUP);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("physics step"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.bg_common, &[]);
        pass.set_bind_group(1, &self.bg_forward, &[]);

        pass.set_pipeline(&self.pipe_clear);
        pass.dispatch_workgroups(cell_groups, 1, 1);

        pass.set_pipeline(&self.pipe_integrate);
        pass.dispatch_workgroups(particle_groups, 1, 1);

        pass.set_pipeline(&self.pipe_bin);
        pass.dispatch_workgroups(particle_groups, 1, 1);

        pass.set_bind_group(1, &self.bg_back, &[]);
        pass.set_pipeline(&self.pipe_collide);
        pass.dispatch_workgroups(particle_groups, 1, 1);
    }

    pub fn step(&self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        self.write_params(queue, dt);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step"),
        });
        self.encode_step(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }

    /// Run `steps` steps back to back, then block until the GPU has finished.
    ///
    /// Long runs are sliced with a hard fence between slices. Submit-and-poll
    /// alone is not enough: measured here, roughly 2.2 seconds of uninterrupted
    /// GPU compute loses the device on macOS no matter how the work is split
    /// across command buffers, and the next buffer map then fails. The windowed
    /// app never reaches this (at most 8 substeps per frame); it only shows up
    /// when a benchmark asks for thousands of steps in one call.
    pub fn run_steps(&self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32, steps: usize) {
        const STEPS_PER_SUBMIT: usize = 32;
        const STEPS_PER_FENCE: usize = 256;

        self.write_params(queue, dt);
        let mut remaining = steps;
        let mut since_fence = 0usize;
        while remaining > 0 {
            let batch = remaining.min(STEPS_PER_SUBMIT);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("steps"),
            });
            for _ in 0..batch {
                self.encode_step(&mut encoder);
            }
            let index = queue.submit(Some(encoder.finish()));
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(index),
                    timeout: None,
                })
                .ok();
            remaining -= batch;
            since_fence += batch;
            if since_fence >= STEPS_PER_FENCE && remaining > 0 {
                self.fence(device, queue);
                since_fence = 0;
            }
        }
        self.fence(device, queue);
    }

    /// Hard CPU/GPU sync: round-trip a few bytes through a buffer map, which
    /// actually drains the queue rather than merely signalling it.
    fn fence(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.count == 0 {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fence"),
        });
        encoder.copy_buffer_to_buffer(&self.pos[0], 0, &self.readback, 0, 8);
        queue.submit(Some(encoder.finish()));

        let slice = self.readback.slice(..8);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let _ = rx.recv();
        self.readback.unmap();
    }

    pub fn read_positions(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<[f32; 2]> {
        self.read_vec2(device, queue, &self.pos[0])
    }

    pub fn read_velocities(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<[f32; 2]> {
        self.read_vec2(device, queue, &self.vel[0])
    }

    fn read_vec2(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &wgpu::Buffer,
    ) -> Vec<[f32; 2]> {
        if self.count == 0 {
            return Vec::new();
        }
        let bytes = self.count as u64 * 8;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
        encoder.copy_buffer_to_buffer(src, 0, &self.readback, 0, bytes);
        queue.submit(Some(encoder.finish()));

        let slice = self.readback.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().expect("map channel closed").expect("map failed");

        let data = slice
            .get_mapped_range()
            .expect("failed to map readback buffer");
        let out: Vec<[f32; 2]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.readback.unmap();
        out
    }
}

/// Instanced particle renderer. Draws straight from the simulation's position
/// buffer, so no particle data crosses the bus per frame.
pub struct GpuRenderer {
    pipeline: wgpu::RenderPipeline,
    halo_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    state: RenderParams,
}

impl GpuRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, sim: &GpuSim) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
        });

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render params"),
            size: std::mem::size_of::<RenderParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let read_storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                read_storage(1),
                read_storage(2),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sim.position_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sim.color_buffer().as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let make_pipeline = |label: &str, vs: &str, fs: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let pipeline = make_pipeline("particles", "vs", "fs");
        let halo_pipeline = make_pipeline("well halo", "vs_halo", "fs_halo");

        GpuRenderer {
            pipeline,
            halo_pipeline,
            bind_group,
            params,
            state: RenderParams {
                width: 0.0,
                height: 0.0,
                radius: 1.0,
                well_gm: 0.0,
                well_x: 0.0,
                well_y: 0.0,
                _pad0: 0.0,
                _pad1: 0.0,
            },
        }
    }

    pub fn set_viewport(&mut self, queue: &wgpu::Queue, width: f32, height: f32, radius: f32) {
        self.state.width = width;
        self.state.height = height;
        self.state.radius = radius;
        self.upload(queue);
    }

    /// Cursor well indicator, in domain space. `gm` of 0 hides it.
    pub fn set_well(&mut self, queue: &wgpu::Queue, gm: f32, x: f32, y: f32) {
        self.state.well_gm = gm;
        self.state.well_x = x;
        self.state.well_y = y;
        self.upload(queue);
    }

    fn upload(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&self.state));
    }

    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, count: u32) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particles"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.02,
                        b: 0.04,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_pipeline(&self.pipeline);
        // Six vertices per particle, generated in the vertex shader.
        pass.draw(0..6, 0..count);

        if self.state.well_gm != 0.0 {
            pass.set_pipeline(&self.halo_pipeline);
            pass.draw(0..6, 0..1);
        }
    }
}
