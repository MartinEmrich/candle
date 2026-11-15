//! Lazily-built shader modules, compute pipelines, and descriptor sets.
//!
//! Pipelines are keyed by (kernel name, binding count, push-constant size);
//! descriptor sets are recycled through a free list backed by growable pools.

use crate::vulkan_backend::device::device_err;
use crate::Result;
use ash::vk;
use std::collections::HashMap;

struct PipelineEntry {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

pub(crate) struct KernelCache {
    modules: HashMap<&'static str, vk::ShaderModule>,
    pipelines: HashMap<(&'static str, usize, u32), PipelineEntry>,
    set_layouts: HashMap<usize, vk::DescriptorSetLayout>,
    pools: Vec<vk::DescriptorPool>,
    free_sets: Vec<(vk::DescriptorSet, usize)>,
}

const SETS_PER_POOL: u32 = 256;
pub(crate) const MAX_BINDINGS: u32 = 8;

impl KernelCache {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            pipelines: HashMap::new(),
            set_layouts: HashMap::new(),
            pools: Vec::new(),
            free_sets: Vec::new(),
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        for entry in self.pipelines.values() {
            unsafe {
                device.destroy_pipeline(entry.pipeline, None);
                device.destroy_pipeline_layout(entry.layout, None);
            }
        }
        for module in self.modules.values() {
            unsafe { device.destroy_shader_module(*module, None) };
        }
        for layout in self.set_layouts.values() {
            unsafe { device.destroy_descriptor_set_layout(*layout, None) };
        }
        for pool in &self.pools {
            unsafe { device.destroy_descriptor_pool(*pool, None) };
        }
        self.pools.clear();
        self.free_sets.clear();
    }

    fn module(
        &mut self,
        device: &ash::Device,
        name: &'static str,
        spv: &[u8],
    ) -> Result<vk::ShaderModule> {
        if let Some(module) = self.modules.get(name) {
            return Ok(*module);
        }
        let code: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let ci = vk::ShaderModuleCreateInfo::default().code(&code);
        let module = unsafe { device.create_shader_module(&ci, None) }
            .map_err(|e| device_err("create_shader_module", e))?;
        self.modules.insert(name, module);
        Ok(module)
    }

    fn set_layout(
        &mut self,
        device: &ash::Device,
        bindings: usize,
    ) -> Result<vk::DescriptorSetLayout> {
        if let Some(layout) = self.set_layouts.get(&bindings) {
            return Ok(*layout);
        }
        let entries: Vec<vk::DescriptorSetLayoutBinding> = (0..bindings as u32)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&entries);
        let layout = unsafe { device.create_descriptor_set_layout(&ci, None) }
            .map_err(|e| device_err("create_descriptor_set_layout", e))?;
        self.set_layouts.insert(bindings, layout);
        Ok(layout)
    }

    pub(super) fn pipeline(
        &mut self,
        device: &ash::Device,
        name: &'static str,
        spv: &[u8],
        bindings: usize,
        push_size: u32,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout)> {
        let key = (name, bindings, push_size);
        if let Some(entry) = self.pipelines.get(&key) {
            return Ok((entry.pipeline, entry.layout));
        }
        let module = self.module(device, name, spv)?;
        let set_layout = self.set_layout(device, bindings)?;

        let pc_range = vk::PushConstantRange::default()
            .offset(0)
            .size(push_size)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&set_layout))
            .push_constant_ranges(std::slice::from_ref(&pc_range));
        let layout = unsafe { device.create_pipeline_layout(&layout_ci, None) }
            .map_err(|e| device_err("create_pipeline_layout", e))?;

        let entry_point = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry_point);
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        let pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&ci), None)
        }
        .map_err(|e| device_err("create_compute_pipelines", e))?[0];

        self.pipelines.insert(key, PipelineEntry { pipeline, layout });
        Ok((pipeline, layout))
    }

    pub(super)     fn alloc_set(&mut self, device: &ash::Device, bindings: usize) -> Result<vk::DescriptorSet> {
        if let Some(i) = self
            .free_sets
            .iter()
            .position(|(_, n)| *n == bindings)
        {
            let (set, _) = self.free_sets.remove(i);
            return Ok(set);
        }
        let set_layout = self.set_layout(device, bindings)?;
        // Try existing pools, creating new ones on exhaustion.
        loop {
            let pool = match self.pools.last() {
                Some(pool) => *pool,
                None => {
                    let size = vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(SETS_PER_POOL * MAX_BINDINGS);
                    let ci = vk::DescriptorPoolCreateInfo::default()
                        .max_sets(SETS_PER_POOL)
                        .pool_sizes(std::slice::from_ref(&size));
                    let pool = unsafe { device.create_descriptor_pool(&ci, None) }
                        .map_err(|e| device_err("create_descriptor_pool", e))?;
                    self.pools.push(pool);
                    pool
                }
            };
            let ci = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(std::slice::from_ref(&set_layout));
            match unsafe { device.allocate_descriptor_sets(&ci) } {
                Ok(sets) => return Ok(sets[0]),
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)
                | Err(vk::Result::ERROR_FRAGMENTED_POOL)
                | Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => {
                    // Grow: next iteration allocates a fresh pool.
                    let size = vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(SETS_PER_POOL * MAX_BINDINGS);
                    let ci = vk::DescriptorPoolCreateInfo::default()
                        .max_sets(SETS_PER_POOL)
                        .pool_sizes(std::slice::from_ref(&size));
                    let pool = unsafe { device.create_descriptor_pool(&ci, None) }
                        .map_err(|e| device_err("create_descriptor_pool", e))?;
                    self.pools.push(pool);
                }
                Err(e) => return Err(device_err("allocate_descriptor_sets", e)),
            }
        }
    }

    pub(super) fn free_set(&mut self, set: vk::DescriptorSet, bindings: usize) {
        self.free_sets.push((set, bindings));
    }
}

/// Compute the 3D dispatch grid (workgroup counts) for `numel` work items.
pub(crate) fn dispatch_groups(numel: usize, workgroup_size: u32) -> (u32, u32, u32) {
    let n = numel.div_ceil(workgroup_size as usize).max(1) as u64;
    let x = n.min(65535) as u32;
    let y = n.div_ceil(x as u64).min(65535) as u32;
    let z = n.div_ceil(x as u64 * y as u64).min(65535) as u32;
    (x, y, z)
}

/// Serialize u32 words into push-constant bytes.
pub(crate) fn push_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}
