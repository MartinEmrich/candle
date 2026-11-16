use crate::vulkan_backend::cache::{push_bytes, KernelCache, MAX_BINDINGS};
use crate::{Error, Result, VulkanError};
use ash::vk;
use std::sync::{Arc, Mutex, OnceLock};

static INSTANCE: OnceLock<Arc<VulkanInstance>> = OnceLock::new();

struct VulkanInstance {
    _entry: ash::Entry,
    instance: ash::Instance,
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

fn instance() -> Result<Arc<VulkanInstance>> {
    if let Some(inst) = INSTANCE.get() {
        return Ok(inst.clone());
    }
    let entry = unsafe { ash::Entry::load() }.map_err(|e| {
        Error::from(VulkanError::Message(format!(
            "failed to load vulkan loader: {e}"
        )))
    })?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"candle-vulkan")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"candle")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_2);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&create_info, None) }.map_err(|e| {
        Error::from(VulkanError::Message(format!(
            "instance creation failed: {e}"
        )))
    })?;
    let inst = Arc::new(VulkanInstance {
        _entry: entry,
        instance,
    });
    let _ = INSTANCE.set(inst.clone());
    Ok(inst)
}

/// A buffer with its memory allocation. Keeps the device alive.
pub struct VulkanBuffer {
    inner: Arc<VulkanDeviceInner>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
    /// When `Some((block, offset, slot_size))` the memory is a sub-allocation
    /// from the device-memory pool; on drop the slot is returned to the pool
    /// instead of `vkFreeMemory`-ing the whole allocation.
    pooled: Option<(usize, u64, u64)>,
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            let d = &self.inner.device;
            d.destroy_buffer(self.buffer, None);
            match self.pooled {
                Some((block, offset, slot)) => {
                    self.inner.pool.lock().unwrap().free(block, offset, slot)
                }
                None => d.free_memory(self.memory, None),
            }
        }
    }
}

impl VulkanBuffer {
    pub fn size(&self) -> usize {
        self.size
    }

    pub(crate) fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    #[allow(dead_code)]
    pub(crate) fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }
}

fn align_up(x: u64, a: u64) -> u64 {
    if a <= 1 {
        x
    } else {
        (x + a - 1) / a * a
    }
}

/// A single device-local `VkDeviceMemory` block carved up by [`MemoryPool`].
struct MemBlock {
    memory: vk::DeviceMemory,
    _size: u64,
}

/// Sub-allocating device-local memory pool. Buffers up to `block_size` are
/// carved out of large `vkAllocateMemory` blocks with a first-fit free list
/// (coalescing on free), so the hot path avoids `vkAllocateMemory`/
/// `vkFreeMemory` entirely. Larger buffers and host-visible staging fall back
/// to dedicated allocations.
struct MemoryPool {
    blocks: Vec<MemBlock>,
    /// Free slots `(block, offset, size)`, kept sorted by `(block, offset)`.
    free: Vec<(usize, u64, u64)>,
    bump_block: usize,
    bump_offset: u64,
    block_size: u64,
    memory_type_index: u32,
    enabled: bool,
}

impl MemoryPool {
    fn new(memory_type_index: u32) -> Self {
        Self {
            blocks: Vec::new(),
            free: Vec::new(),
            bump_block: 0,
            bump_offset: 0,
            block_size: 64 * 1024 * 1024,
            memory_type_index,
            enabled: memory_type_index != u32::MAX,
        }
    }

    /// Reserve `size` bytes with `alignment`; returns `(block, memory, offset)`.
    /// `None` when the request is too large or the pool is disabled (caller
    /// falls back to a dedicated allocation).
    fn allocate(
        &mut self,
        device: &ash::Device,
        size: u64,
        alignment: u64,
    ) -> Option<(usize, vk::DeviceMemory, u64)> {
        if !self.enabled || size > self.block_size {
            return None;
        }
        let align = alignment.max(1);
        // First-fit scan of the free list.
        for i in 0..self.free.len() {
            let (blk, off, fsize) = self.free[i];
            let aoff = align_up(off, align);
            if aoff - off + size <= fsize {
                self.free.remove(i);
                if aoff > off {
                    self.insert_free(blk, off, aoff - off);
                }
                let end = aoff + size;
                if off + fsize > end {
                    self.insert_free(blk, end, off + fsize - end);
                }
                return Some((blk, self.blocks[blk].memory, aoff));
            }
        }
        // Bump-allocate from the active block, growing on demand.
        let mut off = if self.blocks.is_empty() {
            0
        } else {
            align_up(self.bump_offset, align)
        };
        if self.blocks.is_empty() || off + size > self.block_size {
            let ci = vk::MemoryAllocateInfo::default()
                .allocation_size(self.block_size)
                .memory_type_index(self.memory_type_index);
            let memory = unsafe { device.allocate_memory(&ci, None) }.ok()?;
            self.blocks.push(MemBlock {
                memory,
                _size: self.block_size,
            });
            self.bump_block = self.blocks.len() - 1;
            self.bump_offset = 0;
            off = 0;
        }
        self.bump_offset = off + size;
        Some((self.bump_block, self.blocks[self.bump_block].memory, off))
    }

    fn free(&mut self, block: usize, offset: u64, size: u64) {
        self.insert_free(block, offset, size);
    }

    fn insert_free(&mut self, blk: usize, off: u64, size: u64) {
        if size == 0 {
            return;
        }
        let pos = self
            .free
            .iter()
            .position(|(b, o, _)| (*b, *o) > (blk, off))
            .unwrap_or(self.free.len());
        self.free.insert(pos, (blk, off, size));
        // Coalesce with the next slot.
        if pos + 1 < self.free.len() {
            let (b1, o1, s1) = self.free[pos];
            let (b2, o2, s2) = self.free[pos + 1];
            if b1 == b2 && o1 + s1 == o2 {
                self.free[pos] = (b1, o1, s1 + s2);
                self.free.remove(pos + 1);
            }
        }
        // Coalesce with the previous slot.
        if pos > 0 {
            let (b0, o0, s0) = self.free[pos - 1];
            let (b1, o1, s1) = self.free[pos];
            if b0 == b1 && o0 + s0 == o1 {
                self.free[pos - 1] = (b0, o0, s0 + s1);
                self.free.remove(pos);
            }
        }
    }

    fn destroy(&mut self, device: &ash::Device) {
        for block in self.blocks.drain(..) {
            unsafe { device.free_memory(block.memory, None) };
        }
        self.free.clear();
    }
}

pub struct VulkanDeviceInner {
    #[allow(dead_code)]
    instance: Arc<VulkanInstance>,
    #[allow(dead_code)]
    physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family: u32,
    pub queue: vk::Queue,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    properties: vk::PhysicalDeviceProperties,
    command_pool: vk::CommandPool,
    /// Deferred command recording: ops accumulate in `cmd` until a batch is
    /// submitted (auto-flush threshold, explicit flush, or readback).
    pending: Mutex<Pending>,
    cache: Mutex<KernelCache>,
    pool: Mutex<MemoryPool>,
    /// Byte budget for a deferred batch: once the buffers referenced by the
    /// unsubmitted batch exceed this, the batch is flushed (submitted) so their
    /// memory can be reclaimed once the GPU finishes. Bounds peak VRAM so a
    /// large forward pass does not keep every intermediate alive at once.
    flush_bytes: usize,
    ordinal: usize,
    name: String,
}

/// Max recorded ops per command buffer batch before an auto-flush.
const FLUSH_OPS_THRESHOLD: usize = 512;

/// Default deferred-batch byte budget, used when the device heap size is
/// unknown. Overridable via `CANDLE_VULKAN_FLUSH_BYTES`.
const DEFAULT_FLUSH_BYTES: usize = 1024 * 1024 * 1024;

#[derive(Default)]
struct Pending {
    cmd: Option<vk::CommandBuffer>,
    /// Buffers referenced by ops recorded into `cmd` (not yet submitted).
    keepalive: Vec<Arc<VulkanBuffer>>,
    /// Total size (bytes) of `keepalive`.
    pending_bytes: usize,
    /// Descriptor sets used by ops in `cmd`; returned to the cache only after
    /// the batch's fence signals.
    sets: Vec<(vk::DescriptorSet, usize)>,
    ops: usize,
    /// Submitted-but-unfinished batches: (fence, cmd, keepalive, sets).
    /// The command buffer and buffers stay alive until the fence signals.
    inflight: Vec<(vk::Fence, vk::CommandBuffer, Vec<Arc<VulkanBuffer>>, Vec<(vk::DescriptorSet, usize)>)>,
    /// Total size (bytes) of every inflight batch's keepalive buffers.
    inflight_bytes: usize,
}

impl std::fmt::Debug for VulkanDeviceInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VulkanDevice({}, {})", self.name, self.ordinal)
    }
}

impl Drop for VulkanDeviceInner {
    fn drop(&mut self) {
        unsafe {
            let d = &self.device;
            if let Ok(mut p) = self.pending.lock() {
                for (fence, cmd, _, _) in p.inflight.drain(..) {
                    let _ = d.wait_for_fences(&[fence], true, u64::MAX);
                    d.destroy_fence(fence, None);
                    d.free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));
                }
                if let Some(cmd) = p.cmd.take() {
                    d.free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));
                }
            }
            let _ = d.device_wait_idle();
            if let Ok(mut cache) = self.cache.lock() {
                cache.destroy(d);
            }
            if let Ok(mut pool) = self.pool.lock() {
                pool.destroy(d);
            }
            d.destroy_command_pool(self.command_pool, None);
            d.destroy_device(None);
        }
    }
}

pub(crate) fn device_err(op: &str, e: impl std::fmt::Debug) -> Error {
    Error::from(VulkanError::Message(format!("vulkan {op} failed: {e:?}")))
}

impl VulkanDeviceInner {
    pub fn new(ordinal: usize) -> Result<Self> {
        let instance = instance()?;
        let physical_devices = unsafe { instance.instance.enumerate_physical_devices() }
            .map_err(|e| device_err("enumerate_physical_devices", e))?;
        if physical_devices.is_empty() {
            return Err(Error::from(VulkanError::Message(
                "no vulkan physical devices found".into(),
            )));
        }
        // Only consider devices that expose a compute+transfer queue.
        let mut candidates = vec![];
        for (idx, &pd) in physical_devices.iter().enumerate() {
            let families =
                unsafe { instance.instance.get_physical_device_queue_family_properties(pd) };
            if let Some(family) = families.iter().position(|f| {
                f.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && f.queue_flags.contains(vk::QueueFlags::TRANSFER)
            }) {
                candidates.push((idx, pd, family as u32));
            }
        }
        if candidates.is_empty() {
            return Err(Error::from(VulkanError::Message(
                "no vulkan device with a compute queue found".into(),
            )));
        }
        if ordinal >= candidates.len() {
            return Err(Error::from(VulkanError::Message(format!(
                "vulkan device ordinal {ordinal} out of range ({} compute devices)",
                candidates.len()
            ))));
        }
        let (_, physical_device, queue_family) = candidates[ordinal];
        let props = unsafe { instance.instance.get_physical_device_properties(physical_device) };
        let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let memory_properties =
            unsafe { instance.instance.get_physical_device_memory_properties(physical_device) };

        let priority = [1.0f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priority);
        let features = vk::PhysicalDeviceFeatures::default();
        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_ci))
            .enabled_features(&features);
        let device = unsafe {
            instance
                .instance
                .create_device(physical_device, &device_ci, None)
        }
        .map_err(|e| device_err("create_device", e))?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_ci, None) }
            .map_err(|e| device_err("create_command_pool", e))?;

        // Device-local memory type index for the sub-allocation pool; disable
        // the pool (u32::MAX) if none is available.
        let device_local_index = {
            let mut idx = u32::MAX;
            for (i, ty) in memory_properties.memory_types_as_slice().iter().enumerate() {
                if ty
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                {
                    idx = i as u32;
                    break;
                }
            }
            idx
        };

        // Deferred-batch byte budget: a fraction of the device-local heap, so a
        // large forward pass flushes often enough to bound peak VRAM. Env
        // override (bytes) wins; otherwise clamp heap/4 into [64MB, 1GB].
        let flush_bytes = {
            if let Some(v) = std::env::var("CANDLE_VULKAN_FLUSH_BYTES")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
            {
                v.max(1024 * 1024)
            } else {
                let heap = if device_local_index != u32::MAX {
                    let ty = &memory_properties.memory_types_as_slice()[device_local_index as usize];
                    memory_properties.memory_heaps_as_slice()[ty.heap_index as usize].size as usize
                } else {
                    0
                };
                if heap == 0 {
                    DEFAULT_FLUSH_BYTES
                } else {
                    (heap / 4).clamp(64 * 1024 * 1024, DEFAULT_FLUSH_BYTES)
                }
            }
        };

        Ok(Self {
            instance,
            physical_device,
            device,
            queue_family,
            queue,
            memory_properties,
            properties: props,
            command_pool,
            pending: Mutex::new(Pending::default()),
            cache: Mutex::new(KernelCache::new()),
            pool: Mutex::new(MemoryPool::new(device_local_index)),
            flush_bytes,
            ordinal,
            name,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.properties
    }

    fn find_memory_type(&self, filter: u32, props: vk::MemoryPropertyFlags) -> Result<u32> {
        for (i, ty) in self.memory_properties.memory_types_as_slice().iter().enumerate() {
            if filter & (1 << i) != 0 && ty.property_flags.contains(props) {
                return Ok(i as u32);
            }
        }
        Err(Error::from(VulkanError::Message(format!(
            "no suitable memory type for {props:?}",
            props = props.as_raw()
        ))))
    }

    fn alloc_buffer(
        self: &Arc<Self>,
        size: usize,
        usage: vk::BufferUsageFlags,
        props: vk::MemoryPropertyFlags,
        poolable: bool,
    ) -> Result<Arc<VulkanBuffer>> {
        if size == 0 {
            return Err(Error::from(VulkanError::Message(
                "cannot allocate a zero-sized buffer".into(),
            )));
        }
        let buffer_ci = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&buffer_ci, None) }
            .map_err(|e| device_err("create_buffer", e))?;
        let reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let granularity = self.properties.limits.buffer_image_granularity;
        if poolable {
            let alignment = reqs.alignment.max(granularity).max(1);
            let slot = {
                let mut pool = self.pool.lock().unwrap();
                pool.allocate(&self.device, reqs.size, alignment)
            };
            if let Some((block, memory, offset)) = slot {
                unsafe { self.device.bind_buffer_memory(buffer, memory, offset) }
                    .map_err(|e| device_err("bind_buffer_memory", e))?;
                let slot_size = reqs.size;
                return Ok(Arc::new(VulkanBuffer {
                    inner: self.clone(),
                    buffer,
                    memory,
                    size,
                    pooled: Some((block, offset, slot_size)),
                }));
            }
        }
        let type_index = self.find_memory_type(reqs.memory_type_bits, props)?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None) }
            .map_err(|e| device_err("allocate_memory", e))?;
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| device_err("bind_buffer_memory", e))?;
        Ok(Arc::new(VulkanBuffer {
            inner: self.clone(),
            buffer,
            memory,
            size,
            pooled: None,
        }))
    }

    pub fn alloc_device(self: &Arc<Self>, size: usize) -> Result<Arc<VulkanBuffer>> {
        self.alloc_buffer(
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )
    }

    pub fn alloc_host_visible(self: &Arc<Self>, size: usize) -> Result<Arc<VulkanBuffer>> {
        self.alloc_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            false,
        )
    }

    /// Record a command into the current deferred batch. `keepalive` buffers
    /// (and `sets`) stay alive until the batch's fence signals, so in-flight
    /// GPU work can never observe a destroyed buffer.
    fn record(
        &self,
        keepalive: Vec<Arc<VulkanBuffer>>,
        sets: Vec<(vk::DescriptorSet, usize)>,
        f: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<()> {
        let mut p = self.pending.lock().unwrap();
        if p.cmd.is_none() {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = unsafe { self.device.allocate_command_buffers(&alloc_info) }
                .map_err(|e| device_err("allocate_command_buffers", e))?[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { self.device.begin_command_buffer(cmd, &begin) }
                .map_err(|e| device_err("begin_command_buffer", e))?;
            p.cmd = Some(cmd);
        }
        let cmd = p.cmd.unwrap();
        f(&self.device, cmd);
        // Memory barrier: ops in a batch share a command buffer but touch
        // overlapping buffers; without this the driver may reorder accesses.
        // Stages are limited to COMPUTE|TRANSFER (the only stages we use) so
        // RADV skips flushing unrelated pipeline caches.
        let bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE,
            );
        let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                stages,
                stages,
                vk::DependencyFlags::empty(),
                &[bar],
                &[],
                &[],
            );
        }
        let add_bytes: usize = keepalive.iter().map(|b| b.size()).sum();
        p.keepalive.extend(keepalive);
        p.pending_bytes += add_bytes;
        p.sets.extend(sets);
        p.ops += 1;
        // Flush when the batch grows large in ops or bytes. If already-submitted
        // work still holds a lot of memory, wait for it so total live VRAM stays
        // bounded even when the GPU lags behind command recording.
        if p.ops >= FLUSH_OPS_THRESHOLD || p.pending_bytes >= self.flush_bytes {
            let wait = p.inflight_bytes >= self.flush_bytes;
            self.flush_locked(&mut p, wait)?;
        }
        Ok(())
    }

    /// Submit the current batch (if any) without waiting; then optionally
    /// wait for all inflight batches.
    fn flush_locked(&self, p: &mut Pending, wait_all: bool) -> Result<()> {
        if p.cmd.is_some() {
            let cmd = p.cmd.unwrap();
            unsafe { self.device.end_command_buffer(cmd) }
                .map_err(|e| device_err("end_command_buffer", e))?;
            let fence_ci = vk::FenceCreateInfo::default();
            let fence = unsafe { self.device.create_fence(&fence_ci, None) }
                .map_err(|e| device_err("create_fence", e))?;
            let submit_info =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            let res = unsafe { self.device.queue_submit(self.queue, &[submit_info], fence) };
            if res.is_err() {
                unsafe {
                    self.device.destroy_fence(fence, None);
                    self.device
                        .free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));
                }
                return Err(device_err("queue_submit", res.unwrap_err()));
            }
            let moved_keepalive = std::mem::take(&mut p.keepalive);
            let moved_sets = std::mem::take(&mut p.sets);
            p.inflight_bytes += moved_keepalive.iter().map(|b| b.size()).sum::<usize>();
            p.pending_bytes = 0;
            p.inflight.push((fence, cmd, moved_keepalive, moved_sets));
            p.cmd = None;
            p.ops = 0;
        }
        self.reap(p, wait_all)
    }

    /// Reap completed inflight batches (fences); free their resources.
    fn reap(&self, p: &mut Pending, wait: bool) -> Result<()> {
        let mut i = 0;
        while i < p.inflight.len() {
            let fence = p.inflight[i].0;
            let done = if wait {
                unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX) }
                    .map_err(|e| device_err("wait_for_fences", e))?;
                true
            } else {
                match unsafe { self.device.get_fence_status(fence) } {
                    Ok(true) => true,
                    Ok(false) | Err(vk::Result::NOT_READY) => false,
                    Err(e) => return Err(device_err("get_fence_status", e)),
                }
            };
            if done {
                let (fence, cmd, keepalive, sets) = p.inflight.remove(i);
                p.inflight_bytes = p
                    .inflight_bytes
                    .saturating_sub(keepalive.iter().map(|b| b.size()).sum::<usize>());
                unsafe {
                    self.device.destroy_fence(fence, None);
                    self.device
                        .free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));
                }
                drop(keepalive);
                let mut cache = self.cache.lock().unwrap();
                for (set, n) in sets {
                    cache.free_set(set, n);
                }
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// Submit any recorded batch; if `wait`, also wait for completion.
    pub fn flush(&self, wait: bool) -> Result<()> {
        let mut p = self.pending.lock().unwrap();
        self.flush_locked(&mut p, wait)
    }

    /// Record + submit + wait for a one-shot command (used for readbacks).
    pub fn submit(&self, record: impl FnOnce(&ash::Device, vk::CommandBuffer)) -> Result<()> {
        self.record(vec![], vec![], record)?;
        self.flush(true)
    }

    /// Host to device copy via a staging buffer (deferred).
    pub fn copy_h2d(self: &Arc<Self>, data: &[u8], dst: &Arc<VulkanBuffer>) -> Result<()> {
        if data.len() > dst.size {
            return Err(Error::from(VulkanError::Message(format!(
                "copy_h2d: {} bytes into {} bytes",
                data.len(),
                dst.size
            ))));
        }
        let staging = self.alloc_host_visible(data.len().max(4))?;
        unsafe {
            let ptr = self
                .device
                .map_memory(staging.memory, 0, data.len() as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| device_err("map_memory", e))?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            self.device.unmap_memory(staging.memory);
        }
        let (src_buf, dst_buf) = (staging.buffer, dst.buffer);
        let len = data.len();
        self.record(
            vec![staging.clone(), dst.clone()],
            vec![],
            move |d, cmd| unsafe {
                d.cmd_copy_buffer(
                    cmd,
                    src_buf,
                    dst_buf,
                    &[vk::BufferCopy::default()
                        .src_offset(0)
                        .dst_offset(0)
                        .size(len as u64)],
                );
            },
        )
    }

    /// Device to host copy via a staging buffer (synchronous: flushes first).
    pub fn copy_d2h(self: &Arc<Self>, src: &Arc<VulkanBuffer>, len: usize) -> Result<Vec<u8>> {
        let len = len.min(src.size);
        self.flush(true)?;
        let staging = self.alloc_host_visible(len.max(4))?;
        let (src_buf, dst_buf) = (src.buffer, staging.buffer);
        self.submit(move |d, cmd| unsafe {
            d.cmd_copy_buffer(
                cmd,
                src_buf,
                dst_buf,
                &[vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(len as u64)],
            );
        })?;
        let mut out = vec![0u8; len];
        unsafe {
            let ptr = self
                .device
                .map_memory(staging.memory, 0, len as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| device_err("map_memory", e))?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), len);
            self.device.unmap_memory(staging.memory);
        }
        Ok(out)
    }

    /// vkCmdFillBuffer, 32-bit pattern. `size` must be a multiple of 4.
    pub fn fill_buffer(
        &self,
        buffer: &Arc<VulkanBuffer>,
        offset: usize,
        size: usize,
        pattern: u32,
    ) -> Result<()> {
        debug_assert_eq!(size % 4, 0);
        let buf = buffer.buffer;
        self.record(
            vec![buffer.clone()],
            vec![],
            move |d, cmd| unsafe {
                d.cmd_fill_buffer(cmd, buf, offset as u64, size as u64, pattern);
            },
        )
    }

    pub fn copy_buffer(
        &self,
        src: &Arc<VulkanBuffer>,
        dst: &Arc<VulkanBuffer>,
        size: usize,
    ) -> Result<()> {
        let (src_buf, dst_buf) = (src.buffer, dst.buffer);
        self.record(
            vec![src.clone(), dst.clone()],
            vec![],
            move |d, cmd| unsafe {
                d.cmd_copy_buffer(
                    cmd,
                    src_buf,
                    dst_buf,
                    &[vk::BufferCopy::default()
                        .src_offset(0)
                        .dst_offset(0)
                        .size(size as u64)],
                );
            },
        )
    }

    /// Dispatch a compute kernel over storage-buffer `bindings` with
    /// push constants given as 4-byte words. Recorded into the deferred batch.
    pub fn dispatch(
        &self,
        spv: &[u8],
        name: &'static str,
        bindings: &[&Arc<VulkanBuffer>],
        push: &[u32],
        groups: (u32, u32, u32),
    ) -> Result<()> {
        if bindings.len() > MAX_BINDINGS as usize {
            return Err(Error::from(VulkanError::Message(format!(
                "too many bindings for {name}"
            ))));
        }
        let push_bytes = push_bytes(push);
        let (pipeline, layout, set) = {
            let mut cache = self.cache.lock().unwrap();
            let (pipeline, layout) =
                cache.pipeline(&self.device, name, spv, bindings.len(), push_bytes.len() as u32)?;
            let set = cache.alloc_set(&self.device, bindings.len())?;
            (pipeline, layout, set)
        };

        let infos: Vec<vk::DescriptorBufferInfo> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.handle())
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        let keepalive: Vec<Arc<VulkanBuffer>> = bindings.iter().map(|b| (*b).clone()).collect();
        let sets = vec![(set, bindings.len())];
        self.record(keepalive, sets, move |d, cmd| unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                std::slice::from_ref(&set),
                &[],
            );
            if !push_bytes.is_empty() {
                d.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push_bytes,
                );
            }
            d.cmd_dispatch(cmd, groups.0, groups.1, groups.2);
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.flush(true)
    }
}
