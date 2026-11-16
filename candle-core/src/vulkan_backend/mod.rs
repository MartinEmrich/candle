//! Implementation of Backend traits for Vulkan
//!
//! Targets a Vulkan 1.2 compute-only device. Compute is fp32; fp16 is used
//! as a storage format only (dequantized in-shader), which matches hardware
//! without native fp16 ALUs (e.g. AMD GCN4 / gfx803).

use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::vulkan_backend::device::{device_err, VulkanBuffer};
use crate::{CpuStorage, DType, Layout, Result, Shape, WithDType};
use ash::vk;
use half::f16;
use std::sync::Arc;

mod cache;
mod device;

pub use device::VulkanDeviceInner;
pub(crate) use cache::dispatch_groups;

#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

fn not_yet(op: &str) -> crate::Error {
    crate::Error::msg(format!("vulkan: `{op}` is not yet implemented"))
}

#[derive(Clone)]
pub struct VulkanDevice {
    inner: Arc<VulkanDeviceInner>,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.inner, f)
    }
}

impl VulkanDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(VulkanDeviceInner::new(ordinal)?),
        })
    }

    pub fn inner(&self) -> &Arc<VulkanDeviceInner> {
        &self.inner
    }

    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Allocate an uninitialized device buffer for `len` elements of `dtype`.
    fn alloc(&self, len: usize, dtype: DType) -> Result<Arc<VulkanBuffer>> {
        self.inner.alloc_device(len * dtype.size_in_bytes())
    }

    fn alloc_filled(&self, len: usize, dtype: DType, value: u32) -> Result<Arc<VulkanBuffer>> {
        let size = len * dtype.size_in_bytes();
        let buffer = self.inner.alloc_device(size)?;
        if size % 4 == 0 {
            self.inner.fill_buffer(&buffer, 0, size, value)?;
        } else if value == 0 {
            // Fall back to a host-side zero copy for non 4-byte-multiple sizes.
            let zeros = vec![0u8; size];
            self.inner.copy_h2d(&zeros, &buffer)?;
        } else {
            return Err(crate::Error::msg("vulkan: cannot fill this size"));
        }
        Ok(buffer)
    }

    /// Dispatch the fill compute shader over a u32 view of the buffer.
    fn dispatch_fill(
        &self,
        buffer: &VulkanBuffer,
        numel: u32,
        offset: u32,
        value: u32,
    ) -> Result<()> {
        let inner = &*self.inner;
        let spv_u32: Vec<u32> = candle_vulkan_kernels::FILL
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let module_ci = vk::ShaderModuleCreateInfo::default().code(&spv_u32);
        let module = unsafe { inner.device.create_shader_module(&module_ci, None) }
            .map_err(|e| device_err("create_shader_module(fill)", e))?;

        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let set_layout_ci =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding));
        let set_layout = unsafe { inner.device.create_descriptor_set_layout(&set_layout_ci, None) }
            .map_err(|e| device_err("create_descriptor_set_layout", e))?;

        let pc_range = vk::PushConstantRange::default()
            .offset(0)
            .size(12)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&set_layout))
            .push_constant_ranges(std::slice::from_ref(&pc_range));
        let layout = unsafe { inner.device.create_pipeline_layout(&layout_ci, None) }
            .map_err(|e| device_err("create_pipeline_layout", e))?;

        let entry_point = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry_point);
        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let pipeline = unsafe {
            inner.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&create_info),
                None,
            )
        }
        .map_err(|e| device_err("create_compute_pipelines", e))?[0];

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1);
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(std::slice::from_ref(&pool_size));
        let pool = unsafe { inner.device.create_descriptor_pool(&pool_ci, None) }
            .map_err(|e| device_err("create_descriptor_pool", e))?;
        let set = unsafe {
            inner.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(std::slice::from_ref(&set_layout)),
            )
        }
        .map_err(|e| device_err("allocate_descriptor_sets", e))?[0];

        let info = vk::DescriptorBufferInfo::default()
            .buffer(buffer.handle())
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&info));
        unsafe { inner.device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };

        let push: [u32; 3] = [offset, numel, value];
        let push_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(push.as_ptr() as *const u8, 12) };

        let res = (|| -> Result<()> {
            inner.submit(move |d, cmd| unsafe {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    std::slice::from_ref(&set),
                    &[],
                );
                d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
                d.cmd_dispatch(cmd, numel.div_ceil(64), 1, 1);
            })
        })();

        unsafe {
            inner.device.destroy_descriptor_pool(pool, None);
            inner.device.destroy_pipeline(pipeline, None);
            inner.device.destroy_pipeline_layout(layout, None);
            inner.device.destroy_descriptor_set_layout(set_layout, None);
            inner.device.destroy_shader_module(module, None);
        }
        res
    }
}

fn cpu_storage_bytes(storage: &CpuStorage) -> (&[u8], DType, usize) {
    match storage {
        CpuStorage::U8(data) => (unsafe {
            std::slice::from_raw_parts(data.as_ptr(), data.len() * std::mem::size_of::<u8>())
        }, DType::U8, data.len()),
        CpuStorage::U32(data) => (unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        }, DType::U32, data.len()),
        CpuStorage::I64(data) => (unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
        }, DType::I64, data.len()),
        CpuStorage::BF16(data) => (unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
        }, DType::BF16, data.len()),
        CpuStorage::F16(data) => (unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
        }, DType::F16, data.len()),
        CpuStorage::F32(data) => (unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        }, DType::F32, data.len()),
        CpuStorage::F64(data) => (unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
        }, DType::F64, data.len()),
    }
}

#[derive(Clone)]
pub struct VulkanStorage {
    device: VulkanDevice,
    buffer: Arc<VulkanBuffer>,
    len: usize,
    dtype: DType,
}

impl std::fmt::Debug for VulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VulkanStorage(len={}, dtype={:?})",
            self.len, self.dtype
        )
    }
}

impl VulkanStorage {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn buffer(&self) -> &Arc<VulkanBuffer> {
        &self.buffer
    }

    fn from_bytes(device: VulkanDevice, bytes: &[u8], len: usize, dtype: DType) -> Result<Self> {
        let buffer = device.inner.alloc_device(bytes.len().max(4))?;
        device.inner.copy_h2d(bytes, &buffer)?;
        Ok(Self {
            device,
            buffer,
            len,
            dtype,
        })
    }
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        // Only contiguous full-storage clones are supported so far.
        if layout.start_offset() != 0 || !layout.is_contiguous() {
            return Err(not_yet("try_clone(strided)"));
        }
        let buffer = self.device.inner.alloc_device(self.len * self.dtype.size_in_bytes())?;
        self.device
            .inner
            .copy_buffer(&self.buffer, &buffer, self.len * self.dtype.size_in_bytes())?;
        Ok(Self {
            device: self.device.clone(),
            buffer,
            len: self.len,
            dtype: self.dtype,
        })
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &VulkanDevice {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let bytes = self
            .device
            .inner
            .copy_d2h(&self.buffer, self.len * self.dtype.size_in_bytes())?;
        let le = |b: &[u8]| -> u32 { u32::from_le_bytes(b.try_into().unwrap()) };
        let le64 = |b: &[u8]| -> i64 { i64::from_le_bytes(b.try_into().unwrap()) };
        Ok(match self.dtype {
            DType::U8 => CpuStorage::U8(bytes),
            DType::U32 => CpuStorage::U32(bytes.chunks_exact(4).map(le).collect::<Vec<_>>()),
            DType::I64 => CpuStorage::I64(bytes.chunks_exact(8).map(le64).collect::<Vec<_>>()),
            DType::BF16 => CpuStorage::BF16({
                let mut out = Vec::with_capacity(bytes.len() / 2);
                for c in bytes.chunks_exact(2) {
                    out.push(half::bf16::from_le_bytes([c[0], c[1]]));
                }
                out
            }),
            DType::F16 => CpuStorage::F16({
                let mut out = Vec::with_capacity(bytes.len() / 2);
                for c in bytes.chunks_exact(2) {
                    out.push(f16::from_le_bytes([c[0], c[1]]));
                }
                out
            }),
            DType::F32 => CpuStorage::F32(
                bytes
                    .chunks_exact(4)
                    .map(le)
                    .map(f32::from_bits)
                    .collect::<Vec<_>>(),
            ),
            DType::F64 => CpuStorage::F64(
                bytes
                    .chunks_exact(8)
                    .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                    .map(f64::from_bits)
                    .collect::<Vec<_>>(),
            ),
        })
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let dims = layout.dims();
        let strides = layout.stride();
        if dims.len() > 4 {
            return Err(not_yet("affine(rank > 4)"));
        }
        let numel = layout.shape().elem_count();
        let (spv, name, work) = match self.dtype {
            DType::F32 => (candle_vulkan_kernels::AFFINE_F32, "affine_f32", numel),
            DType::F16 => (candle_vulkan_kernels::AFFINE_F16, "affine_f16", numel.div_ceil(2)),
            _ => return Err(not_yet("affine(non-f32/f16)")),
        };
        let out = self.device.alloc(numel, self.dtype)?;
        let mut pc = [0u32; 14];
        pc[0] = layout.start_offset() as u32;
        pc[1] = 0;
        pc[2] = numel as u32;
        pc[3] = (mul as f32).to_bits();
        pc[4] = (add as f32).to_bits();
        pc[5] = dims.len() as u32;
        for (i, d) in dims.iter().enumerate() {
            pc[6 + i] = *d as u32;
        }
        for (i, s) in strides.iter().enumerate() {
            pc[10 + i] = *s as u32;
        }
        let groups = dispatch_groups(work, 256);
        self.device.inner.dispatch(
            spv,
            name,
            &[&self.buffer, &out],
            &pc,
            groups,
        )?;
        Ok(Self {
            device: self.device.clone(),
            buffer: out,
            len: numel,
            dtype: self.dtype,
        })
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(not_yet("powf"))
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(not_yet("elu"))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(not_yet("reduce_op(non-f32)"));
        }
        let dims = layout.dims();
        if reduce_dims.len() != 1 && reduce_dims.len() != dims.len() {
            return Err(not_yet("reduce_op(multi-dim)"));
        }
        let op_id = match op {
            ReduceOp::Sum => 0u32,
            ReduceOp::Min => 1u32,
            ReduceOp::Max => 2u32,
            _ => return Err(not_yet("reduce_op(argmin/argmax)")),
        };
        let dim = reduce_dims[0];
        if dims.len() > 4 {
            return Err(not_yet("reduce_op(rank > 4)"));
        }
        let numel = layout.shape().elem_count();
        let all_dims = reduce_dims.len() == dims.len();
        let dim_len = if all_dims {
            numel
        } else {
            dims[reduce_dims[0]]
        };
        let numel_out = numel / dim_len.max(1);
        let out = self.device.alloc(numel_out, self.dtype)?;
        let mut pc = [0u32; 16];
        pc[0] = layout.start_offset() as u32;
        pc[1] = 0;
        pc[2] = numel_out as u32;
        pc[3] = dim_len as u32;
        pc[4] = layout.stride()[dim] as u32;
        pc[5] = dims.len() as u32;
        for (i, d) in dims.iter().enumerate() {
            pc[6 + i] = if i == dim && !all_dims { 1 } else { *d as u32 };
        }
        for (i, s) in layout.stride().iter().enumerate() {
            pc[10 + i] = if i == dim && !all_dims { 0 } else { *s as u32 };
        }
        pc[14] = op_id;
        pc[15] = if all_dims { 1 } else { 0 };
        let groups = dispatch_groups(numel_out, 1);
        self.device.inner.dispatch(
            candle_vulkan_kernels::REDUCE_F32,
            "reduce_f32",
            &[&self.buffer, &out],
            &pc,
            groups,
        )?;
        Ok(Self {
            device: self.device.clone(),
            buffer: out,
            len: numel_out,
            dtype: self.dtype,
        })
    }

    fn cmp(
        &self,
        op: CmpOp,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(not_yet("cmp(non-f32)"));
        }
        if lhs_layout.start_offset() % 4 != 0 {
            return Err(not_yet("cmp(unaligned bool dst)"));
        }
        let spv = match op {
            CmpOp::Eq => candle_vulkan_kernels::CMP_EQ_F32,
            CmpOp::Ne => candle_vulkan_kernels::CMP_NE_F32,
            CmpOp::Lt => candle_vulkan_kernels::CMP_LT_F32,
            CmpOp::Le => candle_vulkan_kernels::CMP_LE_F32,
            CmpOp::Gt => candle_vulkan_kernels::CMP_GT_F32,
            CmpOp::Ge => candle_vulkan_kernels::CMP_GE_F32,
        };
        let dims = lhs_layout.dims();
        if dims.len() > 4 || dims != rhs_layout.dims() {
            return Err(not_yet("cmp(rank > 4 or mismatched broadcast)"));
        }
        let numel = lhs_layout.shape().elem_count();
        let out = self.device.alloc(numel, DType::U8)?;
        let mut pc = [0u32; 21];
        pc[0] = lhs_layout.start_offset() as u32;
        pc[1] = rhs_layout.start_offset() as u32;
        pc[2] = 0;
        pc[3] = numel as u32;
        pc[4] = dims.len() as u32;
        for (i, d) in dims.iter().enumerate() {
            pc[5 + i] = *d as u32;
            pc[13 + i] = *d as u32;
        }
        for (i, s) in lhs_layout.stride().iter().enumerate() {
            pc[9 + i] = *s as u32;
        }
        for (i, s) in rhs_layout.stride().iter().enumerate() {
            pc[17 + i] = *s as u32;
        }
        let groups = dispatch_groups(numel.div_ceil(4), 256);
        self.device
            .inner
            .dispatch(spv, "cmp_f32", &[&self.buffer, &rhs.buffer, &out], &pc, groups)?;
        Ok(Self {
            device: self.device.clone(),
            buffer: out,
            len: numel,
            dtype: DType::U8,
        })
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if self.dtype == dtype {
            return self.try_clone(layout);
        }
        let numel = layout.shape().elem_count();
        let out = self.device.alloc(numel, dtype)?;
        let rank = layout.dims().len();
        if rank > 4 {
            return Err(not_yet("to_dtype(rank > 4)"));
        }
        let mut pc = [0u32; 13];
        pc[0] = layout.start_offset() as u32;
        pc[1] = 0;
        pc[2] = numel as u32;
        pc[3] = rank as u32;
        for (i, d) in layout.dims().iter().enumerate() {
            pc[4 + i] = *d as u32;
        }
        for (i, s) in layout.stride().iter().enumerate() {
            pc[8 + i] = *s as u32;
        }
        if self.dtype == DType::F32 && dtype == DType::U8 {
            self.device.inner.dispatch(
                candle_vulkan_kernels::F32_TO_BOOL_F32,
                "f32_to_bool_f32",
                &[&self.buffer, &self.buffer, &out],
                &pc,
                dispatch_groups(numel.div_ceil(4), 256),
            )?;
            return Ok(Self {
                device: self.device.clone(),
                buffer: out,
                len: numel,
                dtype,
            });
        }
        let (spv, name, groups) = match (self.dtype, dtype) {
            (DType::F16, DType::F32) => (
                candle_vulkan_kernels::F16_TO_F32,
                "f16_to_f32",
                dispatch_groups(numel, 256),
            ),
            (DType::F32, DType::F16) => (
                candle_vulkan_kernels::F32_TO_F16,
                "f32_to_f16",
                dispatch_groups(numel.div_ceil(2), 256),
            ),
            (DType::BF16, DType::F32) => (
                candle_vulkan_kernels::BF16_TO_F32,
                "bf16_to_f32",
                dispatch_groups(numel, 256),
            ),
            (DType::F32, DType::BF16) => (
                candle_vulkan_kernels::F32_TO_BF16,
                "f32_to_bf16",
                dispatch_groups(numel.div_ceil(2), 256),
            ),
            (DType::U8, DType::F32) => (
                candle_vulkan_kernels::BOOL_TO_F32,
                "bool_to_f32",
                dispatch_groups(numel, 256),
            ),
            (DType::U32, DType::F32) => (
                candle_vulkan_kernels::U32_TO_F32,
                "u32_to_f32",
                dispatch_groups(numel, 256),
            ),
            (DType::F32, DType::U32) => (
                candle_vulkan_kernels::F32_TO_U32,
                "f32_to_u32",
                dispatch_groups(numel, 256),
            ),
            (DType::I64, DType::F32) => (
                candle_vulkan_kernels::I64_TO_F32,
                "i64_to_f32",
                dispatch_groups(numel, 256),
            ),
            _ => {
                // No direct kernel: route float<->float conversions through f32
                // (e.g. bf16 weights loaded for an f16 model).
                let is_float = |d: DType| matches!(d, DType::F16 | DType::BF16 | DType::F32);
                if is_float(self.dtype) && is_float(dtype) && dtype != DType::F32 {
                    let mid = self.to_dtype(layout, DType::F32)?;
                    let mid_layout = Layout::contiguous(layout.shape().clone());
                    return mid.to_dtype(&mid_layout, dtype);
                }
                return Err(not_yet(&format!(
                    "to_dtype({:?} -> {:?})",
                    self.dtype, dtype
                )));
            }
        };
        self.device
            .inner
            .dispatch(spv, name, &[&self.buffer, &out], &pc, groups)?;
        Ok(Self {
            device: self.device.clone(),
            buffer: out,
            len: numel,
            dtype,
        })
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let f16 = self.dtype == DType::F16;
        if self.dtype != DType::F32 && !f16 {
            return Err(not_yet(B::NAME));
        }
        let pick = |f32: &'static [u8], f16k: &'static [u8]| if f16 { f16k } else { f32 };
        let spv = match B::NAME {
            "exp" => pick(candle_vulkan_kernels::EXP_F32, candle_vulkan_kernels::EXP_F16),
            "log" => pick(candle_vulkan_kernels::LOG_F32, candle_vulkan_kernels::LOG_F16),
            "sin" => pick(candle_vulkan_kernels::SIN_F32, candle_vulkan_kernels::SIN_F16),
            "cos" => pick(candle_vulkan_kernels::COS_F32, candle_vulkan_kernels::COS_F16),
            "tanh" => pick(candle_vulkan_kernels::TANH_F32, candle_vulkan_kernels::TANH_F16),
            "neg" => pick(candle_vulkan_kernels::NEG_F32, candle_vulkan_kernels::NEG_F16),
            "recip" => pick(candle_vulkan_kernels::RECIP_F32, candle_vulkan_kernels::RECIP_F16),
            "sqr" => pick(candle_vulkan_kernels::SQR_F32, candle_vulkan_kernels::SQR_F16),
            "sqrt" => pick(candle_vulkan_kernels::SQRT_F32, candle_vulkan_kernels::SQRT_F16),
            "abs" => pick(candle_vulkan_kernels::ABS_F32, candle_vulkan_kernels::ABS_F16),
            "ceil" => pick(candle_vulkan_kernels::CEIL_F32, candle_vulkan_kernels::CEIL_F16),
            "floor" => pick(candle_vulkan_kernels::FLOOR_F32, candle_vulkan_kernels::FLOOR_F16),
            "round" => pick(candle_vulkan_kernels::ROUND_F32, candle_vulkan_kernels::ROUND_F16),
            "relu" => pick(candle_vulkan_kernels::RELU_F32, candle_vulkan_kernels::RELU_F16),
            "sign" => pick(candle_vulkan_kernels::SIGN_F32, candle_vulkan_kernels::SIGN_F16),
            "erf" => pick(candle_vulkan_kernels::ERF_F32, candle_vulkan_kernels::ERF_F16),
            "gelu" => pick(candle_vulkan_kernels::GELU_F32, candle_vulkan_kernels::GELU_F16),
            "gelu_erf" => pick(candle_vulkan_kernels::GELU_ERF_F32, candle_vulkan_kernels::GELU_ERF_F16),
            "silu" => pick(candle_vulkan_kernels::SILU_F32, candle_vulkan_kernels::SILU_F16),
            _ => return Err(not_yet(B::NAME)),
        };
        let dims = layout.dims();
        if dims.len() > 4 {
            return Err(not_yet("unary(rank > 4)"));
        }
        let numel = layout.shape().elem_count();
        let out = self.device.alloc(numel, self.dtype)?;
        let mut pc = [0u32; 13];
        pc[0] = layout.start_offset() as u32;
        pc[1] = 0;
        pc[2] = numel as u32;
        pc[3] = dims.len() as u32;
        for (i, d) in dims.iter().enumerate() {
            pc[4 + i] = *d as u32;
        }
        for (i, s) in layout.stride().iter().enumerate() {
            pc[8 + i] = *s as u32;
        }
        let work = if f16 { numel.div_ceil(2) } else { numel };
        let groups = dispatch_groups(work, 256);
        self.device
            .inner
            .dispatch(spv, B::NAME, &[&self.buffer, &out], &pc, groups)?;
        Ok(Self {
            device: self.device.clone(),
            buffer: out,
            len: numel,
            dtype: self.dtype,
        })
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        let spv = match (B::NAME, self.dtype) {
            ("add", DType::F32) => candle_vulkan_kernels::ADD_F32,
            ("sub", DType::F32) => candle_vulkan_kernels::SUB_F32,
            ("mul", DType::F32) => candle_vulkan_kernels::MUL_F32,
            ("div", DType::F32) => candle_vulkan_kernels::DIV_F32,
            ("minimum", DType::F32) => candle_vulkan_kernels::MINIMUM_F32,
            ("maximum", DType::F32) => candle_vulkan_kernels::MAXIMUM_F32,
            ("add", DType::F16) => candle_vulkan_kernels::ADD_F16,
            ("sub", DType::F16) => candle_vulkan_kernels::SUB_F16,
            ("mul", DType::F16) => candle_vulkan_kernels::MUL_F16,
            ("div", DType::F16) => candle_vulkan_kernels::DIV_F16,
            ("minimum", DType::F16) => candle_vulkan_kernels::MINIMUM_F16,
            ("maximum", DType::F16) => candle_vulkan_kernels::MAXIMUM_F16,
            ("add", DType::U32) => candle_vulkan_kernels::ADD_U32,
            ("sub", DType::U32) => candle_vulkan_kernels::SUB_U32,
            _ => return Err(not_yet(B::NAME)),
        };
        let dims = lhs_layout.dims();
        if dims.len() > 4 || dims != rhs_layout.dims() {
            return Err(not_yet("binary(rank > 4 or mismatched broadcast)"));
        }
        let numel = lhs_layout.shape().elem_count();
        let out = self.device.alloc(numel, self.dtype)?;
        let mut pc = [0u32; 21];
        pc[0] = lhs_layout.start_offset() as u32;
        pc[1] = rhs_layout.start_offset() as u32;
        pc[2] = 0;
        pc[3] = numel as u32;
        pc[4] = dims.len() as u32;
        for (i, d) in dims.iter().enumerate() {
            pc[5 + i] = *d as u32;
            pc[13 + i] = *d as u32;
        }
        for (i, s) in lhs_layout.stride().iter().enumerate() {
            pc[9 + i] = *s as u32;
        }
        for (i, s) in rhs_layout.stride().iter().enumerate() {
            pc[17 + i] = *s as u32;
        }
        let work = if self.dtype == DType::F16 { numel.div_ceil(2) } else { numel };
        let groups = dispatch_groups(work, 256);
        self.device.inner.dispatch(
            spv,
            B::NAME,
            &[&self.buffer, &rhs.buffer, &out],
            &pc,
            groups,
        )?;
        Ok(Self {
            device: self.device.clone(),
            buffer: out,
            len: numel,
            dtype: self.dtype,
        })
    }

    fn where_cond(
        &self,
        layout: &Layout,
        on_true: &Self,
        true_layout: &Layout,
        on_false: &Self,
        false_layout: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::U8 || on_true.dtype != DType::F32 {
            return Err(not_yet("where_cond(non-bool cond or non-f32)"));
        }
        let dims = layout.dims();
        if dims.len() > 4 || dims != true_layout.dims() || dims != false_layout.dims() {
            return Err(not_yet("where_cond(rank > 4 or shape mismatch)"));
        }
        let numel = layout.shape().elem_count();
        let out = self.device.alloc(numel, DType::F32)?;
        let mut pc = [0u32; 22];
        pc[0] = layout.start_offset() as u32;
        pc[1] = true_layout.start_offset() as u32;
        pc[2] = false_layout.start_offset() as u32;
        pc[3] = 0;
        pc[4] = numel as u32;
        pc[5] = dims.len() as u32;
        for (i, d) in dims.iter().enumerate() {
            pc[6 + i] = *d as u32;
        }
        for (i, s) in layout.stride().iter().enumerate() {
            pc[10 + i] = *s as u32;
        }
        for (i, s) in true_layout.stride().iter().enumerate() {
            pc[14 + i] = *s as u32;
        }
        for (i, s) in false_layout.stride().iter().enumerate() {
            pc[18 + i] = *s as u32;
        }
        let groups = dispatch_groups(numel, 256);
        self.device.inner.dispatch(
            candle_vulkan_kernels::WHERE_COND_F32,
            "where_cond_f32",
            &[&self.buffer, &on_true.buffer, &on_false.buffer, &out],
            &pc,
            groups,
        )?;
        Ok(Self {
            device: self.device.clone(),
            buffer: out,
            len: numel,
            dtype: DType::F32,
        })
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &ParamsConv1D,
    ) -> Result<Self> {
        Err(not_yet("conv1d"))
    }

    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(not_yet("conv_transpose1d"))
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &ParamsConv2D,
    ) -> Result<Self> {
        Err(not_yet("conv2d"))
    }

    fn conv_transpose2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(not_yet("conv_transpose2d"))
    }

    fn index_select(
        &self,
        indexes: &Self,
        layout: &Layout,
        indexes_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let f16 = self.dtype == DType::F16;
        if self.dtype != DType::F32 && !f16 {
            return Err(not_yet("index_select(non-f32/f16)"));
        }
        let ids_stride = match indexes.dtype {
            DType::U32 => 1u32,
            DType::I64 => 2u32,
            _ => return Err(not_yet("index_select(id dtype)")),
        };
        if indexes_layout.dims().len() != 1 {
            return Err(not_yet("index_select(non-1d ids)"));
        }
        let dims = layout.dims();
        if dims.len() > 4 {
            return Err(not_yet("index_select(rank > 4)"));
        }
        let ids_len = indexes_layout.dims()[0];
        let numel = layout.shape().elem_count() / dims[dim] * ids_len;
        let out = self.device.alloc(numel, self.dtype)?;
        let mut pc = [0u32; 15];
        pc[0] = layout.start_offset() as u32;
        pc[1] = 0;
        pc[2] = numel as u32;
        pc[3] = dims.len() as u32;
        for (i, d) in dims.iter().enumerate() {
            pc[4 + i] = if i == dim { ids_len } else { *d } as u32;
        }
        for (i, s) in layout.stride().iter().enumerate() {
            pc[8 + i] = *s as u32;
        }
        pc[12] = indexes_layout.start_offset() as u32 * ids_stride;
        pc[13] = ids_stride;
        pc[14] = dim as u32;
        let work = if f16 { numel.div_ceil(2) } else { numel };
        let groups = dispatch_groups(work, 256);
        let (spv, name) = if f16 {
            (candle_vulkan_kernels::INDEX_SELECT_F16, "index_select_f16")
        } else {
            (candle_vulkan_kernels::INDEX_SELECT_F32, "index_select_f32")
        };
        self.device.inner.dispatch(
            spv,
            name,
            &[&self.buffer, &indexes.buffer, &out],
            &pc,
            groups,
        )?;
        Ok(Self {
            device: self.device.clone(),
            buffer: out,
            len: numel,
            dtype: self.dtype,
        })
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(not_yet("gather"))
    }

    fn scatter_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        Err(not_yet("scatter_add"))
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        Err(not_yet("index_add"))
    }

    fn matmul(
        &self,
        r: &Self,
        mnk: (usize, usize, usize, usize),
        l_layout: &Layout,
        r_layout: &Layout,
    ) -> Result<Self> {
        let f16 = self.dtype == DType::F16;
        if self.dtype != r.dtype || (self.dtype != DType::F32 && !f16) {
            return Err(not_yet("matmul(non-f32/f16)"));
        }
        let (b, m, n, k) = mnk;
        if f16 && n % 2 == 1 {
            return Err(not_yet("matmul_f16(odd n)"));
        }
        let l_rank = l_layout.dims().len();
        let r_rank = r_layout.dims().len();
        if !(2..=4).contains(&l_rank) || l_rank != r_rank {
            return Err(not_yet("matmul(rank not in 2..4 or mismatched)"));
        }
        let (lb0, lb1, lm, lk) = match l_rank {
            2 => (0u32, 0u32, l_layout.stride()[0] as u32, l_layout.stride()[1] as u32),
            3 => (
                l_layout.stride()[0] as u32,
                0,
                l_layout.stride()[1] as u32,
                l_layout.stride()[2] as u32,
            ),
            _ => (
                l_layout.stride()[0] as u32,
                l_layout.stride()[1] as u32,
                l_layout.stride()[2] as u32,
                l_layout.stride()[3] as u32,
            ),
        };
        let (rb0, rb1, rk, rn) = match r_rank {
            2 => (0u32, 0u32, r_layout.stride()[0] as u32, r_layout.stride()[1] as u32),
            3 => (
                r_layout.stride()[0] as u32,
                0,
                r_layout.stride()[1] as u32,
                r_layout.stride()[2] as u32,
            ),
            _ => (
                r_layout.stride()[0] as u32,
                r_layout.stride()[1] as u32,
                r_layout.stride()[2] as u32,
                r_layout.stride()[3] as u32,
            ),
        };
        let numel = b * m * n;
        // Split-K: when the plain 64x64 tiling spawns too few workgroups to fill
        // the GPU (small M/N) but K is long, split K across workgroups into partial
        // planes and reduce them. Big win for skinny GEMMs (small M, large K).
        let tm = m.div_ceil(64);
        let tn = n.div_ceil(64);
        let base_groups = tm * tn * b;
        const TARGET_WG: usize = 256;
        if base_groups < TARGET_WG && k >= 256 {
            let mut k_splits = (TARGET_WG + base_groups - 1) / base_groups;
            k_splits = k_splits.min(k / 16);
            let mut k_chunk = 0usize;
            if k_splits >= 2 {
                k_chunk = ((k + k_splits - 1) / k_splits).div_ceil(16) * 16;
                k_splits = k.div_ceil(k_chunk);
            }
            if k_splits >= 2 {
                let split_stride = numel;
                // Partials stay f32 for both dtypes (accuracy of the K-split sum).
                let partial = self.device.alloc(k_splits * split_stride, DType::F32)?;
                let pc_sk = [
                    l_layout.start_offset() as u32,
                    r_layout.start_offset() as u32,
                    0u32,
                    m as u32,
                    n as u32,
                    k as u32,
                    lb0,
                    lb1,
                    lm,
                    lk,
                    rb0,
                    rb1,
                    rk,
                    rn,
                    (m * n) as u32,
                    if l_rank == 4 {
                        l_layout.dims()[1] as u32
                    } else {
                        1
                    },
                    k_chunk as u32,
                    k_splits as u32,
                    split_stride as u32,
                ];
                let (sk_spv, sk_name) = if f16 {
                    (candle_vulkan_kernels::MATMUL_F16_SPLITK, "matmul_f16_splitk")
                } else {
                    (candle_vulkan_kernels::MATMUL_F32_SPLITK, "matmul_f32_splitk")
                };
                self.device.inner.dispatch(
                    sk_spv,
                    sk_name,
                    &[&self.buffer, &r.buffer, &partial],
                    &pc_sk,
                    (
                        tm.min(65535) as u32,
                        tn.min(65535) as u32,
                        (b * k_splits) as u32,
                    ),
                )?;
                let out = self.device.alloc(numel, self.dtype)?;
                let pc_red = [numel as u32, k_splits as u32, split_stride as u32];
                let (rd_spv, rd_name) = if f16 {
                    (candle_vulkan_kernels::MATMUL_F16_REDUCE, "matmul_f16_reduce")
                } else {
                    (candle_vulkan_kernels::MATMUL_F32_REDUCE, "matmul_f32_reduce")
                };
                let red_work = if f16 { numel / 2 } else { numel };
                self.device.inner.dispatch(
                    rd_spv,
                    rd_name,
                    &[&partial, &out],
                    &pc_red,
                    dispatch_groups(red_work, 256),
                )?;
                return Ok(Self {
                    device: self.device.clone(),
                    buffer: out,
                    len: numel,
                    dtype: self.dtype,
                });
            }
        }
        let out = self.device.alloc(numel, self.dtype)?;
        let pc = [
            l_layout.start_offset() as u32,
            r_layout.start_offset() as u32,
            0u32,
            m as u32,
            n as u32,
            k as u32,
            lb0,
            lb1,
            lm,
            lk,
            rb0,
            rb1,
            rk,
            rn,
            (m * n) as u32,
            if l_rank == 4 {
                l_layout.dims()[1] as u32
            } else {
                1
            },
        ];
        let groups = (
            (m.div_ceil(64)).min(65535) as u32,
            (n.div_ceil(64)).min(65535) as u32,
            b as u32,
        );
        let (spv, name) = if f16 {
            (candle_vulkan_kernels::MATMUL_F16, "matmul_f16")
        } else {
            (candle_vulkan_kernels::MATMUL_F32, "matmul_f32")
        };
        self.device.inner.dispatch(
            spv,
            name,
            &[&self.buffer, &r.buffer, &out],
            &pc,
            groups,
        )?;
        Ok(Self {
            device: self.device.clone(),
            buffer: out,
            len: numel,
            dtype: self.dtype,
        })
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_layout: &Layout) -> Result<()> {
        match self.dtype {
            DType::F32 => {
                let dims = src_layout.dims();
                if dims.len() > 4 {
                    return Err(not_yet("copy_strided_src(rank > 4)"));
                }
                let numel = src_layout.shape().elem_count();
                let mut pc = [0u32; 14];
                pc[0] = src_layout.start_offset() as u32;
                pc[1] = dst_offset as u32;
                pc[2] = numel as u32;
                pc[3] = 1.0f32.to_bits();
                pc[4] = 0.0f32.to_bits();
                pc[5] = dims.len() as u32;
                for (i, d) in dims.iter().enumerate() {
                    pc[6 + i] = *d as u32;
                }
                for (i, s) in src_layout.stride().iter().enumerate() {
                    pc[10 + i] = *s as u32;
                }
                let groups = dispatch_groups(numel, 256);
                self.device.inner.dispatch(
                    candle_vulkan_kernels::AFFINE_F32,
                    "affine_f32",
                    &[&self.buffer, &dst.buffer],
                    &pc,
                    groups,
                )
            }
            DType::F16 => {
                let dims = src_layout.dims();
                if dims.len() > 4 || dst_offset % 2 != 0 {
                    return Err(not_yet("copy_strided_src(f16 rank>4 or odd dst)"));
                }
                let numel = src_layout.shape().elem_count();
                let mut pc = [0u32; 13];
                pc[0] = src_layout.start_offset() as u32;
                pc[1] = dst_offset as u32;
                pc[2] = numel as u32;
                pc[3] = dims.len() as u32;
                for (i, d) in dims.iter().enumerate() {
                    pc[4 + i] = *d as u32;
                }
                for (i, s) in src_layout.stride().iter().enumerate() {
                    pc[8 + i] = *s as u32;
                }
                let groups = dispatch_groups(numel.div_ceil(2), 256);
                self.device.inner.dispatch(
                    candle_vulkan_kernels::COPY_F16,
                    "copy_f16",
                    &[&self.buffer, &dst.buffer],
                    &pc,
                    groups,
                )
            }
            _ => Err(not_yet("copy_strided_src(dtype)")),
        }
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        let f16 = self.dtype == DType::F16;
        if self.dtype != DType::F32 && !f16 {
            return Err(not_yet("copy2d(non-f32/f16)"));
        }
        if f16 && (d2 % 2 == 1 || dst_offset % 2 == 1 || dst_stride1 % 2 == 1) {
            return Err(not_yet("copy2d(f16: unaligned rows)"));
        }
        let pc = [
            src_offset as u32,
            dst_offset as u32,
            d1 as u32,
            d2 as u32,
            src_stride1 as u32,
            dst_stride1 as u32,
        ];
        let n = d1 * d2;
        let groups = dispatch_groups(if f16 { n / 2 } else { n }, 256);
        let (spv, name) = if f16 {
            (candle_vulkan_kernels::COPY2D_F16, "copy2d_f16")
        } else {
            (candle_vulkan_kernels::COPY2D_F32, "copy2d_f32")
        };
        self.device
            .inner
            .dispatch(spv, name, &[&self.buffer, &dst.buffer], &pc, groups)
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(not_yet("avg_pool2d"))
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(not_yet("max_pool2d"))
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(not_yet("upsample_nearest1d"))
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(not_yet("upsample_nearest2d"))
    }
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> Result<Self> {
        VulkanDevice::new(ordinal)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan {
            gpu_id: self.inner.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &rhs.inner)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<VulkanStorage> {
        let len = shape.elem_count();
        let buffer = self.alloc_filled(len, dtype, 0)?;
        Ok(VulkanStorage {
            device: self.clone(),
            buffer,
            len,
            dtype,
        })
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<VulkanStorage> {
        let len = shape.elem_count();
        let size = len * dtype.size_in_bytes();
        let buffer = self.alloc(len, dtype)?;
        if dtype == DType::F32 && size % 4 == 0 {
            // Exercises the compute pipeline (fill shader) end to end.
            self.dispatch_fill(&buffer, len as u32, 0, f32::to_bits(1.0))?;
        } else {
            let pattern: Option<u32> = match dtype {
                DType::F32 => Some(f32::to_bits(1.0)),
                DType::F16 => Some(0x3C00_3C00),
                DType::BF16 => Some(0x3F80_3F80),
                DType::U32 => Some(1),
                DType::U8 if size % 4 == 0 => Some(0x0101_0101),
                _ => None,
            };
            match pattern {
                Some(p) if size % 4 == 0 => self.inner.fill_buffer(&buffer, 0, size, p)?,
                _ => {
                    // One per element via host staging.
                    let mut bytes = Vec::with_capacity(size);
                    for _ in 0..len {
                        match dtype {
                            DType::U8 => bytes.push(1u8),
                            DType::U32 => bytes.extend_from_slice(&1u32.to_le_bytes()),
                            DType::I64 => bytes.extend_from_slice(&1i64.to_le_bytes()),
                            DType::BF16 => bytes.extend_from_slice(&[0x3F, 0x80]),
                            DType::F16 => bytes.extend_from_slice(&[0x00, 0x3C]),
                            DType::F32 => bytes.extend_from_slice(&1.0f32.to_le_bytes()),
                            DType::F64 => bytes.extend_from_slice(&1.0f64.to_le_bytes()),
                        }
                    }
                    self.inner.copy_h2d(&bytes, &buffer)?;
                }
            }
        }
        Ok(VulkanStorage {
            device: self.clone(),
            buffer,
            len,
            dtype,
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<VulkanStorage> {
        let len = shape.elem_count();
        let buffer = self.alloc(len, dtype)?;
        Ok(VulkanStorage {
            device: self.clone(),
            buffer,
            len,
            dtype,
        })
    }

    fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<VulkanStorage> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };
        VulkanStorage::from_bytes(self.clone(), bytes, data.len(), T::DTYPE)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<VulkanStorage> {
        let (bytes, dtype, len) = cpu_storage_bytes(storage);
        VulkanStorage::from_bytes(self.clone(), bytes, len, dtype)
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<VulkanStorage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<VulkanStorage> {
        Err(not_yet("rand_uniform"))
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<VulkanStorage> {
        Err(not_yet("rand_normal"))
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        self.inner.synchronize()
    }
}

/// Fused row-softmax over the last dimension (contiguous f32 input).
/// Called from candle-nn's `softmax-last-dim` custom op.
pub fn softmax_last_dim(storage: &VulkanStorage, layout: &Layout) -> Result<VulkanStorage> {
    if storage.dtype != DType::F32 {
        return Err(not_yet("softmax_last_dim(non-f32)"));
    }
    if layout.contiguous_offsets().is_none() {
        return Err(not_yet("softmax_last_dim(non-contiguous)"));
    }
    let dims = layout.dims();
    let n_cols = *dims.last().unwrap_or(&0);
    let numel = layout.shape().elem_count();
    let n_rows = numel / n_cols.max(1);
    let out = storage.device.alloc(numel, DType::F32)?;
    let pc = [
        layout.start_offset() as u32,
        0,
        n_rows as u32,
        n_cols as u32,
    ];
    let groups = dispatch_groups(n_rows, 1);
    storage.device.inner.dispatch(
        candle_vulkan_kernels::SOFTMAX_F32,
        "softmax_f32",
        &[&storage.buffer, &out],
        &pc,
        groups,
    )?;
    Ok(VulkanStorage {
        device: storage.device.clone(),
        buffer: out,
        len: numel,
        dtype: DType::F32,
    })
}

/// Variable-length flash attention (f32). `q` is `[total_q, num_heads,
/// head_size]`, `k`/`v` are `[total_k, num_heads_k, head_size]`, all
/// contiguous (the caller `.contiguous()`-es them). `seqlens_*` are contiguous
/// `u32` cumulative-length tensors of length `batch + 1`. Supports causal /
/// sliding-window masking (`window_left`/`window_right`, `-1` = unbounded),
/// ALiBi slopes and soft-capping. Returns `[total_q, num_heads, head_size]`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen(
    q: &VulkanStorage,
    q_l: &Layout,
    k: &VulkanStorage,
    k_l: &Layout,
    v: &VulkanStorage,
    v_l: &Layout,
    seqlens_q: &VulkanStorage,
    seqlens_q_l: &Layout,
    seqlens_k: &VulkanStorage,
    seqlens_k_l: &Layout,
    alibi: Option<(&VulkanStorage, &Layout)>,
    softmax_scale: f32,
    window_left: Option<usize>,
    window_right: Option<usize>,
    softcap: Option<f32>,
) -> Result<VulkanStorage> {
    let f16 = q.dtype == DType::F16;
    if !(q.dtype == k.dtype && q.dtype == v.dtype && (q.dtype == DType::F32 || f16)) {
        return Err(not_yet("flash_attn_varlen(non-f32/f16)"));
    }
    let qd = q_l.dims();
    let kd = k_l.dims();
    if qd.len() != 3 || kd.len() != 3 {
        return Err(crate::Error::msg("flash_attn_varlen: q/k/v must be rank-3"));
    }
    let (total_q, num_heads, head_size) = (qd[0], qd[1], qd[2]);
    let (total_k, num_heads_k, head_size_k) = (kd[0], kd[1], kd[2]);
    if head_size != head_size_k {
        return Err(crate::Error::msg("flash_attn_varlen: q/k head size mismatch"));
    }
    if head_size > 256 {
        return Err(crate::Error::msg("flash_attn_varlen: head_size > 256"));
    }
    if f16 {
        if head_size % 2 == 1 {
            return Err(crate::Error::msg("flash_attn_varlen: f16 needs even head_size"));
        }
        if alibi.is_some() {
            return Err(not_yet("flash_attn_varlen(alibi + f16)"));
        }
    }
    if num_heads % num_heads_k != 0 {
        return Err(crate::Error::msg("flash_attn_varlen: num_heads_k must divide num_heads"));
    }
    for (what, l) in [
        ("q", q_l),
        ("k", k_l),
        ("v", v_l),
        ("seqlens_q", seqlens_q_l),
        ("seqlens_k", seqlens_k_l),
    ] {
        if l.contiguous_offsets().is_none() {
            return Err(crate::Error::msg(format!("flash_attn_varlen: {what} not contiguous")));
        }
    }
    if seqlens_q.dtype != DType::U32 || seqlens_k.dtype != DType::U32 {
        return Err(crate::Error::msg("flash_attn_varlen: seqlens must be u32"));
    }
    let batch = seqlens_q_l.dims()[0] - 1;
    let _ = total_k;

    let numel_out = total_q * num_heads * head_size;
    let out = q.device.alloc(numel_out, q.dtype)?;

    let window_left_i: i32 = window_left.map(|w| w as i64).unwrap_or(-1).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let window_right_i: i32 = window_right.map(|w| w as i64).unwrap_or(-1).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let has_alibi = alibi.is_some() as u32;

    let pc: Vec<u32> = vec![
        head_size as u32,
        num_heads as u32,
        num_heads_k as u32,
        total_q as u32,
        q_l.start_offset() as u32,
        k_l.start_offset() as u32,
        v_l.start_offset() as u32,
        0, // o_off (fresh alloc)
        seqlens_q_l.start_offset() as u32,
        seqlens_k_l.start_offset() as u32,
        batch as u32,
        window_left_i as u32,
        window_right_i as u32,
        has_alibi,
        softmax_scale.to_bits(),
        softcap.unwrap_or(0.0).to_bits(),
    ];

    let alibi_buf = match alibi {
        Some((s, _)) => &s.buffer,
        None => &seqlens_k.buffer,
    };
    let bindings = [
        &q.buffer,
        &k.buffer,
        &v.buffer,
        &out,
        &seqlens_q.buffer,
        &seqlens_k.buffer,
        alibi_buf,
    ];
    let groups = (total_q.div_ceil(32).max(1) as u32, num_heads as u32, 1);
    let (spv, name) = if f16 {
        (candle_vulkan_kernels::FA_VARLEN_F16, "fa_varlen_f16")
    } else {
        (candle_vulkan_kernels::FA_VARLEN_F32, "fa_varlen_f32")
    };
    q.device.inner.dispatch(
        spv,
        name,
        &bindings,
        &pc,
        groups,
    )?;
    Ok(VulkanStorage {
        device: q.device.clone(),
        buffer: out,
        len: numel_out,
        dtype: q.dtype,
    })
}
