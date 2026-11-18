#version 450
// Tiled varlen flash attention, f16 storage + native f16 compute, head_size = 64
// (GTE / Arctic). Selected over the f32-compute variant only when a runtime
// benchmark shows fp16 compute is faster on the device.
#define HEAD_SIZE 64
#include "fa_varlen_tiled_f16c_body.glslh"
