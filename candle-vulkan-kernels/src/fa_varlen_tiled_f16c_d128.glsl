#version 450
// Tiled varlen flash attention, f16 storage + native f16 compute, head_size = 128
// (Qwen3). Selected over the f32-compute variant only when a runtime benchmark
// shows fp16 compute is faster on the device.
#define HEAD_SIZE 128
#include "fa_varlen_tiled_f16c_body.glslh"
