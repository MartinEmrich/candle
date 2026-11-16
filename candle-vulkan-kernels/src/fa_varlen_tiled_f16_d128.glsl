#version 450
// Tiled varlen flash attention, f16 storage, head_size = 128 (Qwen3).
#define HEAD_SIZE 128
#include "fa_varlen_tiled_f16_body.glslh"
