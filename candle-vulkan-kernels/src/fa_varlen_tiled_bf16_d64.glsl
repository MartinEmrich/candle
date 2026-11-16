#version 450
// Tiled varlen flash attention, bf16 storage, head_size = 64.
#define HEAD_SIZE 64
#include "fa_varlen_tiled_bf16_body.glslh"
