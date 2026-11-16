#version 450
// Tiled varlen flash attention, bf16 storage, head_size = 128.
#define HEAD_SIZE 128
#include "fa_varlen_tiled_bf16_body.glslh"
