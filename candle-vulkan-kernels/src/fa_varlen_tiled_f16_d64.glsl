#version 450
// Tiled varlen flash attention, f16 storage, head_size = 64 (GTE / Arctic).
#define HEAD_SIZE 64
#include "fa_varlen_tiled_f16_body.glslh"
