#version 450
// Tiled varlen flash attention, f32, head_size = 64 (Snowflake GTE / Arctic).
#define HEAD_SIZE 64
#include "fa_varlen_tiled_f32_body.glslh"
