#version 450
#define UN_OP(a) ((a) / (1.0 + exp(-(a))))
#include "unary_common_bf16.glslh"
