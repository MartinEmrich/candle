#version 450
#define UN_OP(a) (0.5 * (a) * (1.0 + tanh(0.7978845608028654 * (a) * (1.0 + 0.044715 * (a) * (a)))))
#include "unary_common_bf16.glslh"
