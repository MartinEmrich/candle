#version 450
#define UN_OP(a) ((erf_impl((a) * 0.7071067811865476) + 1.0) * 0.5 * (a))
#include "unary_common_f16.glslh"
