"""Source-term shader sources for SWE solver."""

SOURCE_GLSL = """\
#version 450

layout(local_size_x = 256) in;

layout(set=0, binding=0) buffer H   { float h[];   };
layout(set=0, binding=1) buffer SRC { float src[]; };

layout(push_constant) uniform PC {
    uint  num_cells;
    float dt;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.num_cells) return;
    float s = src[i];
    if (s != 0.0) {
        h[i] = max(h[i] + pc.dt * s, 0.0);
    }
}

"""
