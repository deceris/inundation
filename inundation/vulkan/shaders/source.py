"""Source-term shader sources for SWE solver."""

SOURCE_GLSL = """\
#version 450

layout(local_size_x = 256) in;

layout(set=0, binding=0) buffer H   { float h[];   };
layout(set=0, binding=1) buffer SRC { float src[]; };

layout(push_constant) uniform PC {
    float num_cells;
    float dt;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(pc.num_cells)) return;
    float s = src[i];
    if (s != 0.0) {
        h[i] = max(h[i] + pc.dt * s, 0.0);
    }
}

"""

SOURCE_DTBUF_GLSL = """\
#version 450

layout(local_size_x = 256) in;

layout(set=0, binding=0)  buffer H      { float h[];      };
layout(set=0, binding=1)  buffer HU     { float hu[];     };
layout(set=0, binding=2)  buffer HV     { float hv[];     };
layout(set=0, binding=3)  buffer EDZ    { float edz[];    };
layout(set=0, binding=4)  buffer AREA   { float area[];   };
layout(set=0, binding=5)  buffer ELEN   { float elen[];   };
layout(set=0, binding=6)  buffer ENX    { float enx[];    };
layout(set=0, binding=7)  buffer ENY    { float eny[];    };
layout(set=0, binding=8)  buffer ECL    { int   ecL[];    };
layout(set=0, binding=9)  buffer ECR    { int   ecR[];    };
layout(set=0, binding=10) buffer DH     { float dh[];     };
layout(set=0, binding=11) buffer DHU    { float dhu[];    };
layout(set=0, binding=12) buffer DHV    { float dhv[];    };
layout(set=0, binding=13) buffer H1     { float h1[];     };
layout(set=0, binding=14) buffer HU1    { float hu1[];    };
layout(set=0, binding=15) buffer HV1    { float hv1[];    };
layout(set=0, binding=16) buffer NMANN  { float n_mann[]; };
layout(set=0, binding=17) buffer DTBUF  { float dt_buf[]; };
layout(set=0, binding=18) buffer CFLSC  { float cfl_scratch[]; };
layout(set=0, binding=19) buffer SRC    { float src[]; };

layout(push_constant) uniform PC {
    float num_cells;
    float _pad0;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(pc.num_cells)) return;
    float s = src[i];
    float dt = dt_buf[0];
    if (s != 0.0) {
        h[i] = max(h[i] + dt * s, 0.0);
    }
}

"""
