"""CFL and time-control shader sources for SWE solver."""

from .flux import HLLC_GLSL

CFL_ACCUM_GLSL = (
    """\
#version 450
#extension GL_EXT_shader_atomic_float : require

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

layout(push_constant) uniform PC {
    float num_edges;
    float dt;
    float g;
    float dry_tol;
    float cfl_number;
} pc;

"""
    + HLLC_GLSL
    + """\
void main() {
    uint eid = gl_GlobalInvocationID.x;
    if (eid >= uint(pc.num_edges)) return;

    int   cL  = ecL[eid];
    int   cR  = ecR[eid];
    float nx_ = enx[eid];
    float ny_ = eny[eid];
    float len = elen[eid];

    float hL = h[cL], huL = hu[cL], hvL = hv[cL];
    float hR, huR, hvR;
    if (cR < 0) {
        hR  = hL;
        huR = huL - 2.0 * (huL * nx_ + hvL * ny_) * nx_;
        hvR = hvL - 2.0 * (huL * nx_ + hvL * ny_) * ny_;
    } else {
        hR = h[cR]; huR = hu[cR]; hvR = hv[cR];
    }
    // A fully dry edge carries no flux and no CFL constraint, as in the flux
    // kernels: a cell whose edges are all dry gets no dt estimate at all.
    if (hL < pc.dry_tol && hR < pc.dry_tol) return;

    float dzL = edz[2u * eid];
    float dzR = edz[2u * eid + 1u];
    float contrib = wave_bound(hL, huL, hvL, dzL, hR, huR, hvR, dzR, nx_, ny_) * len;
    atomicAdd(cfl_scratch[cL], contrib);
    if (cR >= 0) { atomicAdd(cfl_scratch[cR], contrib); }
}

"""
)

CFL_REDUCE_GLSL = """\
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
layout(set=0, binding=17) buffer DTBUF  { uint  dt_buf_u[]; };
layout(set=0, binding=18) buffer CFLSC  { float cfl_scratch[]; };

layout(push_constant) uniform PC {
    float num_cells;
    float dt;
    float g;
    float dry_tol;
    float cfl_number;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(pc.num_cells)) return;

    float lam = cfl_scratch[i];
    cfl_scratch[i] = 0.0;

    // Skip degenerate cells (near-zero area) to avoid clamping dt to zero
    if (lam > 1e-30 && area[i] > 1e-10) {
        float dt_i = pc.cfl_number * area[i] / lam;
        atomicMin(dt_buf_u[0], floatBitsToUint(dt_i));
    }
}

"""

CFL_RESOLVE_GLSL = """\
#version 450

layout(local_size_x = 1) in;

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
layout(set=0, binding=17) buffer DTBUF  { uint  dt_buf_u[]; };
layout(set=0, binding=18) buffer CFLSC  { float cfl_scratch[]; };

layout(push_constant) uniform PC {
    float dt_prev;
    float dt_cap;
    float cfl_safety;
    float _pad0;
    float _pad1;
} pc;

void main() {
    float dt_raw = uintBitsToFloat(dt_buf_u[0]);
    float dt_new;
    if (dt_raw < 1e-10) {
        dt_new = 0.0;
    } else if (dt_raw < 1e10) {
        dt_new = min(dt_raw * pc.cfl_safety, pc.dt_cap);
    } else {
        dt_new = min(pc.dt_prev, pc.dt_cap);
    }
    dt_buf_u[0] = floatBitsToUint(dt_new);
}

"""

DT_RESET_GLSL = """\
#version 450

layout(local_size_x = 1) in;

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
layout(set=0, binding=17) buffer DTBUF  { uint  dt_buf_u[]; };
layout(set=0, binding=18) buffer CFLSC  { float cfl_scratch[]; };

layout(push_constant) uniform PC {
    float sentinel;
    float _pad0;
    float _pad1;
    float _pad2;
} pc;

void main() {
    dt_buf_u[0] = floatBitsToUint(pc.sentinel);
}

"""
