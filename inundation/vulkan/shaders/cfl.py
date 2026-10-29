"""CFL and time-control shader sources for SWE solver."""

CFL_ACCUM_GLSL = """\
#version 450
#extension GL_EXT_shader_atomic_float : require

layout(local_size_x = 256) in;

layout(set=0, binding=0)  buffer H      { float h[];      };
layout(set=0, binding=1)  buffer HU     { float hu[];     };
layout(set=0, binding=2)  buffer HV     { float hv[];     };
layout(set=0, binding=3)  buffer ZB     { float zb[];     };
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
layout(set=0, binding=19) buffer BC     { float bc[];     };

layout(push_constant) uniform PC {
    float num_edges;
    float dt;
    float g;
    float dry_tol;
    float cfl_number;
} pc;

void main() {
    uint eid = gl_GlobalInvocationID.x;
    if (eid >= uint(pc.num_edges)) return;

    int   cL  = ecL[eid];
    int   cR  = ecR[eid];
    float nx_ = enx[eid];
    float ny_ = eny[eid];
    float len = elen[eid];

    float hL  = max(h[cL], 0.0);
    float cWL = sqrt(pc.g * hL);
    float unL = (hL > pc.dry_tol) ? (hu[cL]*nx_ + hv[cL]*ny_) / hL : 0.0;
    float sL  = abs(unL) + cWL;

    float sR;
    if (cR >= 0) {
        float hR  = max(h[cR], 0.0);
        float cWR = sqrt(pc.g * hR);
        float unR = (hR > pc.dry_tol) ? (hu[cR]*nx_ + hv[cR]*ny_) / hR : 0.0;
        sR = abs(unR) + cWR;
    } else {
        // Open boundaries: the exterior wave speed, not the mirrored interior.
        int   bkind = int(bc[4u * eid]);
        float bval  = bc[4u * eid + 1u];
        if (bkind == 2) {
            sR = abs(unL) + sqrt(pc.g * bval);
        } else if (bkind == 1) {
            float hf = max(hL, pow(bval * bval / pc.g, 1.0 / 3.0));
            sR = (hf > pc.dry_tol) ? abs(bval) / hf + sqrt(pc.g * hf) : sL;
        } else {
            sR = sL;
        }
    }

    float contrib = max(sL, sR) * len;
    atomicAdd(cfl_scratch[cL], contrib);
    if (cR >= 0) { atomicAdd(cfl_scratch[cR], contrib); }
}

"""

CFL_REDUCE_GLSL = """\
#version 450

layout(local_size_x = 256) in;

layout(set=0, binding=0)  buffer H      { float h[];      };
layout(set=0, binding=1)  buffer HU     { float hu[];     };
layout(set=0, binding=2)  buffer HV     { float hv[];     };
layout(set=0, binding=3)  buffer ZB     { float zb[];     };
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
layout(set=0, binding=3)  buffer ZB     { float zb[];     };
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
layout(set=0, binding=3)  buffer ZB     { float zb[];     };
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

CFL_RESOLVE_TIME_GLSL = """\
#version 450

layout(local_size_x = 1) in;

layout(set=0, binding=0)  buffer H      { float h[];      };
layout(set=0, binding=1)  buffer HU     { float hu[];     };
layout(set=0, binding=2)  buffer HV     { float hv[];     };
layout(set=0, binding=3)  buffer ZB     { float zb[];     };
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
layout(set=0, binding=19) buffer TIME   { float time_buf[]; };

layout(push_constant) uniform PC {
    float dt_max;
    float stop_time;
    float cfl_safety;
    float dry_dt;
} pc;

void main() {
    float remaining = pc.stop_time - time_buf[0];
    if (remaining <= 1e-7) {
        dt_buf_u[0] = floatBitsToUint(0.0);
        return;
    }

    float dt_cap = min(pc.dt_max, remaining);
    float dt_raw = uintBitsToFloat(dt_buf_u[0]);
    float dt_new;
    if (dt_raw < 1e-10) {
        dt_new = 0.0;
    } else if (dt_raw < 1e10) {
        dt_new = min(dt_raw * pc.cfl_safety, dt_cap);
    } else {
        dt_new = min(pc.dry_dt, dt_cap);
    }
    dt_buf_u[0] = floatBitsToUint(dt_new);
}

"""

TIME_ADVANCE_GLSL = """\
#version 450

layout(local_size_x = 1) in;

layout(set=0, binding=0)  buffer H      { float h[];      };
layout(set=0, binding=1)  buffer HU     { float hu[];     };
layout(set=0, binding=2)  buffer HV     { float hv[];     };
layout(set=0, binding=3)  buffer ZB     { float zb[];     };
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
layout(set=0, binding=19) buffer TIME   { float time_buf[]; };

void main() {
    time_buf[0] += dt_buf[0];
}

"""
