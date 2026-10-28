"""Kernels for the GPU-resident strategy: integer clock and in-step CFL.

The resident loop records many steps into one command buffer, so each step must
pick its own dt on device. Binding 17 holds two dt words, read as raw bits:

- ``dt_buf_u[0]``: this step's dt as float bits, written by ``step_control``.
  Zero means the step is a no-op (the window's stop time was reached), and
  every kernel here returns immediately.
- ``dt_buf_u[1]``: the CFL-limited dt for the next step, reduced by
  ``atomicMin`` over cells in the corrector update. Positive floats order like
  their bit patterns, so the integer minimum is the float minimum.

Simulated time is a uint64 tick count (2^-20 s) in the control block that
``step_control`` owns. dt is always a whole number of ticks no larger than
2^24, so ``float(ticks) * 2^-20`` is exact and the clock advances by exactly
the dt the physics used.

Push-constant structs of the flux and update kernels match the ``dtbuf``
variants field for field, so ``_pc_flux`` and ``_pc_update`` drive both.

Active set: cells are grouped in blocks of 256 (one cell workgroup, ``i >> 8``)
with two flag words each in binding 20: ``wet`` (a cell is wet or still has
momentum) and ``active`` (``wet`` dilated over each block's two-ring of
neighbouring blocks, bindings 22-23). One step changes nothing outside two cell
rings of its initially wet cells, so the flux and update kernels skip every
block that is not ``active``. Edges are sorted by the block of their left cell
and binding 21 holds each edge workgroup's first and last block. See
``docs/implementation/22-active-set/plan.md`` for the argument.

Accumulation: the ``*_GATHER_GLSL`` kernels replace the float ``atomicAdd`` of
fluxes and CFL terms with per-edge writes and a fixed-order per-cell sum, so
repeated runs are bit-identical. The remaining atomics (``atomicMin`` for dt,
block flag stores) do not depend on order.
"""

from .flux import HLLC_GLSL

# Bindings 0-18 shared by every resident kernel. dt is declared as uint words.
_BINDINGS = """\
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
"""

# Active-set bindings; binding 19 stays per-kernel (clamp mass, source list).
_ACTIVE_BINDINGS = """\
layout(set=0, binding=20) buffer BLKF   { uint blk[];    };
layout(set=0, binding=21) buffer ERANGE { uint erange[]; };
layout(set=0, binding=22) buffer BPTR   { uint bptr[];   };
layout(set=0, binding=23) buffer BIDX   { uint bidx[];   };

const uint BLK_WORDS = 2u;
const uint BLK_WET = 0u;
const uint BLK_ACTIVE = 1u;

bool block_active(uint b) { return blk[BLK_WORDS * b + BLK_ACTIVE] != 0u; }
uint block_of(uint cell) { return cell >> 8; }
"""

# Deterministic accumulation (``deterministic`` on the solver): the flux kernel
# writes each edge's contributions to binding 24 instead of adding them into
# its cells, and the update kernel sums each cell's edges in the fixed order of
# the per-cell CSR (bindings 25-26). Binding 27 marks the cells that have a
# live edge this stage, so idle cells skip the sum. See
# ``docs/implementation/23-deterministic-accumulation/plan.md``.
_GATHER_BINDINGS = """\
layout(set=0, binding=24) buffer EFLUX  { float ef[];      };
layout(set=0, binding=25) buffer CPTR   { uint  cptr[];    };
layout(set=0, binding=26) buffer CIDX   { uint  cidx[];    };
layout(set=0, binding=27) buffer TOUCH  { uint  touched[]; };

// Per-edge words: mass flux; left momentum x, y; right momentum x, y (they
// differ by the hydrostatic correction); corrector CFL wave-speed term.
const uint EF_WORDS = 6u;
const uint EF_CFL = 5u;
"""

_FLUX_ACCUM_ATOMIC = """\
    atomicAdd(dh[cL],   Fh  * len);
    atomicAdd(dhu[cL],  Fhu * len + SL_g * nx_ * len);
    atomicAdd(dhv[cL],  Fhv * len + SL_g * ny_ * len);

    if (cR >= 0) {
        atomicAdd(dh[cR],  -Fh  * len);
        atomicAdd(dhu[cR], -Fhu * len - SR_g * nx_ * len);
        atomicAdd(dhv[cR], -Fhv * len - SR_g * ny_ * len);
    }
"""

_FLUX_ACCUM_GATHER = """\
    // Every writer stores the same value, so the race on a shared cell is benign.
    touched[cL] = 1u;
    if (cR >= 0) { touched[cR] = 1u; }
    uint o = EF_WORDS * eid;
    ef[o]      = Fh  * len;
    ef[o + 1u] = Fhu * len + SL_g * nx_ * len;
    ef[o + 2u] = Fhv * len + SL_g * ny_ * len;
    ef[o + 3u] = Fhu * len + SR_g * nx_ * len;
    ef[o + 4u] = Fhv * len + SR_g * ny_ * len;
"""

_FLUX_CFL_ATOMIC = """\
        atomicAdd(cfl_scratch[cL], contrib);
        if (cR >= 0) { atomicAdd(cfl_scratch[cR], contrib); }
"""

_FLUX_CFL_GATHER = """\
        ef[EF_WORDS * eid + EF_CFL] = contrib;
"""


def _flux_resident(gather: bool) -> str:
    return (
        "#version 450\n"
        + ("" if gather else "#extension GL_EXT_shader_atomic_float : require\n")
        + "\nlayout(local_size_x = 256) in;\n\n"
        + _BINDINGS
        + _ACTIVE_BINDINGS
        + (_GATHER_BINDINGS if gather else "")
        + """
layout(push_constant) uniform PC {
    float num_edges;
    float dt_unused;
    float g;
    float dry_tol;
    float cfl_number;
    float stage;
} pc;

"""
        + HLLC_GLSL
        + """\
// Wave-speed bound of one cell's raw (unreconstructed) state along the edge
// normal: the same |u_n| + sqrt(g h) that CFL_ACCUM_GLSL uses.
float wave_speed(float h_, float hu_, float hv_, float nx_, float ny_) {
    float hp = max(h_, 0.0);
    float un = (hp > pc.dry_tol) ? (hu_ * nx_ + hv_ * ny_) / hp : 0.0;
    return abs(un) + sqrt(pc.g * hp);
}

// The corrector reads the predictor state, except in an inactive block: its
// predictor state was not computed this step and is stale, and the full step
// would have left it equal to the start-of-step state.
bool use_predictor(int c) {
    return pc.stage > 0.5 && block_active(block_of(uint(c)));
}

void load_state(int c, out float h_, out float hu_, out float hv_) {
    if (use_predictor(c)) {
        h_  = h1[c];
        hu_ = hu1[c];
        hv_ = hv1[c];
    } else {
        h_  = h[c];
        hu_ = hu[c];
        hv_ = hv[c];
    }
}

void main() {
    uint eid = gl_GlobalInvocationID.x;
    if (eid >= uint(pc.num_edges)) return;
    if (dt_buf_u[0] == 0u) return;

    // Skip the workgroup unless one of its left-cell blocks is active.
    uint wg = gl_WorkGroupID.x;
    bool live = false;
    for (uint b = erange[2u * wg]; b <= erange[2u * wg + 1u]; ++b) {
        if (block_active(b)) { live = true; break; }
    }
    if (!live) return;

    int  cL   = ecL[eid];
    int  cR   = ecR[eid];
    float nx_ = enx[eid];
    float ny_ = eny[eid];
    float len = elen[eid];

    float hL, huL, hvL;
    load_state(cL, hL, huL, hvL);
    float zbL = zb[cL];

    float hR, huR, hvR, zbR;
    if (cR < 0) {
        hR  = hL;
        huR = huL - 2.0 * (huL * nx_ + hvL * ny_) * nx_;
        hvR = hvL - 2.0 * (huL * nx_ + hvL * ny_) * ny_;
        zbR = zbL;
    } else {
        load_state(cR, hR, huR, hvR);
        zbR = zb[cR];
    }

    // Fully dry edges carry no flux and, below dry_tol, no CFL constraint that
    // could bind: a cell whose edges are all dry gets no dt estimate at all.
    // The gather update applies the same test and skips them.
    if (hL < pc.dry_tol && hR < pc.dry_tol) return;


    float z_face = max(zbL, zbR);
    float hLs    = max(0.0, hL + zbL - z_face);
    float hRs    = max(0.0, hR + zbR - z_face);

    float uL = safe_vel(hL, huL);
    float vL = safe_vel(hL, hvL);
    float uR = safe_vel(hR, huR);
    float vR = safe_vel(hR, hvR);

    float huLs = hLs * uL;
    float hvLs = hLs * vL;
    float huRs = hRs * uR;
    float hvRs = hRs * vR;

    float hqnL =  huLs * nx_ + hvLs * ny_;
    float hqtL = -huLs * ny_ + hvLs * nx_;
    float hqnR =  huRs * nx_ + hvRs * ny_;
    float hqtR = -huRs * ny_ + hvRs * nx_;

    float Fh, Fhn, Fht;
    hllc(hLs, hqnL, hqtL, hRs, hqnR, hqtR, Fh, Fhn, Fht);

    float Fhu = Fhn * nx_ - Fht * ny_;
    float Fhv = Fhn * ny_ + Fht * nx_;

    // Audusse (2004) hydrostatic-reconstruction source term — see FLUX_GLSL
    // for the derivation. Identical to the other flux variants.
    float SL_g = 0.5 * pc.g * (hL * hL - hLs * hLs);
    float SR_g = 0.5 * pc.g * (hR * hR - hRs * hRs);

"""
        + (_FLUX_ACCUM_GATHER if gather else _FLUX_ACCUM_ATOMIC)
        + """
    // Corrector only: accumulate the next step's CFL bound from this stage's
    // input state. A wall mirrors the left state, so its speed equals sL.
    if (pc.stage > 0.5) {
        float sL = wave_speed(hL, huL, hvL, nx_, ny_);
        float sR = wave_speed(hR, huR, hvR, nx_, ny_);
        float contrib = max(sL, sR) * len;
"""
        + (_FLUX_CFL_GATHER if gather else _FLUX_CFL_ATOMIC)
        + """\
    }
}

"""
    )


FLUX_RESIDENT_GLSL = _flux_resident(gather=False)
FLUX_RESIDENT_GATHER_GLSL = _flux_resident(gather=True)

_UPDATE_SUM_ATOMIC = """\
    float sdh  = dh[i];
    float sdhu = dhu[i];
    float sdhv = dhv[i];
    dh[i]  = 0.0;
    dhu[i] = 0.0;
    dhv[i] = 0.0;
    // Only the corrector flux accumulates the CFL bound.
    float lam = 0.0;
    if (pc.stage > 0.5) {
        lam = cfl_scratch[i];
        cfl_scratch[i] = 0.0;
    }
"""

# The flux kernel skips an edge exactly when both sides are dry in the stage's
# input state, and an inactive block's edges are dry on both sides; the same
# test here, on the same bits, keeps the sum from reading a value not written
# this stage. A cell with no live edge (not touched) sums nothing.
_UPDATE_SUM_GATHER = """\
    float sdh = 0.0, sdhu = 0.0, sdhv = 0.0, lam = 0.0;
    uint kend = cptr[i];
    if (touched[i] != 0u) {
        touched[i] = 0u;
        kend = cptr[i + 1u];
    }
    bool own_dry = hi < pc.dry_tol;
    for (uint k = cptr[i]; k < kend; ++k) {
        uint s = cidx[k];
        uint e = s >> 1;
        bool right = (s & 1u) != 0u;
        if (own_dry) {
            int nb = right ? ecL[e] : ecR[e];
            if (nb < 0) continue;  // a wall mirrors this dry cell
            bool pred = pc.stage > 0.5 && block_active(block_of(uint(nb)));
            if ((pred ? h1[nb] : h[nb]) < pc.dry_tol) continue;
        }
        uint o = EF_WORDS * e;
        if (right) {
            sdh  -= ef[o];
            sdhu -= ef[o + 3u];
            sdhv -= ef[o + 4u];
        } else {
            sdh  += ef[o];
            sdhu += ef[o + 1u];
            sdhv += ef[o + 2u];
        }
        if (pc.stage > 0.5) { lam += ef[o + EF_CFL]; }
    }
"""


def _update_resident(gather: bool) -> str:
    return (
        "#version 450\n\nlayout(local_size_x = 256) in;\n\n"
        + _BINDINGS
        + _ACTIVE_BINDINGS
        + (_GATHER_BINDINGS if gather else "")
        + """\
layout(set=0, binding=19) buffer CLAMPM { float clamp_mass[]; };

layout(push_constant) uniform PC {
    float num_cells;
    float dt_unused;
    float g;
    float dry_tol;
    float cfl_number;
    float stage;
    float track_clamp;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(pc.num_cells)) return;
    if (dt_buf_u[0] == 0u) return;
    // One workgroup is one block.
    uint b = gl_WorkGroupID.x;
    if (!block_active(b)) return;
    // The predictor clears the block's mark; the corrector and sources set it
    // again. A marked block is always active next step, so every mark is
    // cleared before it can go stale.
    if (pc.stage < 0.5 && gl_LocalInvocationID.x == 0u) {
        blk[BLK_WORDS * b + BLK_WET] = 0u;
    }

    float A   = area[i];
    float dt  = uintBitsToFloat(dt_buf_u[0]);
    float dti = dt / A;

    float hi, hui, hvi;
    if (pc.stage < 0.5) {
        hi  = h[i];
        hui = hu[i];
        hvi = hv[i];
    } else {
        hi  = h1[i];
        hui = hu1[i];
        hvi = hv1[i];
    }

"""
        + (_UPDATE_SUM_GATHER if gather else _UPDATE_SUM_ATOMIC)
        + """
    float h_raw  = hi  - dti * sdh;
    float hu_new = hui - dti * sdhu;
    float hv_new = hvi - dti * sdhv;

    float h_new = max(h_raw, 0.0);
    if (pc.track_clamp > 0.5) {
        clamp_mass[i] += 0.5 * A * (h_new - h_raw);
    }
    if (h_new < pc.dry_tol) { hu_new = 0.0; hv_new = 0.0; }

    if (h_new >= pc.dry_tol) {
        float n  = n_mann[i];
        float u_ = hu_new / h_new;
        float v_ = hv_new / h_new;
        float spd = sqrt(u_ * u_ + v_ * v_);
        float h43 = pow(h_new, 4.0 / 3.0);
        float Sf  = pc.g * n * n * spd / h43;
        float den = 1.0 + dt * Sf;
        hu_new = hu_new / den;
        hv_new = hv_new / den;
    }

    if (pc.stage < 0.5) {
        h1[i]  = h_new;
        hu1[i] = hu_new;
        hv1[i] = hv_new;
    } else {
        float hf  = 0.5 * (h[i]  + h_new);
        float huf = 0.5 * (hu[i] + hu_new);
        float hvf = 0.5 * (hv[i] + hv_new);
        h[i]  = hf;
        hu[i] = huf;
        hv[i] = hvf;

        // Keep the block live for the next step while a cell is wet or still
        // carries momentum (a cell that just dried keeps half of it).
        if (hf >= pc.dry_tol || huf != 0.0 || hvf != 0.0) {
            blk[BLK_WORDS * b + BLK_WET] = 1u;
        }

        // Reduce this cell's CFL bound into the next step's dt (CFL_REDUCE_GLSL
        // semantics).
        if (lam > 1e-30 && A > 1e-10) {
            atomicMin(dt_buf_u[1], floatBitsToUint(pc.cfl_number * A / lam));
        }
    }
}

"""
    )


UPDATE_RESIDENT_GLSL = _update_resident(gather=False)
UPDATE_RESIDENT_GATHER_GLSL = _update_resident(gather=True)

# Prologue CFL bound for the gather path: each edge's wave-speed term (as in
# CFL_ACCUM_GLSL) into binding 24, then each cell sums its edges in CSR order
# and reduces into dt slot 0 (as CFL_REDUCE_GLSL does).
CFL_EDGE_GATHER_GLSL = (
    """\
#version 450

layout(local_size_x = 256) in;

"""
    + _BINDINGS
    + _ACTIVE_BINDINGS
    + _GATHER_BINDINGS
    + """\
layout(push_constant) uniform PC {
    float num_edges;
    float g;
    float dry_tol;
} pc;

float wave_speed(int c, float nx_, float ny_) {
    float hp = max(h[c], 0.0);
    float un = (hp > pc.dry_tol) ? (hu[c] * nx_ + hv[c] * ny_) / hp : 0.0;
    return abs(un) + sqrt(pc.g * hp);
}

void main() {
    uint eid = gl_GlobalInvocationID.x;
    if (eid >= uint(pc.num_edges)) return;
    int cL = ecL[eid];
    int cR = ecR[eid];
    float sL = wave_speed(cL, enx[eid], eny[eid]);
    float sR = (cR >= 0) ? wave_speed(cR, enx[eid], eny[eid]) : sL;
    ef[EF_WORDS * eid + EF_CFL] = max(sL, sR) * elen[eid];
}

"""
)

CFL_GATHER_GLSL = (
    """\
#version 450

layout(local_size_x = 256) in;

"""
    + _BINDINGS
    + _ACTIVE_BINDINGS
    + _GATHER_BINDINGS
    + """\
layout(push_constant) uniform PC {
    float num_cells;
    float cfl_number;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(pc.num_cells)) return;
    float lam = 0.0;
    for (uint k = cptr[i]; k < cptr[i + 1u]; ++k) {
        lam += ef[EF_WORDS * (cidx[k] >> 1) + EF_CFL];
    }
    if (lam > 1e-30 && area[i] > 1e-10) {
        atomicMin(dt_buf_u[0], floatBitsToUint(pc.cfl_number * area[i] / lam));
    }
}

"""
)


# Sources as a compact list: binding 19 holds (cell index, dh/dt) word pairs,
# the rate as float bits. A source can wet a cell in an inactive block, so it
# marks that block for the next step.
SOURCE_RESIDENT_GLSL = (
    """\
#version 450

layout(local_size_x = 256) in;

"""
    + _BINDINGS
    + _ACTIVE_BINDINGS
    + """\
layout(set=0, binding=19) buffer SRC    { uint src_u[]; };

layout(push_constant) uniform PC {
    float num_sources;
    float dry_tol;
} pc;

void main() {
    uint k = gl_GlobalInvocationID.x;
    if (k >= uint(pc.num_sources)) return;
    if (dt_buf_u[0] == 0u) return;
    uint i = src_u[2u * k];
    float s = uintBitsToFloat(src_u[2u * k + 1u]);
    float h_new = max(h[i] + uintBitsToFloat(dt_buf_u[0]) * s, 0.0);
    h[i] = h_new;
    if (h_new >= pc.dry_tol) {
        blk[BLK_WORDS * block_of(i) + BLK_WET] = 1u;
    }
}

"""
)

# Start of a run: set each block's wet flag from the current state (wet or
# nonzero momentum), or, with the active set off, mark every block active.
# One thread per block.
BLOCK_MARK_GLSL = (
    """\
#version 450

layout(local_size_x = 64) in;

"""
    + _BINDINGS
    + _ACTIVE_BINDINGS
    + """\
layout(push_constant) uniform PC {
    float num_cells;
    float num_blocks;
    float dry_tol;
    float all_active;
} pc;

void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= uint(pc.num_blocks)) return;
    if (pc.all_active > 0.5) {
        blk[BLK_WORDS * b + BLK_WET] = 0u;
        blk[BLK_WORDS * b + BLK_ACTIVE] = 1u;
        return;
    }
    uint first = b << 8;
    uint last = min(first + 256u, uint(pc.num_cells));
    uint live = 0u;
    for (uint i = first; i < last; ++i) {
        if (h[i] >= pc.dry_tol || hu[i] != 0.0 || hv[i] != 0.0) { live = 1u; break; }
    }
    blk[BLK_WORDS * b + BLK_WET] = live;
}

"""
)

# Active blocks for the next step: a block is active if it or any block in its
# two-ring is marked. Runs after the corrector and the sources; a no-op step
# (dt == 0) changed nothing, so it keeps the flags. The prologue forces it.
BLOCK_DILATE_GLSL = (
    """\
#version 450

layout(local_size_x = 64) in;

"""
    + _BINDINGS
    + _ACTIVE_BINDINGS
    + """\
layout(push_constant) uniform PC {
    float num_blocks;
    float force;
} pc;

void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= uint(pc.num_blocks)) return;
    if (pc.force < 0.5 && dt_buf_u[0] == 0u) return;
    uint v = blk[BLK_WORDS * b + BLK_WET];
    for (uint k = bptr[b]; v == 0u && k < bptr[b + 1u]; ++k) {
        v = blk[BLK_WORDS * bidx[k] + BLK_WET];
    }
    blk[BLK_WORDS * b + BLK_ACTIVE] = v;
}

"""
)

# Control block layout; the host mirrors it as a numpy structured dtype
# (``RESIDENT_CTRL_DTYPE`` in gpu_resident_batch.py). Host-written fields:
# stop_ticks, dt_max_ticks, dry_dt_ticks, cfl_safety. Mode 0 (start of a run)
# takes the CFL bound from a prologue reduction in dt slot 0; mode 1 (after a
# step) advances the clock and takes the bound from slot 1; mode 2 (new output
# window) re-plans from the last bound without advancing.
STEP_CONTROL_GLSL = (
    """\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 1) in;

"""
    + _BINDINGS
    + """\
layout(std430, set=0, binding=19) buffer CTRL {
    uint64_t now_ticks;
    uint64_t stop_ticks;
    uint64_t steps;
    uint     dt_ticks;
    uint     dt_max_ticks;
    uint     dry_dt_ticks;
    float    last_raw;
    float    cfl_safety;
    uint     _pad;
} ctrl;

layout(push_constant) uniform PC {
    float mode;
} pc;

const float TICKS_PER_SECOND = 1048576.0;
const float DT_SENTINEL      = 1e30;

void main() {
    float raw;
    if (pc.mode < 0.5) {
        raw = uintBitsToFloat(dt_buf_u[0]);
        ctrl.last_raw = raw;
    } else if (pc.mode < 1.5) {
        uint done = ctrl.dt_ticks;
        ctrl.now_ticks += uint64_t(done);
        if (done > 0u) {
            ctrl.steps += 1ul;
            raw = uintBitsToFloat(dt_buf_u[1]);
            ctrl.last_raw = raw;
        } else {
            // A no-op step computed no CFL bound; keep the last real one.
            raw = ctrl.last_raw;
        }
    } else {
        raw = ctrl.last_raw;
    }

    uint64_t remaining =
        (ctrl.stop_ticks > ctrl.now_ticks) ? ctrl.stop_ticks - ctrl.now_ticks : 0ul;

    uint ticks;
    if (remaining == 0ul || raw < 1e-10) {
        // Window done, or a degenerate bound: the host sees no progress.
        ticks = 0u;
    } else if (raw >= 1e10) {
        // No wet edge anywhere: fall back to the configured dry-domain dt.
        ticks = ctrl.dry_dt_ticks;
    } else {
        float cap = float(ctrl.dt_max_ticks) / TICKS_PER_SECOND;
        ticks = uint(min(raw * ctrl.cfl_safety, cap) * TICKS_PER_SECOND);
    }
    ticks = min(ticks, ctrl.dt_max_ticks);
    if (uint64_t(ticks) > remaining) { ticks = uint(remaining); }

    ctrl.dt_ticks = ticks;
    dt_buf_u[0] = floatBitsToUint(float(ticks) / TICKS_PER_SECOND);
    dt_buf_u[1] = floatBitsToUint(DT_SENTINEL);
}

"""
)

__all__ = [
    "BLOCK_DILATE_GLSL",
    "BLOCK_MARK_GLSL",
    "CFL_EDGE_GATHER_GLSL",
    "CFL_GATHER_GLSL",
    "FLUX_RESIDENT_GATHER_GLSL",
    "FLUX_RESIDENT_GLSL",
    "SOURCE_RESIDENT_GLSL",
    "STEP_CONTROL_GLSL",
    "UPDATE_RESIDENT_GATHER_GLSL",
    "UPDATE_RESIDENT_GLSL",
]
