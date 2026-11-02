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

Open boundaries: ``ecR[e] == -1`` is a wall and ``ecR[e] == -2 - k`` is open edge
``k``, whose condition is in binding 19 of the kernels that build its ghost state.
An open edge always writes its gather words, so an update can read it without
repeating the flux's dry test. ``TALLY_RESIDENT_GLSL`` measures the volume through
each stretch. See ``docs/implementation/29-open-boundaries/plan.md`` and, for
``OPEN_EXCHANGE``, ``docs/implementation/30-two-way-coupling/plan.md``.

Accumulation: the ``*_GATHER_GLSL`` kernels replace the float ``atomicAdd`` of
fluxes and CFL terms with per-edge writes and a fixed-order per-cell sum, so
repeated runs are bit-identical. The remaining atomics (``atomicMin`` for dt,
block flag stores) do not depend on order.
"""

import sys

from .flux import FILM_DRAG_GLSL, HLLC_GLSL

# Bindings 0-18 shared by every resident kernel. dt is declared as uint words.
_BINDINGS = """\
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
"""

# Active-set bindings; binding 19 stays per-kernel (clamp mass, source list,
# open-edge conditions). BLK_FORCED is set by the host for a block an open edge
# can wet from outside: it counts as wet however dry the block is.
_ACTIVE_BINDINGS = """\
layout(set=0, binding=20) buffer BLKF   { uint blk[];    };
layout(set=0, binding=21) buffer ERANGE { uint erange[]; };
layout(set=0, binding=22) buffer BPTR   { uint bptr[];   };
layout(set=0, binding=23) buffer BIDX   { uint bidx[];   };

const uint BLK_WORDS = 3u;
const uint BLK_WET = 0u;
const uint BLK_ACTIVE = 1u;
const uint BLK_FORCED = 2u;

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

# Open edges: the ghost state beyond edge ``-2 - cR`` and its mass flux. A
# boundary edge's bed steps are zero, except below a higher neighbour's bed, where
# the ghost is reconstructed to it like the cell (``set_open_steps``). Words per
# edge: condition, its value and the discharge per unit width an exchange edge
# receives (see the docstring of ``open_ghost``).
_OPEN_BINDINGS = """\
layout(set=0, binding=19) buffer OPEN   { uint open_u[]; };
"""

_OPEN_GHOST_GLSL = """\
const uint OPEN_WORDS = 3u;
const uint OPEN_WALL = 0u;
const uint OPEN_FREE = 1u;
const uint OPEN_NORMAL = 2u;
const uint OPEN_STAGE = 3u;
const uint OPEN_INFLOW = 4u;
const uint OPEN_EXCHANGE = 5u;
const uint OPEN_NEWTON_STEPS = 4u;

uint open_kind(int cR) { return open_u[OPEN_WORDS * uint(-2 - cR)]; }
float open_value(int cR) { return uintBitsToFloat(open_u[OPEN_WORDS * uint(-2 - cR) + 1u]); }

// Inflow keeps its edge live while dry, or its volume would be lost.
float open_received(int cR) { return uintBitsToFloat(open_u[OPEN_WORDS * uint(-2 - cR) + 2u]); }
bool open_received_here(int cR, float hL) {
    return open_kind(cR) == OPEN_EXCHANGE && open_received(cR) > 0.0 && open_value(cR) >= hL;
}
bool open_forced(int cR) {
    if (cR >= -1) { return false; }
    uint kind = open_kind(cR);
    return (kind == OPEN_INFLOW && open_value(cR) > 0.0)
        || (kind == OPEN_EXCHANGE && open_received(cR) > 0.0);
}

// Depth beyond an inflow edge of discharge q per unit width: the root of
// -q/hG + 2 sqrt(g hG) = unL + 2 cL, the outgoing characteristic. Solved in
// s = sqrt(hG), where the left side rises and is concave, so Newton from below
// climbs to the root without passing it; a start above it is halved at worst.
float inflow_depth(float q, float riemann) {
    float root_g = sqrt(pc.g);
    float s = max(pow(q / (2.0 * root_g), 1.0 / 3.0), riemann / (2.0 * root_g));
    for (uint k = 0u; k < OPEN_NEWTON_STEPS; ++k) {
        float f  = 2.0 * root_g * s - q / (s * s) - riemann;
        float df = 2.0 * root_g + 2.0 * q / (s * s * s);
        s = max(s - f / df, 0.5 * s);
    }
    return s * s;
}

// The state beyond an open edge, from the cell inside it. The value word is
// sqrt(S)/n for OPEN_NORMAL, the depth over this cell's bed for OPEN_STAGE and
// OPEN_EXCHANGE, and the discharge per unit width for OPEN_INFLOW. An exchange edge
// is a stage while the neighbour's level is lower, and otherwise an inflow of what
// it received, or a wall: a level above the cell must not push on it.
void open_ghost(int cR, float hL, float huL, float hvL, float nx_, float ny_,
                out float hG, out float huG, out float hvG)
{
    uint kind = open_kind(cR);
    float value = open_value(cR);
    float uL = safe_vel(hL, huL);
    float vL = safe_vel(hL, hvL);
    float unL = uL * nx_ + vL * ny_;
    float ut = -uL * ny_ + vL * nx_;
    float cL = sqrt(pc.g * max(hL, 0.0));
    float un = -unL;
    hG = hL;
    float inflow = kind == OPEN_INFLOW ? value : (kind == OPEN_EXCHANGE ? open_received(cR) : 0.0);
    if (kind == OPEN_FREE) {
        un = abs(unL);
    } else if (kind == OPEN_NORMAL) {
        un = value * pow(max(hL, 0.0), 2.0 / 3.0);
    } else if (kind == OPEN_STAGE || (kind == OPEN_EXCHANGE && value < hL)) {
        // A level cannot reach a boundary against supercritical outflow.
        if (hL < pc.dry_tol || unL < cL) {
            hG = value;
            un = unL + 2.0 * (cL - sqrt(pc.g * value));
        } else {
            un = unL;
        }
    } else if (inflow > 0.0) {
        hG = inflow_depth(inflow, unL + 2.0 * cL);
        un = -inflow / hG;
        ut = 0.0;
    }
    huG = hG * (un * nx_ - ut * ny_);
    hvG = hG * (un * ny_ + ut * nx_);
}

// Mass flux through a boundary edge: a wall passes none, as FLUX_GLSL explains;
// outflow conditions never draw water in; inflow carries exactly its discharge,
// and an exchange edge carries what it received on top of what leaves.
float boundary_mass_flux(int cR, float Fh) {
    if (cR == -1) { return 0.0; }
    uint kind = open_kind(cR);
    if (kind == OPEN_FREE || kind == OPEN_NORMAL) { return max(Fh, 0.0); }
    if (kind == OPEN_STAGE) { return Fh; }
    if (kind == OPEN_INFLOW) { return -open_value(cR); }
    if (kind == OPEN_EXCHANGE) { return max(Fh, 0.0) - open_received(cR); }
    return 0.0;
}
"""

_FLUX_GHOST_ATOMIC = """\
    if (cR < 0) {
        hR  = hL;
        huR = huL - 2.0 * (huL * nx_ + hvL * ny_) * nx_;
        hvR = hvL - 2.0 * (huL * nx_ + hvL * ny_) * ny_;
    } else {
        load_state(cR, hR, huR, hvR);
    }

    // Fully dry edges carry no flux and, below dry_tol, no CFL constraint that
    // could bind: a cell whose edges are all dry gets no dt estimate at all.
    // The gather update applies the same test and skips them.
    if (hL < pc.dry_tol && hR < pc.dry_tol) return;
"""

_FLUX_GHOST_GATHER = """\
    if (cR == -1) {
        hR  = hL;
        huR = huL - 2.0 * (huL * nx_ + hvL * ny_) * nx_;
        hvR = hvL - 2.0 * (huL * nx_ + hvL * ny_) * ny_;
    } else if (cR < 0) {
        open_ghost(cR, hL, huL, hvL, nx_, ny_, hR, huR, hvR);
    } else {
        load_state(cR, hR, huR, hvR);
    }

    // Fully dry edges carry no flux and, below dry_tol, no CFL constraint that
    // could bind: a cell whose edges are all dry gets no dt estimate at all.
    // The gather update applies the same test and skips them, except on an
    // open edge, which clears its words here instead.
    if (hL < pc.dry_tol && hR < pc.dry_tol && !open_forced(cR)) {
        if (cR < -1) {
            for (uint w = 0u; w < EF_WORDS; ++w) { ef[EF_WORDS * eid + w] = 0.0; }
        }
        return;
    }
"""

# Water arriving over a crest falls past it, so its momentum reaches the cell below:
# clipped to the crest, the ghost would hand over none.
_FLUX_STEP_GATHER = """\
    if (cR < -1 && open_received_here(cR, hL)) { dzL = 0.0; dzR = 0.0; }
"""

_FLUX_MASS_ATOMIC = """\
    // A wall passes no mass (see FLUX_GLSL).
    if (cR < 0) { Fh = 0.0; }
"""

_FLUX_MASS_GATHER = """\
    if (cR < 0) { Fh = boundary_mass_flux(cR, Fh); }
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
        + (_GATHER_BINDINGS + _OPEN_BINDINGS if gather else "")
        + """
layout(push_constant) uniform PC {
    uint  num_edges;
    float dt_unused;
    float g;
    float dry_tol;
    float cfl_number;
    float stage;
} pc;

"""
        + HLLC_GLSL
        + (_OPEN_GHOST_GLSL if gather else "")
        + """\
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
    if (eid >= pc.num_edges) return;
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

    float hR, huR, hvR;
"""
        + (_FLUX_GHOST_GATHER if gather else _FLUX_GHOST_ATOMIC)
        + """
    float dzL = edz[2u * eid];
    float dzR = edz[2u * eid + 1u];
"""
        + (_FLUX_STEP_GATHER if gather else "")
        + """
    float hLs, hqnL, hqtL, hRs, hqnR, hqtR;
    face_states(hL, huL, hvL, dzL, hR, huR, hvR, dzR, nx_, ny_,
                hLs, hqnL, hqtL, hRs, hqnR, hqtR);

    float Fh, Fhn, Fht, Smax;
    hllc(hLs, hqnL, hqtL, hRs, hqnR, hqtR, Fh, Fhn, Fht, Smax);
"""
        + (_FLUX_MASS_GATHER if gather else _FLUX_MASS_ATOMIC)
        + """
    float Fhu = Fhn * nx_ - Fht * ny_;
    float Fhv = Fhn * ny_ + Fht * nx_;

    // Audusse (2004) hydrostatic-reconstruction source term — see FLUX_GLSL
    // for the derivation. Identical to the other flux variants.
    float SL_g = 0.5 * pc.g * (hL * hL - hLs * hLs);
    float SR_g = 0.5 * pc.g * (hR * hR - hRs * hRs);

"""
        + (_FLUX_ACCUM_GATHER if gather else _FLUX_ACCUM_ATOMIC)
        + """
    // Corrector only: accumulate the next step's CFL bound from the wave
    // speeds this stage's flux assumed.
    if (pc.stage > 0.5) {
        float contrib = Smax * len;
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
            if (nb == -1) continue;  // a wall mirrors this dry cell
            if (nb >= 0) {
                bool pred = pc.stage > 0.5 && block_active(block_of(uint(nb)));
                if ((pred ? h1[nb] : h[nb]) < pc.dry_tol) continue;
            }
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


# Compensated depth (``compensated`` on the solver): the depth of a cell is the
# exact sum of two float32 words, h + h_lo with |h_lo| <= ulp(h)/2 (bindings
# 28-29; the predictor keeps its own pair). Every update goes through an
# error-free transform, so an increment smaller than half an ulp of h is kept
# in h_lo instead of being rounded away. The flux kernels read h alone.
_COMP_BINDINGS = """\
layout(set=0, binding=28) buffer HLO    { float h_lo[];   };
layout(set=0, binding=29) buffer H1LO   { float h1_lo[];  };
"""

# Knuth's two-sum: s + e == a + b exactly. `precise` keeps a compiler from
# reassociating the error term away. MoltenVK is the exception: its fast math
# breaks the transforms with or without the qualifier, so the solver turns fast
# math off there, and the qualifier alone adds about 50 % per step (0.327 against
# 0.222 ms on a 32 k wet mesh; 0.210 ms without compensation).
_EFT_TEMPLATE = """\
void two_sum(float a, float b, out float s, out float e) {
    PRECISE float t  = a + b;
    PRECISE float bb = t - a;
    PRECISE float err = (a - (t - bb)) + (b - bb);
    s = t;
    e = err;
}

// (hi + loi) + inc as a renormalised pair, clamped at zero.
void add_compensated(float hi, float loi, float inc, out float h_new, out float lo_new) {
    float s, e;
    two_sum(hi, inc, s, e);
    e += loi;
    two_sum(s, e, h_new, lo_new);
    if (h_new < 0.0 || (h_new == 0.0 && lo_new < 0.0)) { h_new = 0.0; lo_new = 0.0; }
}
"""


def error_free_glsl(platform: str) -> str:
    """Two-sum kernels for a host platform (``sys.platform``)."""
    return _EFT_TEMPLATE.replace("PRECISE ", "" if platform == "darwin" else "precise ")


_EFT_GLSL = error_free_glsl(sys.platform)

_UPDATE_DEPTH_PLAIN = """\
    float h_raw  = hi  - dti * sdh;
    float h_new = max(h_raw, 0.0);
    float lo_new = 0.0;
"""

# h_raw is the unclamped high word; the clamp diagnostic ignores the low word.
_UPDATE_DEPTH_COMP = """\
    float loi = (pc.stage < 0.5) ? h_lo[i] : h1_lo[i];
    float h_new, lo_new;
    add_compensated(hi, loi, -dti * sdh, h_new, lo_new);
    float s_raw, e_raw;
    two_sum(hi, -dti * sdh, s_raw, e_raw);
    float h_raw = s_raw + (e_raw + loi);
"""

_UPDATE_STORE_PLAIN = """\
    if (pc.stage < 0.5) {
        h1[i]  = h_new;
    } else {
        hf = 0.5 * (h[i] + h_new);
        h[i] = hf;
    }
"""

_UPDATE_STORE_COMP = """\
    if (pc.stage < 0.5) {
        h1[i]    = h_new;
        h1_lo[i] = lo_new;
    } else {
        // Heun average of two compensated depths; halving is exact.
        float s, e;
        two_sum(h[i], h_new, s, e);
        e += h_lo[i] + lo_new;
        float lof;
        two_sum(0.5 * s, 0.5 * e, hf, lof);
        h[i]    = hf;
        h_lo[i] = lof;
    }
"""


def _update_resident(gather: bool, compensated: bool = False) -> str:
    return (
        "#version 450\n\nlayout(local_size_x = 256) in;\n\n"
        + _BINDINGS
        + _ACTIVE_BINDINGS
        + (_GATHER_BINDINGS if gather else "")
        + (_COMP_BINDINGS if compensated else "")
        + """\
layout(set=0, binding=19) buffer CLAMPM { float clamp_mass[]; };

layout(push_constant) uniform PC {
    uint  num_cells;
    float dt_unused;
    float g;
    float dry_tol;
    float cfl_number;
    float stage;
    float track_clamp;
} pc;

"""
        + FILM_DRAG_GLSL
        + (_EFT_GLSL if compensated else "")
        + """
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.num_cells) return;
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
        + "\n"
        + (_UPDATE_DEPTH_COMP if compensated else _UPDATE_DEPTH_PLAIN)
        + """\
    float hu_new = hui - dti * sdhu;
    float hv_new = hvi - dti * sdhv;

    if (pc.track_clamp > 0.5) {
        clamp_mass[i] += 0.5 * A * (h_new - h_raw);
    }
    if (h_new < pc.dry_tol) { hu_new = 0.0; hv_new = 0.0; }
    // Keep the stored momentum consistent with the film velocity the flux sees.
    if (h_new >= pc.dry_tol) {
        float drag = film_drag(h_new, dt);
        hu_new *= drag;
        hv_new *= drag;
    }

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

    float hf = 0.0;
"""
        + (_UPDATE_STORE_COMP if compensated else _UPDATE_STORE_PLAIN)
        + """\
    if (pc.stage < 0.5) {
        hu1[i] = hu_new;
        hv1[i] = hv_new;
    } else {
        float huf = 0.5 * (hu[i] + hu_new);
        float hvf = 0.5 * (hv[i] + hv_new);
        // A cell that ends the step dry carries no momentum. The stage results
        // zeroed theirs; averaging in the start-of-step value would leave half
        // of it behind, to re-enter the flow when the cell wets again.
        if (hf < pc.dry_tol) { huf = 0.0; hvf = 0.0; }
        hu[i] = huf;
        hv[i] = hvf;

        // Keep the block live for the next step while a cell is wet.
        if (hf >= pc.dry_tol) {
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
UPDATE_RESIDENT_COMP_GLSL = _update_resident(gather=False, compensated=True)
UPDATE_RESIDENT_GATHER_COMP_GLSL = _update_resident(gather=True, compensated=True)

# Prologue CFL bound for the gather path: each edge's wave-speed term (as in
# CFL_ACCUM_GLSL) into binding 24, then each cell sums its edges in CSR order
# and reduces into dt slot 0 (as CFL_REDUCE_GLSL does). The gather reads every
# edge, so a fully dry edge writes zero rather than nothing.
CFL_EDGE_GATHER_GLSL = (
    """\
#version 450

layout(local_size_x = 256) in;

"""
    + _BINDINGS
    + _ACTIVE_BINDINGS
    + _GATHER_BINDINGS
    + _OPEN_BINDINGS
    + """\
layout(push_constant) uniform PC {
    uint  num_edges;
    float g;
    float dry_tol;
} pc;

"""
    + HLLC_GLSL
    + _OPEN_GHOST_GLSL
    + """\
void main() {
    uint eid = gl_GlobalInvocationID.x;
    if (eid >= pc.num_edges) return;
    int cL = ecL[eid];
    int cR = ecR[eid];
    float nx_ = enx[eid];
    float ny_ = eny[eid];

    float hL = h[cL], huL = hu[cL], hvL = hv[cL];
    float hR, huR, hvR;
    if (cR == -1) {
        hR  = hL;
        huR = huL - 2.0 * (huL * nx_ + hvL * ny_) * nx_;
        hvR = hvL - 2.0 * (huL * nx_ + hvL * ny_) * ny_;
    } else if (cR < 0) {
        open_ghost(cR, hL, huL, hvL, nx_, ny_, hR, huR, hvR);
    } else {
        hR = h[cR]; huR = hu[cR]; hvR = hv[cR];
    }
    float contrib = 0.0;
    if (hL >= pc.dry_tol || hR >= pc.dry_tol) {
        float dzL = edz[2u * eid];
        float dzR = edz[2u * eid + 1u];
        contrib = wave_bound(hL, huL, hvL, dzL, hR, huR, hvR, dzR, nx_, ny_) * elen[eid];
    }
    ef[EF_WORDS * eid + EF_CFL] = contrib;
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
    uint  num_cells;
    float cfl_number;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.num_cells) return;
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


# Volume through each stretch after one flux stage, as an exact pair of float32
# words per direction: out (hi, lo), in (hi, lo). A corrector-averaged step moves
# dt/2 of each stage's flux. One invocation per stretch sums its edges in a fixed
# order. An inactive block's words are stale, and its cells did not update either.
TALLY_RESIDENT_GLSL = (
    """\
#version 450

layout(local_size_x = 1) in;

layout(set=0, binding=0) buffer DTBUF { uint  dt_buf_u[]; };
layout(set=0, binding=1) buffer EFLUX { float ef[];       };
layout(set=0, binding=2) buffer ECL   { int   ecL[];      };
layout(set=0, binding=3) buffer BLKF  { uint  blk[];      };
layout(set=0, binding=4) buffer SPTR  { uint  stretch_ptr[];  };
layout(set=0, binding=5) buffer SEDGE { uint  stretch_edge[]; };
layout(set=0, binding=6) buffer TALLY { float tally[];    };

const uint BLK_WORDS = 3u;
const uint BLK_ACTIVE = 1u;
const uint EF_WORDS = 6u;

"""
    + _EFT_GLSL
    + """
void main() {
    uint stretch = gl_GlobalInvocationID.x;
    if (dt_buf_u[0] == 0u) return;
    float half_dt = 0.5 * uintBitsToFloat(dt_buf_u[0]);
    for (uint k = stretch_ptr[stretch]; k < stretch_ptr[stretch + 1u]; ++k) {
        uint e = stretch_edge[k];
        uint cell = uint(ecL[e]);
        if (blk[BLK_WORDS * (cell >> 8) + BLK_ACTIVE] == 0u) continue;
        float v = half_dt * ef[EF_WORDS * e];
        uint o = 4u * stretch + ((v < 0.0) ? 2u : 0u);
        float s, err, hi, lo;
        two_sum(tally[o], abs(v), s, err);
        err += tally[o + 1u];
        two_sum(s, err, hi, lo);
        tally[o] = hi;
        tally[o + 1u] = lo;
    }
}

"""
)


# Sources as a compact list: binding 19 holds (cell index, dh/dt) word pairs,
# the rate as float bits. A source can wet a cell in an inactive block, so it
# marks that block for the next step. With ``compensate`` set the increment
# goes into the (h, h_lo) pair (h_lo at binding 24 here).
SOURCE_RESIDENT_GLSL = (
    """\
#version 450

layout(local_size_x = 256) in;

"""
    + _BINDINGS
    + _ACTIVE_BINDINGS
    + """\
layout(set=0, binding=19) buffer SRC    { uint src_u[]; };
layout(set=0, binding=24) buffer HLO    { float h_lo[]; };
layout(set=0, binding=25) buffer CPTR   { uint  cptr[];  };
layout(set=0, binding=26) buffer CIDX   { uint  cidx[];  };

layout(push_constant) uniform PC {
    uint  num_sources;
    float dry_tol;
    float compensate;
    float g;
    float cfl_number;
} pc;

"""
    + _EFT_GLSL
    + """
void main() {
    uint k = gl_GlobalInvocationID.x;
    if (k >= pc.num_sources) return;
    if (dt_buf_u[0] == 0u) return;
    uint i = src_u[2u * k];
    float s = uintBitsToFloat(src_u[2u * k + 1u]);
    float inc = uintBitsToFloat(dt_buf_u[0]) * s;
    float h_new;
    if (pc.compensate > 0.5) {
        float lo_new;
        add_compensated(h[i], h_lo[i], inc, h_new, lo_new);
        h_lo[i] = lo_new;
    } else {
        h_new = max(h[i] + inc, 0.0);
    }
    h[i] = h_new;
    if (h_new >= pc.dry_tol) {
        blk[BLK_WORDS * block_of(i) + BLK_WET] = 1u;
        // The next step's dt was bounded from the corrector's input state,
        // before this water arrived: a point source into a thin cell can
        // raise its wave speed many-fold in one step. Bound dt by this cell's
        // post-source state edge by edge: the dry-bed 2c towards a dry
        // neighbour, |u| + c otherwise, so the cell cannot empty below zero. A
        // wet neighbour's own speed needs no term here: the source did not
        // change it, and the corrector's bound already covers it.
        float u = sqrt(hu[i] * hu[i] + hv[i] * hv[i]) / h_new;
        float c = sqrt(pc.g * h_new);
        float lam = 0.0;
        for (uint k = cptr[i]; k < cptr[i + 1u]; ++k) {
            uint s = cidx[k];
            uint e = s >> 1;
            int nb = ((s & 1u) != 0u) ? ecL[e] : ecR[e];
            bool dry_side = nb >= 0 && h[nb] < pc.dry_tol;
            lam += (u + (dry_side ? 2.0 : 1.0) * c) * elen[e];
        }
        if (lam > 1e-30 && area[i] > 1e-10) {
            atomicMin(dt_buf_u[1], floatBitsToUint(pc.cfl_number * area[i] / lam));
        }
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
    uint  num_cells;
    uint  num_blocks;
    float dry_tol;
    float all_active;
} pc;

void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= pc.num_blocks) return;
    if (pc.all_active > 0.5) {
        blk[BLK_WORDS * b + BLK_WET] = 0u;
        blk[BLK_WORDS * b + BLK_ACTIVE] = 1u;
        return;
    }
    uint first = b << 8;
    uint last = min(first + 256u, pc.num_cells);
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
    uint  num_blocks;
    float force;
} pc;

void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= pc.num_blocks) return;
    if (pc.force < 0.5 && dt_buf_u[0] == 0u) return;
    uint v = blk[BLK_WORDS * b + BLK_WET] | blk[BLK_WORDS * b + BLK_FORCED];
    for (uint k = bptr[b]; v == 0u && k < bptr[b + 1u]; ++k) {
        v = blk[BLK_WORDS * bidx[k] + BLK_WET] | blk[BLK_WORDS * bidx[k] + BLK_FORCED];
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

# Start-of-run check that the compiler kept the error-free arithmetic: the error
# term of 1 + 1e-9 must survive. Fast math folds it to zero.
EFT_PROBE_GLSL = (
    """\
#version 450

layout(local_size_x = 1) in;

layout(set=0, binding=0) buffer V { float v[]; };

"""
    + _EFT_GLSL
    + """
void main() {
    float s, e;
    two_sum(v[0], v[1], s, e);
    v[2] = e;
}

"""
)

__all__ = [
    "BLOCK_DILATE_GLSL",
    "BLOCK_MARK_GLSL",
    "CFL_EDGE_GATHER_GLSL",
    "CFL_GATHER_GLSL",
    "EFT_PROBE_GLSL",
    "FLUX_RESIDENT_GATHER_GLSL",
    "FLUX_RESIDENT_GLSL",
    "SOURCE_RESIDENT_GLSL",
    "STEP_CONTROL_GLSL",
    "UPDATE_RESIDENT_COMP_GLSL",
    "UPDATE_RESIDENT_GATHER_COMP_GLSL",
    "UPDATE_RESIDENT_GATHER_GLSL",
    "UPDATE_RESIDENT_GLSL",
]
