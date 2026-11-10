"""Flux shader sources for SWE solver.

``HLLC_GLSL`` holds the face reconstruction, the HLLC Riemann solver and the
wave-speed bound shared by every flux and CFL kernel; it needs a push-constant
block with ``g`` and ``dry_tol``. The two kernels here differ only in the name
of their unused dt slot, so the flux and CFL kernels of every backend run the
same arithmetic.
"""

import sys

from ...tuning import FILM_DEPTHS

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
layout(set=0, binding=17) buffer DTBUF  { float dt_buf[]; };
layout(set=0, binding=18) buffer CFLSC  { float cfl_scratch[]; };
"""

# Riemann solver and face reconstruction shared by every flux and CFL kernel.
_FILM_CONST_GLSL = f"const float FILM_DEPTHS = {FILM_DEPTHS};\n"

# Momentum decay of a thin film, semi-implicit like the friction term. The update
# applies it so the stored momentum follows the velocity the flux sees; it is a
# rate, so it does not depend on the tick size. The rate is the ratio the
# desingularised velocity falls short by, over the film's gravity time scale.
FILM_DRAG_GLSL = (
    _FILM_CONST_GLSL
    + """\
float film_drag(float h_, float dt_) {
    float e_ = FILM_DEPTHS * pc.dry_tol;
    if (h_ >= e_) return 1.0;
    float rate = (e_ * e_ - h_ * h_) / (2.0 * h_ * h_) * sqrt(pc.g / e_);
    return 1.0 / (1.0 + dt_ * rate);
}

"""
)

HLLC_GLSL = (
    _FILM_CONST_GLSL
    + """\
// Velocity of a thin film. hq / h is singular as h falls: momentum flowing into
// a cell barely above dry_tol reached 26 000 m/s and stalled the CFL bound. The
// desingularised form (Kurganov & Petrova 2007) equals hq / h above FILM_DEPTHS
// times dry_tol and damps towards zero below it.
float safe_vel(float h_, float hq) {
    float e_ = FILM_DEPTHS * pc.dry_tol;
    return (h_ > pc.dry_tol) ? 2.0 * h_ * hq / (h_ * h_ + max(h_ * h_, e_ * e_)) : 0.0;
}

// Wave-speed bounds of the HLLC solver. A wet-wet face uses the Roe-average
// estimate. A face with a dry side uses Toro's dry-bed estimates
// (Shock-Capturing Methods for Free-Surface Shallow Flows, 2001, s. 10.5):
// the wet side's rarefaction tip runs at u - 2c (dry left) or u + 2c (dry
// right), twice what the Roe average gives, and the CFL bound must see it. The
// two blend continuously as a side goes from one to two dry_tol deep: a jump
// lets float32 rounding flip a whole flux and shift a front between runs.
void hllc_speeds(float hL, float unL, float hR, float unR, out float SL, out float SR) {
    float cL = sqrt(pc.g * max(hL, 0.0));
    float cR = sqrt(pc.g * max(hR, 0.0));

    float sqL = sqrt(max(hL, 0.0));
    float sqR = sqrt(max(hR, 0.0));
    float den  = sqL + sqR;
    float u_roe = (den > 1e-12) ? (sqL * unL + sqR * unR) / den : 0.0;
    float c_roe = sqrt(pc.g * 0.5 * (hL + hR));

    SL = min(unL - cL, u_roe - c_roe);
    SR = max(unR + cR, u_roe + c_roe);

    float wetL = clamp(hL / pc.dry_tol - 1.0, 0.0, 1.0);
    float wetR = clamp(hR / pc.dry_tol - 1.0, 0.0, 1.0);
    float dryLeft  = (1.0 - wetL) * wetR;
    float dryRight = (1.0 - wetR) * wetL;
    float roe = 1.0 - dryLeft - dryRight;
    SL = roe * SL + dryLeft * (unR - 2.0 * cR) + dryRight * (unL - cL);
    SR = roe * SR + dryLeft * (unR + cR) + dryRight * (unL + 2.0 * cL);
}

// HLLC flux in the face frame; Smax is the largest wave speed it assumed.
void hllc(float hL, float hqnL, float hqtL,
          float hR, float hqnR, float hqtR,
          out float Fh, out float Fhn, out float Fht, out float Smax)
{
    float unL = safe_vel(hL, hqnL);
    float utL = safe_vel(hL, hqtL);
    float unR = safe_vel(hR, hqnR);
    float utR = safe_vel(hR, hqtR);

    float SL, SR;
    hllc_speeds(hL, unL, hR, unR, SL, SR);
    Smax = max(abs(SL), abs(SR));

    // Contact wave speed (Toro 2001, Eq. 10.43)
    float num_sm = SL * hR * (unR - SR) - SR * hL * (unL - SL);
    float den_sm  = hR * (unR - SR) - hL * (unL - SL);
    float SM = (abs(den_sm) > 1e-12) ? num_sm / den_sm : 0.0;

    float FL_h  = hL * unL;
    float FL_hn = hL * unL * unL + 0.5 * pc.g * hL * hL;
    float FL_ht = hL * unL * utL;
    float FR_h  = hR * unR;
    float FR_hn = hR * unR * unR + 0.5 * pc.g * hR * hR;
    float FR_ht = hR * unR * utR;

    float dSL = SL - SM;
    float dSR = SR - SM;
    float kL = hL * (SL - unL) / ((abs(dSL) > 1e-14) ? dSL : 1e-14);
    float kR = hR * (SR - unR) / ((abs(dSR) > 1e-14) ? dSR : 1e-14);

    float FsL_h  = FL_h  + SL * (kL       - hL);
    float FsL_hn = FL_hn + SL * (kL * SM  - hL * unL);
    float FsL_ht = FL_ht + SL * (kL * utL - hL * utL);

    float FsR_h  = FR_h  + SR * (kR       - hR);
    float FsR_hn = FR_hn + SR * (kR * SM  - hR * unR);
    float FsR_ht = FR_ht + SR * (kR * utR - hR * utR);

    if      (SL >= 0.0) { Fh = FL_h;  Fhn = FL_hn;  Fht = FL_ht;  }
    else if (SM >= 0.0) { Fh = FsL_h; Fhn = FsL_hn; Fht = FsL_ht; }
    else if (SR >= 0.0) { Fh = FsR_h; Fhn = FsR_hn; Fht = FsR_ht; }
    else                { Fh = FR_h;  Fhn = FR_hn;  Fht = FR_ht;  }
}

// Face states of one edge: depths reconstructed to the face elevation
// (Audusse 2004 hydrostatic reconstruction; dzL/dzR are the precomputed
// face-minus-bed steps of the two sides), momenta rebuilt from the cells'
// velocities and rotated into the face frame.
void face_states(float hL, float huL, float hvL, float dzL,
                 float hR, float huR, float hvR, float dzR,
                 float nx_, float ny_,
                 out float hLs, out float hqnL, out float hqtL,
                 out float hRs, out float hqnR, out float hqtR)
{
    hLs = max(0.0, hL - dzL);
    hRs = max(0.0, hR - dzR);

    float uL = safe_vel(hL, huL);
    float vL = safe_vel(hL, hvL);
    float uR = safe_vel(hR, huR);
    float vR = safe_vel(hR, hvR);

    float huLs = hLs * uL;
    float hvLs = hLs * vL;
    float huRs = hRs * uR;
    float hvRs = hRs * vR;

    hqnL =  huLs * nx_ + hvLs * ny_;
    hqtL = -huLs * ny_ + hvLs * nx_;
    hqnR =  huRs * nx_ + hvRs * ny_;
    hqtR = -huRs * ny_ + hvRs * nx_;
}

// Largest wave speed the flux would assume on this edge: the CFL term.
float wave_bound(float hL, float huL, float hvL, float dzL,
                 float hR, float huR, float hvR, float dzR,
                 float nx_, float ny_)
{
    float hLs, hqnL, hqtL, hRs, hqnR, hqtR;
    face_states(hL, huL, hvL, dzL, hR, huR, hvR, dzR, nx_, ny_,
                hLs, hqnL, hqtL, hRs, hqnR, hqtR);
    float SL, SR;
    hllc_speeds(hLs, safe_vel(hLs, hqnL), hRs, safe_vel(hRs, hqnR), SL, SR);
    return max(abs(SL), abs(SR));
}

"""
)

# Open edges: the ghost state beyond edge ``-2 - cR`` and its mass flux. A
# boundary edge's bed steps are zero, so the ghost needs no reconstruction. Words
# per edge: condition, then its value (see the docstring of ``open_ghost``).
OPEN_BINDINGS = """\
layout(set=0, binding=19) buffer OPEN   { uint open_u[]; };
"""

OPEN_GHOST_GLSL = """\
const uint OPEN_WORDS = 2u;
const uint OPEN_WALL = 0u;
const uint OPEN_FREE = 1u;
const uint OPEN_NORMAL = 2u;
const uint OPEN_STAGE = 3u;
const uint OPEN_INFLOW = 4u;
const uint OPEN_NEWTON_STEPS = 4u;

uint open_kind(int cR) { return open_u[OPEN_WORDS * uint(-2 - cR)]; }
float open_value(int cR) { return uintBitsToFloat(open_u[OPEN_WORDS * uint(-2 - cR) + 1u]); }

// Inflow keeps its edge live while dry, or its volume would be lost.
bool open_forced(int cR) { return cR < -1 && open_kind(cR) == OPEN_INFLOW && open_value(cR) > 0.0; }

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

// The state beyond an open edge of condition `kind`, from the cell inside it.
// `value` is sqrt(S)/n for OPEN_NORMAL, the depth over this cell's bed for
// OPEN_STAGE and the discharge per unit width for OPEN_INFLOW.
void ghost_state(uint kind, float value, float hL, float huL, float hvL, float nx_, float ny_,
                 out float hG, out float huG, out float hvG)
{
    float uL = safe_vel(hL, huL);
    float vL = safe_vel(hL, hvL);
    float unL = uL * nx_ + vL * ny_;
    float ut = -uL * ny_ + vL * nx_;
    float cL = sqrt(pc.g * max(hL, 0.0));
    float un = -unL;
    hG = hL;
    if (kind == OPEN_FREE) {
        un = abs(unL);
    } else if (kind == OPEN_NORMAL) {
        un = value * pow(max(hL, 0.0), 2.0 / 3.0);
    } else if (kind == OPEN_STAGE) {
        // A level cannot reach a boundary against supercritical outflow.
        if (hL < pc.dry_tol || unL < cL) {
            hG = value;
            un = unL + 2.0 * (cL - sqrt(pc.g * value));
        } else {
            un = unL;
        }
    } else if (kind == OPEN_INFLOW && value > 0.0) {
        hG = inflow_depth(value, unL + 2.0 * cL);
        un = -value / hG;
        ut = 0.0;
    }
    huG = hG * (un * nx_ - ut * ny_);
    hvG = hG * (un * ny_ + ut * nx_);
}

void open_ghost(int cR, float hL, float huL, float hvL, float nx_, float ny_,
                out float hG, out float huG, out float hvG)
{
    ghost_state(open_kind(cR), open_value(cR), hL, huL, hvL, nx_, ny_, hG, huG, hvG);
}

bool outflow_only(int cR) {
    uint kind = open_kind(cR);
    return kind == OPEN_FREE || kind == OPEN_NORMAL;
}

// Mass flux through a boundary edge: a wall passes none, as FLUX_GLSL explains;
// outflow conditions never draw water in; inflow carries exactly its discharge.
float boundary_mass_flux(int cR, float Fh) {
    if (cR == -1) { return 0.0; }
    uint kind = open_kind(cR);
    if (kind == OPEN_FREE || kind == OPEN_NORMAL) { return max(Fh, 0.0); }
    if (kind == OPEN_STAGE) { return Fh; }
    if (kind == OPEN_INFLOW) { return -open_value(cR); }
    return 0.0;
}
"""

# An outflow-only edge whose flux would draw water in is a wall for that stage:
# its whole flux, momentum included, is redone against the wall ghost. Zeroing
# only Fh would leave the inward momentum flux of the outflow ghost in place.
OPEN_FLUX_MASS_GLSL = """\
    if (cR < -1 && outflow_only(cR) && Fh < 0.0) {
        ghost_state(OPEN_WALL, 0.0, hL, huL, hvL, nx_, ny_, hR, huR, hvR);
        face_states(hL, huL, hvL, dzL, hR, huR, hvR, dzR, nx_, ny_,
                    hLs, hqnL, hqtL, hRs, hqnR, hqtR);
        hllc(hLs, hqnL, hqtL, hRs, hqnR, hqtR, Fh, Fhn, Fht, Smax);
        Fh = 0.0;
    } else if (cR < 0) {
        Fh = boundary_mass_flux(cR, Fh);
    }
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

// (hi + loi) + inc as a renormalised pair, clamped at zero. `e += loi` rounds,
// so the pair is not error-free; the loss is ~ulp(loi), far below the float32
// flux sums that bound the measured drift.
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


EFT_GLSL = error_free_glsl(sys.platform)

# The flux kernel body: identical for every host-driven backend.
# Open edges' volume, per edge: out (hi, lo), in (hi, lo) as compensated pairs [m^3].
# Each edge has one writer, so the sum is deterministic without a gather.
OPEN_VOLUME_BINDINGS = """\
layout(set=0, binding=20) buffer OPENV  { float open_vol[]; };
"""

# The flux kernel body: identical for every host-driven backend. DT is the step's dt.
_FLUX_MAIN = (
    """\
void main() {
    uint eid = gl_GlobalInvocationID.x;
    if (eid >= pc.num_edges) return;

    int  cL   = ecL[eid];
    int  cR   = ecR[eid];
    float nx_ = enx[eid];
    float ny_ = eny[eid];
    float len = elen[eid];
    float dzL = edz[2u * eid];
    float dzR = edz[2u * eid + 1u];

    // Read from predictor state (h1/hu1/hv1) in stage 2
    float hL, huL, hvL;
    if (pc.stage < 0.5) {
        hL  = h[cL];
        huL = hu[cL];
        hvL = hv[cL];
    } else {
        hL  = h1[cL];
        huL = hu1[cL];
        hvL = hv1[cL];
    }

    float hR, huR, hvR;
    if (cR == -1) {
        // Wall (reflective) BC: mirror the left state with negated normal velocity.
        // The HLLC solver sees identical depths on both sides and zero net normal
        // flux, so no mass crosses the domain boundary.
        hR  = hL;
        huR = huL - 2.0 * (huL * nx_ + hvL * ny_) * nx_;
        hvR = hvL - 2.0 * (huL * nx_ + hvL * ny_) * ny_;
    } else if (cR < 0) {
        open_ghost(cR, hL, huL, hvL, nx_, ny_, hR, huR, hvR);
    } else {
        if (pc.stage < 0.5) {
            hR  = h[cR];
            huR = hu[cR];
            hvR = hv[cR];
        } else {
            hR  = h1[cR];
            huR = hu1[cR];
            hvR = hv1[cR];
        }
    }

    // Early exit: skip fully-dry edges, except an inflow that wets its cell.
    if (hL < pc.dry_tol && hR < pc.dry_tol && !open_forced(cR)) return;

    float hLs, hqnL, hqtL, hRs, hqnR, hqtR;
    face_states(hL, huL, hvL, dzL, hR, huR, hvR, dzR, nx_, ny_,
                hLs, hqnL, hqtL, hRs, hqnR, hqtR);

    float Fh, Fhn, Fht, Smax;
    hllc(hLs, hqnL, hqtL, hRs, hqnR, hqtR, Fh, Fhn, Fht, Smax);
    // A wall passes no mass. The mirrored Riemann problem gives zero only up
    // to float32 rounding, with a sign set by the flow direction, which over
    // many steps is a systematic leak.
"""
    + OPEN_FLUX_MASS_GLSL
    + """
    // Half of dt per Heun stage: the corrector average moves 0.5 dt (F0 + F1).
    if (cR < -1) {
        float v = 0.5 * DT * Fh * len;
        uint o = 4u * uint(-2 - cR) + ((v < 0.0) ? 2u : 0u);
        float hi, lo;
        add_compensated(open_vol[o], open_vol[o + 1u], abs(v), hi, lo);
        open_vol[o] = hi;
        open_vol[o + 1u] = lo;
    }

    float Fhu = Fhn * nx_ - Fht * ny_;
    float Fhv = Fhn * ny_ + Fht * nx_;

    // Audusse (2004) hydrostatic-reconstruction source term: compensates the
    // momentum imbalance introduced by clamping depths to the face elevation
    // (hLs/hRs above), so still water over a sloped/bumpy bed stays at rest
    // (the "well-balanced" / C-property fix). Each cell is corrected using
    // its OWN actual vs. reconstructed depth, added along its OWN outward
    // normal at this face: +n for cL (nx_,ny_ points cL->cR, i.e. outward
    // for cL), -n for cR. This is NOT symmetric like the raw flux — do not
    // "simplify" the two signs to match, that was tried and is wrong.
    float SL_g = 0.5 * pc.g * (hL * hL - hLs * hLs);
    float SR_g = 0.5 * pc.g * (hR * hR - hRs * hRs);

    atomicAdd(dh[cL],   Fh  * len);
    atomicAdd(dhu[cL],  Fhu * len + SL_g * nx_ * len);
    atomicAdd(dhv[cL],  Fhv * len + SL_g * ny_ * len);

    if (cR >= 0) {
        atomicAdd(dh[cR],  -Fh  * len);
        atomicAdd(dhu[cR], -Fhu * len - SR_g * nx_ * len);
        atomicAdd(dhv[cR], -Fhv * len - SR_g * ny_ * len);
    }
}

"""
)


def _flux(dt_field: str, dt_expr: str) -> str:
    return (
        "#version 450\n#extension GL_EXT_shader_atomic_float : require\n\n"
        "layout(local_size_x = 256) in;\n\n"
        + _BINDINGS
        + OPEN_BINDINGS
        + OPEN_VOLUME_BINDINGS
        + f"""
layout(push_constant) uniform PC {{
    uint  num_edges;
    float {dt_field};
    float g;
    float dry_tol;
    float cfl_number;
    float stage;
}} pc;

#define DT ({dt_expr})

"""
        + HLLC_GLSL
        + OPEN_GHOST_GLSL
        + EFT_GLSL
        + _FLUX_MAIN
    )


# dt arrives as a push constant (host-driven backends).
FLUX_GLSL = _flux("dt", "pc.dt")
# dt lives in DTBUF (device-side CFL backends).
FLUX_DTBUF_GLSL = _flux("dt_unused", "dt_buf[0]")
