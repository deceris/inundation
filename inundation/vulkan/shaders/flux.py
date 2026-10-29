"""Flux shader sources for SWE solver."""

# Shared by both flux variants. Binding 19 holds 4 floats per edge:
# (kind, value, volume-out coarse + fine sums [m^3]). kind 0 = reflective
# wall, 1 = prescribed outward unit discharge q_n [m^2/s], 2 = prescribed depth
# [m]. Only read for boundary edges (cR < 0); see inundation/boundary.py.
_OPEN_BC_GLSL = """
layout(set=0, binding=19) buffer BC     { float bc[];     };

// One thread per edge owns its slots, so no atomics. Two-level sum: a long
// steady run adds ~1e5 tiny increments, which a bare float32 total silently
// drops once they fall below half an ULP. `fine` stays small, so increments
// keep full precision; it is folded into `coarse` once it reaches 2^-12 of
// it. Kahan would be shorter, but fast-math (MoltenVK's default) folds its
// (t - s) - y to zero, and `precise` makes the whole shader ~50 % slower.
void accumulate_volume(uint eid, float dv) {
    float coarse = bc[4u * eid + 2u];
    float fine   = bc[4u * eid + 3u] + dv;
    if (abs(fine) >= abs(coarse) * (1.0 / 4096.0)) {
        bc[4u * eid + 2u] = coarse + fine;
        fine = 0.0;
    }
    bc[4u * eid + 3u] = fine;
}

// Prescribed normal discharge: the mass flux is imposed exactly (no Riemann
// solve). Outflow (q >= 0) upwinds the interior state and is capped at the
// volume the cell holds this stage; inflow (q < 0) enters normal to the edge
// at max(interior depth, critical depth), so a dry cell can be filled.
void discharge_flux(float hL, float hqtL, float q, float q_cap,
                    out float Fh, out float Fhn, out float Fht)
{
    if (q >= 0.0) {
        float qo = min(q, q_cap);
        float un = (hL > pc.dry_tol) ? qo / hL : 0.0;
        Fh  = qo;
        Fhn = qo * un + 0.5 * pc.g * hL * hL;
        Fht = qo * safe_vel(hL, hqtL);
    } else {
        float hf = max(hL, pow(q * q / pc.g, 1.0 / 3.0));
        Fh  = q;
        Fhn = q * q / hf + 0.5 * pc.g * hf * hf;
        Fht = 0.0;
    }
}
"""

FLUX_GLSL = (
    """\
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

layout(push_constant) uniform PC {
    float num_edges;
    float dt;
    float g;
    float dry_tol;
    float cfl_number;
    float stage;
} pc;

float safe_vel(float h_, float hq) {
    return (h_ > pc.dry_tol) ? hq / h_ : 0.0;
}

void hllc(float hL, float hqnL, float hqtL,
          float hR, float hqnR, float hqtR,
          out float Fh, out float Fhn, out float Fht)
{
    float unL = safe_vel(hL, hqnL);
    float utL = safe_vel(hL, hqtL);
    float unR = safe_vel(hR, hqnR);
    float utR = safe_vel(hR, hqtR);

    float cL = sqrt(pc.g * max(hL, 0.0));
    float cR = sqrt(pc.g * max(hR, 0.0));

    float sqL = sqrt(max(hL, 0.0));
    float sqR = sqrt(max(hR, 0.0));
    float den  = sqL + sqR;
    float u_roe = (den > 1e-12) ? (sqL * unL + sqR * unR) / den : 0.0;
    float c_roe = sqrt(pc.g * 0.5 * (hL + hR));

    float SL = min(unL - cL, u_roe - c_roe);
    float SR = max(unR + cR, u_roe + c_roe);

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
"""
    + _OPEN_BC_GLSL
    + """
void main() {
    uint eid = gl_GlobalInvocationID.x;
    if (eid >= uint(pc.num_edges)) return;

    int  cL   = ecL[eid];
    int  cR   = ecR[eid];
    float nx_ = enx[eid];
    float ny_ = eny[eid];
    float len = elen[eid];
    int   bkind = 0;
    float bval  = 0.0;
    if (cR < 0) {
        bkind = int(bc[4u * eid]);
        bval  = bc[4u * eid + 1u];
    }

    // Early exit: skip fully-dry edges (both sides below dry tolerance).
    // An open boundary's exterior is the prescribed depth, or "wet" for inflow.
    float hL_check = (pc.stage < 0.5) ? h[cL] : h1[cL];
    float hR_check = (cR >= 0) ? ((pc.stage < 0.5) ? h[cR] : h1[cR])
                   : (bkind == 2) ? bval
                   : (bkind == 1 && bval < 0.0) ? 1.0 : hL_check;
    if (hL_check < pc.dry_tol && hR_check < pc.dry_tol) return;

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
    float zbL = zb[cL];

    float hR, huR, hvR, zbR;
    if (cR < 0 && bkind == 2) {
        // Fixed-depth ghost state, interior velocity (not a characteristic BC).
        hR  = bval;
        huR = bval * safe_vel(hL, huL);
        hvR = bval * safe_vel(hL, hvL);
        zbR = zbL;
    } else if (cR < 0) {
        // Wall (reflective) BC: mirror the left state with negated normal velocity.
        // The HLLC solver sees identical depths on both sides and zero net normal
        // flux, so no mass crosses the domain boundary. Discharge edges reuse it
        // only for the (zero) hydrostatic correction; their flux is set below.
        hR  = hL;
        huR = huL - 2.0 * (huL * nx_ + hvL * ny_) * nx_;
        hvR = hvL - 2.0 * (huL * nx_ + hvL * ny_) * ny_;
        zbR = zbL;
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
        zbR = zb[cR];
    }

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
    if (bkind == 1) {
        discharge_flux(hLs, hqtL, bval, hLs * area[cL] / (len * max(pc.dt, 1e-30)), Fh, Fhn, Fht);
    } else {
        hllc(hLs, hqnL, hqtL, hRs, hqnR, hqtR, Fh, Fhn, Fht);
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
    } else if (bkind != 0) {
        // Half-dt per Heun
        // stage sums to exactly the volume the corrector average removes.
        accumulate_volume(eid, 0.5 * pc.dt * Fh * len);
    }
}

"""
)

FLUX_DTBUF_GLSL = (
    """\
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

layout(push_constant) uniform PC {
    float num_edges;
    float dt_unused;
    float g;
    float dry_tol;
    float cfl_number;
    float stage;
} pc;

float safe_vel(float h_, float hq) {
    return (h_ > pc.dry_tol) ? hq / h_ : 0.0;
}

void hllc(float hL, float hqnL, float hqtL,
          float hR, float hqnR, float hqtR,
          out float Fh, out float Fhn, out float Fht)
{
    float unL = safe_vel(hL, hqnL);
    float utL = safe_vel(hL, hqtL);
    float unR = safe_vel(hR, hqnR);
    float utR = safe_vel(hR, hqtR);

    float cL = sqrt(pc.g * max(hL, 0.0));
    float cR = sqrt(pc.g * max(hR, 0.0));

    float sqL = sqrt(max(hL, 0.0));
    float sqR = sqrt(max(hR, 0.0));
    float den  = sqL + sqR;
    float u_roe = (den > 1e-12) ? (sqL * unL + sqR * unR) / den : 0.0;
    float c_roe = sqrt(pc.g * 0.5 * (hL + hR));

    float SL = min(unL - cL, u_roe - c_roe);
    float SR = max(unR + cR, u_roe + c_roe);

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
"""
    + _OPEN_BC_GLSL
    + """
void main() {
    uint eid = gl_GlobalInvocationID.x;
    if (eid >= uint(pc.num_edges)) return;

    int  cL   = ecL[eid];
    int  cR   = ecR[eid];
    float nx_ = enx[eid];
    float ny_ = eny[eid];
    float len = elen[eid];
    float dt  = dt_buf[0];
    int   bkind = 0;
    float bval  = 0.0;
    if (cR < 0) {
        bkind = int(bc[4u * eid]);
        bval  = bc[4u * eid + 1u];
    }

    float hL_check = (pc.stage < 0.5) ? h[cL] : h1[cL];
    float hR_check = (cR >= 0) ? ((pc.stage < 0.5) ? h[cR] : h1[cR])
                   : (bkind == 2) ? bval
                   : (bkind == 1 && bval < 0.0) ? 1.0 : hL_check;
    if (hL_check < pc.dry_tol && hR_check < pc.dry_tol) return;

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
    float zbL = zb[cL];

    float hR, huR, hvR, zbR;
    if (cR < 0 && bkind == 2) {
        hR  = bval;
        huR = bval * safe_vel(hL, huL);
        hvR = bval * safe_vel(hL, hvL);
        zbR = zbL;
    } else if (cR < 0) {
        hR  = hL;
        huR = huL - 2.0 * (huL * nx_ + hvL * ny_) * nx_;
        hvR = hvL - 2.0 * (huL * nx_ + hvL * ny_) * ny_;
        zbR = zbL;
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
        zbR = zb[cR];
    }

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
    if (bkind == 1) {
        discharge_flux(hLs, hqtL, bval, hLs * area[cL] / (len * max(dt, 1e-30)), Fh, Fhn, Fht);
    } else {
        hllc(hLs, hqnL, hqtL, hRs, hqnR, hqtR, Fh, Fhn, Fht);
    }

    float Fhu = Fhn * nx_ - Fht * ny_;
    float Fhv = Fhn * ny_ + Fht * nx_;

    // Audusse (2004) hydrostatic-reconstruction source term — see FLUX_GLSL
    // above for the derivation. Kept identical between the two shader
    // variants (this one only differs in reading dt from DTBUF).
    float SL_g = 0.5 * pc.g * (hL * hL - hLs * hLs);
    float SR_g = 0.5 * pc.g * (hR * hR - hRs * hRs);

    atomicAdd(dh[cL],   Fh  * len);
    atomicAdd(dhu[cL],  Fhu * len + SL_g * nx_ * len);
    atomicAdd(dhv[cL],  Fhv * len + SL_g * ny_ * len);

    if (cR >= 0) {
        atomicAdd(dh[cR],  -Fh  * len);
        atomicAdd(dhu[cR], -Fhu * len - SR_g * nx_ * len);
        atomicAdd(dhv[cR], -Fhv * len - SR_g * ny_ * len);
    } else if (bkind != 0) {
        accumulate_volume(eid, 0.5 * dt * Fh * len);
    }
}

"""
)
