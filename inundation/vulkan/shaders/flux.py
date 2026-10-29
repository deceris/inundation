"""Flux shader sources for SWE solver.

``HLLC_GLSL`` holds the face reconstruction, the HLLC Riemann solver and the
wave-speed bound shared by every flux and CFL kernel; it needs a push-constant
block with ``g`` and ``dry_tol``. The two kernels here differ only in the name
of their unused dt slot, so the flux and CFL kernels of every backend run the
same arithmetic.
"""

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
layout(set=0, binding=17) buffer DTBUF  { float dt_buf[]; };
layout(set=0, binding=18) buffer CFLSC  { float cfl_scratch[]; };
"""

# Riemann solver and face reconstruction shared by every flux and CFL kernel.
HLLC_GLSL = """\
float safe_vel(float h_, float hq) {
    return (h_ > pc.dry_tol) ? hq / h_ : 0.0;
}

// Wave-speed bounds of the HLLC solver. A wet-wet face uses the Roe-average
// estimate. A face with a dry side uses Toro's dry-bed estimates
// (Shock-Capturing Methods for Free-Surface Shallow Flows, 2001, s. 10.5):
// the wet side's rarefaction tip runs at u - 2c (dry left) or u + 2c (dry
// right), twice what the Roe average gives, and the CFL bound must see it.
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

    bool dryL = hL < pc.dry_tol;
    bool dryR = hR < pc.dry_tol;
    if (dryL && !dryR) { SL = unR - 2.0 * cR; SR = unR + cR; }
    if (dryR && !dryL) { SL = unL - cL; SR = unL + 2.0 * cL; }
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
// (Audusse 2004 hydrostatic reconstruction), momenta rebuilt from the cells'
// velocities and rotated into the face frame.
void face_states(float hL, float huL, float hvL, float zbL,
                 float hR, float huR, float hvR, float zbR,
                 float nx_, float ny_,
                 out float hLs, out float hqnL, out float hqtL,
                 out float hRs, out float hqnR, out float hqtR)
{
    float z_face = max(zbL, zbR);
    hLs = max(0.0, hL + zbL - z_face);
    hRs = max(0.0, hR + zbR - z_face);

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
float wave_bound(float hL, float huL, float hvL, float zbL,
                 float hR, float huR, float hvR, float zbR,
                 float nx_, float ny_)
{
    float hLs, hqnL, hqtL, hRs, hqnR, hqtR;
    face_states(hL, huL, hvL, zbL, hR, huR, hvR, zbR, nx_, ny_,
                hLs, hqnL, hqtL, hRs, hqnR, hqtR);
    float SL, SR;
    hllc_speeds(hLs, safe_vel(hLs, hqnL), hRs, safe_vel(hRs, hqnR), SL, SR);
    return max(abs(SL), abs(SR));
}

"""

# The flux kernel body: identical for every host-driven backend.
_FLUX_MAIN = """\
void main() {
    uint eid = gl_GlobalInvocationID.x;
    if (eid >= uint(pc.num_edges)) return;

    int  cL   = ecL[eid];
    int  cR   = ecR[eid];
    float nx_ = enx[eid];
    float ny_ = eny[eid];
    float len = elen[eid];

    // Early exit: skip fully-dry edges (both sides below dry tolerance)
    float hL_check = (pc.stage < 0.5) ? h[cL] : h1[cL];
    float hR_check = (cR >= 0) ? ((pc.stage < 0.5) ? h[cR] : h1[cR]) : hL_check;
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
    if (cR < 0) {
        // Wall (reflective) BC: mirror the left state with negated normal velocity.
        // The HLLC solver sees identical depths on both sides and zero net normal
        // flux, so no mass crosses the domain boundary.
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

    float hLs, hqnL, hqtL, hRs, hqnR, hqtR;
    face_states(hL, huL, hvL, zbL, hR, huR, hvR, zbR, nx_, ny_,
                hLs, hqnL, hqtL, hRs, hqnR, hqtR);

    float Fh, Fhn, Fht, Smax;
    hllc(hLs, hqnL, hqtL, hRs, hqnR, hqtR, Fh, Fhn, Fht, Smax);

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


def _flux(dt_field: str) -> str:
    return (
        "#version 450\n#extension GL_EXT_shader_atomic_float : require\n\n"
        "layout(local_size_x = 256) in;\n\n"
        + _BINDINGS
        + f"""
layout(push_constant) uniform PC {{
    float num_edges;
    float {dt_field};
    float g;
    float dry_tol;
    float cfl_number;
    float stage;
}} pc;

"""
        + HLLC_GLSL
        + _FLUX_MAIN
    )


# dt arrives as a push constant (host-driven backends).
FLUX_GLSL = _flux("dt")
# dt lives in DTBUF (device-side CFL backends); the kernel does not read it.
FLUX_DTBUF_GLSL = _flux("dt_unused")
