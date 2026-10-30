"""Update shader sources for SWE solver.

Both variants carry an optional clamp-mass diagnostic on binding 19. The
positivity clamp ``max(h_new, 0.0)`` is the solver's only non-conservative
operation on mass, and it is one-sided, so it shows up as a slow volume gain
that is otherwise indistinguishable from the arithmetic losses it competes
with. Accumulating it makes the two separable.

The weight is 0.5 on *both* RK stages, which is exact rather than approximate.
Writing the clamp deficits as c0 (predictor) and c1 (corrector), and using that
the flux divergence telescopes to zero over a closed domain:

    sum(A*h1)  = sum(A*h^n) + C0
    sum(A*h**) = sum(A*h1)  + C1
    V_{n+1}    = 0.5*(sum(A*h^n) + sum(A*h**)) = V_n + 0.5*(C0 + C1)

so 0.5*(C0 + C1) is precisely the volume the clamp added over the step.
"""

UPDATE_GLSL = """\
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
layout(set=0, binding=19) buffer CLAMPM { float clamp_mass[]; };

layout(push_constant) uniform PC {
    uint  num_cells;
    float dt;
    float g;
    float dry_tol;
    float cfl_number;
    float stage;
    float track_clamp;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.num_cells) return;

    float A   = area[i];
    float dt  = pc.dt;
    float dti = dt / A;

    // Stage 0 (predictor): advance from y_n (h/hu/hv)
    // Stage 1 (corrector): advance from y* (h1/hu1/hv1)
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

    float h_raw  = hi  - dti * dh[i];
    float hu_new = hui - dti * dhu[i];
    float hv_new = hvi - dti * dhv[i];

    float h_new = max(h_raw, 0.0);
    if (pc.track_clamp > 0.5) {
        clamp_mass[i] += 0.5 * A * (h_new - h_raw);
    }
    if (h_new < pc.dry_tol) { hu_new = 0.0; hv_new = 0.0; }
    // Keep the stored momentum consistent with the film velocity the flux sees.
    float film = 4.0 * pc.dry_tol;
    if (h_new < film) {
        float damp = 2.0 * h_new * h_new / (h_new * h_new + film * film);
        hu_new *= damp;
        hv_new *= damp;
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

    dh[i]  = 0.0;
    dhu[i] = 0.0;
    dhv[i] = 0.0;

    if (pc.stage < 0.5) {
        h1[i]  = h_new;
        hu1[i] = hu_new;
        hv1[i] = hv_new;
    } else {
        // Heun average: y_{n+1} = 0.5*(y_n + y**)
        // h[i] still holds y_n; h_new = y* + dt*f(y*) = y**
        float hf  = 0.5 * (h[i]  + h_new);
        float huf = 0.5 * (hu[i] + hu_new);
        float hvf = 0.5 * (hv[i] + hv_new);
        // A cell that ends the step dry carries no momentum. The stage results
        // zeroed theirs; averaging in the start-of-step value would leave half
        // of it behind, to re-enter the flow when the cell wets again.
        if (hf < pc.dry_tol) { huf = 0.0; hvf = 0.0; }
        h[i]  = hf;
        hu[i] = huf;
        hv[i] = hvf;
    }
}

"""

UPDATE_DTBUF_GLSL = """\
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

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.num_cells) return;

    float A   = area[i];
    float dt  = dt_buf[0];
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

    float h_raw  = hi  - dti * dh[i];
    float hu_new = hui - dti * dhu[i];
    float hv_new = hvi - dti * dhv[i];

    float h_new = max(h_raw, 0.0);
    if (pc.track_clamp > 0.5) {
        clamp_mass[i] += 0.5 * A * (h_new - h_raw);
    }
    if (h_new < pc.dry_tol) { hu_new = 0.0; hv_new = 0.0; }
    // Keep the stored momentum consistent with the film velocity the flux sees.
    float film = 4.0 * pc.dry_tol;
    if (h_new < film) {
        float damp = 2.0 * h_new * h_new / (h_new * h_new + film * film);
        hu_new *= damp;
        hv_new *= damp;
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

    dh[i]  = 0.0;
    dhu[i] = 0.0;
    dhv[i] = 0.0;

    if (pc.stage < 0.5) {
        h1[i]  = h_new;
        hu1[i] = hu_new;
        hv1[i] = hv_new;
    } else {
        float hf  = 0.5 * (h[i]  + h_new);
        float huf = 0.5 * (hu[i] + hu_new);
        float hvf = 0.5 * (hv[i] + hv_new);
        if (hf < pc.dry_tol) { huf = 0.0; hvf = 0.0; }
        h[i]  = hf;
        hu[i] = huf;
        hv[i] = hvf;
    }
}

"""
