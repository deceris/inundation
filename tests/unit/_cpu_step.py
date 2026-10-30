"""Float64 reference of one Heun step, mirroring the GPU kernels.

The GPU tests compare a few fixed-dt steps against this implementation. It is
deliberately literal: the same reconstruction, the same HLLC, the same dry
tests and friction, in the same order, so a difference between the two is a
kernel defect or float32 rounding, never a modelling choice.

Where this file is ahead of the kernels it states the intended behaviour; the
GPU parity tests then go red until the kernels follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry

F64 = np.float64


@dataclass(frozen=True)
class State:
    h: NDArray[np.float64]
    hu: NDArray[np.float64]
    hv: NDArray[np.float64]


def _safe_vel(
    h: NDArray[np.float64], hq: NDArray[np.float64], dry_tol: float
) -> NDArray[np.float64]:
    wet = h > dry_tol
    return np.where(wet, hq / np.where(wet, h, 1.0), 0.0)


def hllc_speeds(
    hL: NDArray[np.float64],
    unL: NDArray[np.float64],
    hR: NDArray[np.float64],
    unR: NDArray[np.float64],
    *,
    g: float,
    dry_tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Left and right wave-speed bounds of the HLLC solver.

    Wet-wet faces use the Roe-average estimate. A face with a dry side uses
    Toro's dry-bed estimates (Shock-Capturing Methods for Free-Surface Shallow
    Flows, 2001, section 10.5): the wet side's rarefaction tip runs at
    ``u - 2c`` (dry left) or ``u + 2c`` (dry right). The two blend continuously
    as a side goes from one to two ``dry_tol`` deep: a jump would let float32
    rounding flip a whole flux.
    """
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))
    sqL = np.sqrt(np.maximum(hL, 0.0))
    sqR = np.sqrt(np.maximum(hR, 0.0))
    den = sqL + sqR
    u_roe = np.where(den > 1e-12, (sqL * unL + sqR * unR) / np.where(den > 1e-12, den, 1.0), 0.0)
    c_roe = np.sqrt(g * 0.5 * (hL + hR))
    s_l = np.minimum(unL - cL, u_roe - c_roe)
    s_r = np.maximum(unR + cR, u_roe + c_roe)
    wet_l = np.clip(hL / dry_tol - 1.0, 0.0, 1.0)
    wet_r = np.clip(hR / dry_tol - 1.0, 0.0, 1.0)
    dry_left = (1.0 - wet_l) * wet_r
    dry_right = (1.0 - wet_r) * wet_l
    roe = 1.0 - dry_left - dry_right
    return (
        roe * s_l + dry_left * (unR - 2.0 * cR) + dry_right * (unL - cL),
        roe * s_r + dry_left * (unR + cR) + dry_right * (unL + 2.0 * cL),
    )


def hllc(
    hL: NDArray[np.float64],
    hqnL: NDArray[np.float64],
    hqtL: NDArray[np.float64],
    hR: NDArray[np.float64],
    hqnR: NDArray[np.float64],
    hqtR: NDArray[np.float64],
    *,
    g: float,
    dry_tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """HLLC flux in the face frame: (Fh, Fhn, Fht, max |wave speed|)."""
    unL = _safe_vel(hL, hqnL, dry_tol)
    utL = _safe_vel(hL, hqtL, dry_tol)
    unR = _safe_vel(hR, hqnR, dry_tol)
    utR = _safe_vel(hR, hqtR, dry_tol)
    SL, SR = hllc_speeds(hL, unL, hR, unR, g=g, dry_tol=dry_tol)

    num_sm = SL * hR * (unR - SR) - SR * hL * (unL - SL)
    den_sm = hR * (unR - SR) - hL * (unL - SL)
    ok = np.abs(den_sm) > 1e-12
    SM = np.where(ok, num_sm / np.where(ok, den_sm, 1.0), 0.0)

    FL_h = hL * unL
    FL_hn = hL * unL * unL + 0.5 * g * hL * hL
    FL_ht = hL * unL * utL
    FR_h = hR * unR
    FR_hn = hR * unR * unR + 0.5 * g * hR * hR
    FR_ht = hR * unR * utR

    dSL = SL - SM
    dSR = SR - SM
    kL = hL * (SL - unL) / np.where(np.abs(dSL) > 1e-14, dSL, 1e-14)
    kR = hR * (SR - unR) / np.where(np.abs(dSR) > 1e-14, dSR, 1e-14)

    FsL_h = FL_h + SL * (kL - hL)
    FsL_hn = FL_hn + SL * (kL * SM - hL * unL)
    FsL_ht = FL_ht + SL * (kL * utL - hL * utL)
    FsR_h = FR_h + SR * (kR - hR)
    FsR_hn = FR_hn + SR * (kR * SM - hR * unR)
    FsR_ht = FR_ht + SR * (kR * utR - hR * utR)

    Fh = np.where(SL >= 0, FL_h, np.where(SM >= 0, FsL_h, np.where(SR >= 0, FsR_h, FR_h)))
    Fhn = np.where(SL >= 0, FL_hn, np.where(SM >= 0, FsL_hn, np.where(SR >= 0, FsR_hn, FR_hn)))
    Fht = np.where(SL >= 0, FL_ht, np.where(SM >= 0, FsL_ht, np.where(SR >= 0, FsR_ht, FR_ht)))
    return Fh, Fhn, Fht, np.maximum(np.abs(SL), np.abs(SR))


def _edge_states(
    geom: MeshGeometry, s: State, *, dry_tol: float
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
]:
    """Reconstructed face states (hLs, hqnL, hqtL, hRs, hqnR, hqtR, hL, hR, live)."""
    cL = geom.edge_cellL
    cR = geom.edge_cellR
    nx = geom.edge_nx.astype(F64)
    ny = geom.edge_ny.astype(F64)
    wall = cR < 0
    cRi = np.where(wall, cL, cR)
    zb = geom.zb.astype(F64)

    hL, huL, hvL, zbL = s.h[cL], s.hu[cL], s.hv[cL], zb[cL]
    hR, huR, hvR, zbR = s.h[cRi], s.hu[cRi], s.hv[cRi], zb[cRi]
    qn = huL * nx + hvL * ny
    huR = np.where(wall, huL - 2.0 * qn * nx, huR)
    hvR = np.where(wall, hvL - 2.0 * qn * ny, hvR)

    live = ~((hL < dry_tol) & (hR < dry_tol))
    z_face = np.maximum(zbL, zbR)
    hLs = np.maximum(0.0, hL + zbL - z_face)
    hRs = np.maximum(0.0, hR + zbR - z_face)
    uL, vL = _safe_vel(hL, huL, dry_tol), _safe_vel(hL, hvL, dry_tol)
    uR, vR = _safe_vel(hR, huR, dry_tol), _safe_vel(hR, hvR, dry_tol)
    huLs, hvLs, huRs, hvRs = hLs * uL, hLs * vL, hRs * uR, hRs * vR
    hqnL = huLs * nx + hvLs * ny
    hqtL = -huLs * ny + hvLs * nx
    hqnR = huRs * nx + hvRs * ny
    hqtR = -huRs * ny + hvRs * nx
    return hLs, hqnL, hqtL, hRs, hqnR, hqtR, hL, hR, live


def flux_divergence(
    geom: MeshGeometry, s: State, *, g: float, dry_tol: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per-cell sums of outward edge fluxes, with the hydrostatic correction."""
    nx = geom.edge_nx.astype(F64)
    ny = geom.edge_ny.astype(F64)
    length = geom.edge_len.astype(F64)
    hLs, hqnL, hqtL, hRs, hqnR, hqtR, hL, hR, live = _edge_states(geom, s, dry_tol=dry_tol)
    Fh, Fhn, Fht, _ = hllc(hLs, hqnL, hqtL, hRs, hqnR, hqtR, g=g, dry_tol=dry_tol)
    Fh = np.where(geom.edge_cellR < 0, 0.0, Fh)  # a wall passes no mass
    Fhu = Fhn * nx - Fht * ny
    Fhv = Fhn * ny + Fht * nx
    SL_g = 0.5 * g * (hL * hL - hLs * hLs)
    SR_g = 0.5 * g * (hR * hR - hRs * hRs)
    live_f = live.astype(F64)

    dh = np.zeros(geom.N, dtype=F64)
    dhu = np.zeros(geom.N, dtype=F64)
    dhv = np.zeros(geom.N, dtype=F64)
    cL, cR = geom.edge_cellL, geom.edge_cellR
    np.add.at(dh, cL, Fh * length * live_f)
    np.add.at(dhu, cL, (Fhu + SL_g * nx) * length * live_f)
    np.add.at(dhv, cL, (Fhv + SL_g * ny) * length * live_f)
    interior = cR >= 0
    np.add.at(dh, cR[interior], (-Fh * length * live_f)[interior])
    np.add.at(dhu, cR[interior], ((-Fhu - SR_g * nx) * length * live_f)[interior])
    np.add.at(dhv, cR[interior], ((-Fhv - SR_g * ny) * length * live_f)[interior])
    return dh, dhu, dhv


def cfl_dt(geom: MeshGeometry, s: State, *, g: float, dry_tol: float, cfl: float) -> float:
    """Smallest cell dt bound ``cfl * A / sum(max|S| * len)`` over live edges."""
    length = geom.edge_len.astype(F64)
    hLs, hqnL, hqtL, hRs, hqnR, hqtR, _hL, _hR, live = _edge_states(geom, s, dry_tol=dry_tol)
    _Fh, _Fhn, _Fht, smax = hllc(hLs, hqnL, hqtL, hRs, hqnR, hqtR, g=g, dry_tol=dry_tol)
    contrib = smax * length * live.astype(F64)
    lam = np.zeros(geom.N, dtype=F64)
    np.add.at(lam, geom.edge_cellL, contrib)
    interior = geom.edge_cellR >= 0
    np.add.at(lam, geom.edge_cellR[interior], contrib[interior])
    area = geom.area.astype(F64)
    ok = (lam > 1e-30) & (area > 1e-10)
    if not ok.any():
        return float("inf")
    return float(np.min(cfl * area[ok] / lam[ok]))


def _stage(
    geom: MeshGeometry,
    s: State,
    n: NDArray[np.float64],
    dt: float,
    *,
    g: float,
    dry_tol: float,
) -> State:
    """One forward-Euler stage: fluxes, positivity clamp, dry test, friction."""
    dh, dhu, dhv = flux_divergence(geom, s, g=g, dry_tol=dry_tol)
    dti = dt / geom.area.astype(F64)
    h_new = np.maximum(s.h - dti * dh, 0.0)
    hu_new = s.hu - dti * dhu
    hv_new = s.hv - dti * dhv
    wet = h_new >= dry_tol
    hu_new = np.where(wet, hu_new, 0.0)
    hv_new = np.where(wet, hv_new, 0.0)
    h_safe = np.where(wet, h_new, 1.0)
    u = hu_new / h_safe
    v = hv_new / h_safe
    spd = np.sqrt(u * u + v * v)
    Sf = g * n * n * spd / h_safe ** (4.0 / 3.0)
    den = np.where(wet, 1.0 + dt * Sf, 1.0)
    return State(h_new, hu_new / den, hv_new / den)


def heun_step(
    geom: MeshGeometry,
    s: State,
    n: NDArray[np.float64],
    dt: float,
    *,
    g: float,
    dry_tol: float,
) -> State:
    """SSP-RK2 (Heun) step. A cell that ends the step dry carries no momentum."""
    s1 = _stage(geom, s, n, dt, g=g, dry_tol=dry_tol)
    s2 = _stage(geom, s1, n, dt, g=g, dry_tol=dry_tol)
    h = 0.5 * (s.h + s2.h)
    hu = 0.5 * (s.hu + s2.hu)
    hv = 0.5 * (s.hv + s2.hv)
    wet = h >= dry_tol
    return State(h, np.where(wet, hu, 0.0), np.where(wet, hv, 0.0))


def run_steps(
    geom: MeshGeometry,
    h: NDArray[np.float32],
    hu: NDArray[np.float32],
    hv: NDArray[np.float32],
    n: NDArray[np.float32],
    dt: float,
    steps: int,
    *,
    g: float,
    dry_tol: float,
) -> State:
    s = State(h.astype(F64), hu.astype(F64), hv.astype(F64))
    for _ in range(steps):
        s = heun_step(geom, s, n.astype(F64), dt, g=g, dry_tol=dry_tol)
    return s


__all__ = ["State", "cfl_dt", "heun_step", "hllc", "hllc_speeds", "run_steps"]
