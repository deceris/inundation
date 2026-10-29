"""Option #4 SWE solver: fixed-dt timestep batches between CFL checks.

This class is the base of ``SWESolverFixedDtBatchBarrier`` and no longer runs
on its own: it recorded dependent dispatches without compute barriers, so the
flux written by one dispatch was not guaranteed visible to the next, and the
results raced (``docs/planning/decisions.md``, "Vulkan reference
implementation"). ``run()`` refuses; use the barrier subclass or
``gpu_resident_batch``.
"""
# pyright: reportMissingImports=false,reportPrivateUsage=false,reportUnknownArgumentType=false,reportUnknownMemberType=false,reportUnknownVariableType=false,reportUnusedExpression=false

from __future__ import annotations

from typing import TYPE_CHECKING

from .solver import SWESolver

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray


class SWESolverFixedDtBatch(SWESolver):
    """Base of the barrier backend; its own unbarriered run() is closed."""

    def run(
        self,
        t_end: float,
        output_interval_s: float,
        dt_max: float,
        dt_init: float,
        cfl_interval: int,
        progress: bool = True,
        source_rate: NDArray[np.float32] | None = None,
        source_fn: Callable[[float], NDArray[np.float32]] | None = None,
        resume: bool = False,
        t_start: float = 0.0,
        capture_momentum_snapshots: bool = False,
        output_every_steps: int | None = None,
    ) -> (
        tuple[list[NDArray[np.float32]], list[float]]
        | tuple[
            list[NDArray[np.float32]],
            list[float],
            list[NDArray[np.float32]],
            list[NDArray[np.float32]],
        ]
    ):
        """Refuse: this backend recorded dependent dispatches without barriers and raced."""
        raise RuntimeError(
            "fixed_dt_batch records dependent dispatches without compute barriers and "
            "raced (docs/planning/decisions.md); use fixed_dt_batch_barrier or "
            "gpu_resident_batch"
        )


__all__ = ["SWESolverFixedDtBatch"]
