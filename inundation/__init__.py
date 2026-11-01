"""Inundation tool — GPU-accelerated shallow water equation flood simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .workflow import (
        FreeOutflow,
        Inflow,
        NormalDepth,
        OpenBoundary,
        PointSource,
        SimulationPhase,
        Stage,
        SWEWorkflow,
        Wall,
        WorkflowConfig,
        WorkflowResult,
        inflow_phases,
    )

__all__ = [
    "FreeOutflow",
    "Inflow",
    "NormalDepth",
    "OpenBoundary",
    "PointSource",
    "SWEWorkflow",
    "SimulationPhase",
    "Stage",
    "Wall",
    "WorkflowConfig",
    "WorkflowResult",
    "inflow_phases",
]

_SOLVER_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> object:
    if name in _SOLVER_EXPORTS:
        from . import workflow  # pyright: ignore[reportMissingModuleSource]

        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
