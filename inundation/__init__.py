"""Inundation tool — GPU-accelerated shallow water equation flood simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .boundary import OpenBoundary
    from .workflow import (
        PointSource,
        SimulationPhase,
        SWEWorkflow,
        WorkflowConfig,
        WorkflowResult,
    )

__all__ = [
    "OpenBoundary",
    "PointSource",
    "SWEWorkflow",
    "SimulationPhase",
    "WorkflowConfig",
    "WorkflowResult",
]

_SOLVER_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> object:
    if name == "OpenBoundary":
        from .boundary import OpenBoundary

        return OpenBoundary
    if name in _SOLVER_EXPORTS:
        from .workflow import (  # pyright: ignore[reportMissingModuleSource]
            PointSource,
            SimulationPhase,
            SWEWorkflow,
            WorkflowConfig,
            WorkflowResult,
        )

        return {
            "PointSource": PointSource,
            "SimulationPhase": SimulationPhase,
            "SWEWorkflow": SWEWorkflow,
            "WorkflowConfig": WorkflowConfig,
            "WorkflowResult": WorkflowResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
