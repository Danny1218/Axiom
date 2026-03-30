"""Axiom: neural compiler / hybrid symbolic execution.

Public surface: ``load`` and ``AxiomModel``. Deeper imports (compiler, gateway, serve) are for
apps that opt into those subsystems; keep package root minimal for importers and future tooling.
"""

from .api import AxiomModel, load

__all__ = ["AxiomModel", "load"]
