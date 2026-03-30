"""Register and resolve :class:`~axiom.experts.base.SemanticExpert` implementations by name."""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

from axiom.experts.base import SemanticExpert


class DuplicateExpertRegistrationError(ValueError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Expert backend {name!r} is already registered")
        self.name = name


class UnknownExpertError(KeyError):
    def __init__(self, name: str) -> None:
        super().__init__(f"No expert backend registered as {name!r}")
        self.name = name


_registry: Dict[str, SemanticExpert] = {}


def register(name: str, expert: SemanticExpert, *, allow_replace: bool = False) -> None:
    """Register ``expert`` under ``name``. Raises :exc:`DuplicateExpertRegistrationError` unless ``allow_replace``."""
    if name in _registry and not allow_replace:
        raise DuplicateExpertRegistrationError(name)
    _registry[name] = expert


def unregister(name: str) -> None:
    """Remove ``name`` if present (no-op if missing)."""
    _registry.pop(name, None)


def resolve(name: str) -> SemanticExpert:
    """Return the expert registered as ``name``. Raises :exc:`UnknownExpertError` if absent."""
    try:
        return _registry[name]
    except KeyError:
        raise UnknownExpertError(name) from None


def registered_names() -> Tuple[str, ...]:
    """Snapshot of registered backend names (sorted)."""
    return tuple(sorted(_registry))


def iter_registered() -> Iterator[Tuple[str, SemanticExpert]]:
    """Yield ``(name, expert)`` pairs in sorted name order."""
    for n in sorted(_registry):
        yield n, _registry[n]


def clear_registry() -> None:
    """Remove all registrations (intended for tests)."""
    _registry.clear()
