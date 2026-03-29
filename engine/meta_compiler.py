from __future__ import annotations

from typing import Iterable, List, Optional

from engine.router import SinkhornRouter
from engine.supernet import LatentSupernet


class MetaCompiler:
    """Runtime NAS hook: high-entropy Sinkhorn routing ⇒ unmask a latent expert (shadow)."""

    def __init__(self, supernet: LatentSupernet) -> None:
        self.supernet = supernet

    def react_to_router_signals(
        self,
        routers: Iterable[SinkhornRouter],
        *,
        max_unmasks: int = 1,
    ) -> List[str]:
        """
        For each router whose last forward emitted `MutationSignal.triggered`, unmask up to
        `max_unmasks` inactive experts as shadow (sandbox).
        """
        out: List[str] = []
        n = 0
        for r in routers:
            if n >= max_unmasks:
                break
            sig = r.last_mutation_signal
            if sig is None or not sig.triggered:
                continue
            name = self.supernet.unmask_next_inactive(shadow=True)
            if name is not None:
                out.append(name)
                n += 1
        return out

    def on_mutation_signal(self, router: SinkhornRouter) -> Optional[str]:
        """Convenience: single-router reaction (first unmask if triggered)."""
        names = self.react_to_router_signals([router], max_unmasks=1)
        return names[0] if names else None
