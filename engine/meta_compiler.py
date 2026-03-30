from __future__ import annotations

from typing import Dict, List, Optional

import torch

from engine.supernet import LatentSupernet


class MetaCompiler:
    """Runtime NAS hook: high-entropy Sinkhorn routing ⇒ unmask a latent expert (shadow)."""

    def __init__(self, supernet: LatentSupernet) -> None:
        self.supernet = supernet

    def react_to_signals(
        self,
        signals_dict: Dict[str, torch.Tensor],
        supernet: LatentSupernet,
        max_unmasks: int = 1,
        *,
        block_thresholds: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        For each conditional block whose normalized routing entropy (scalar tensor) meets that
        block's threshold, unmask up to `max_unmasks` inactive experts as shadow. `.item()` is
        only used here, outside the compiled forward.
        """
        out: List[str] = []
        n = 0
        thr_map = block_thresholds or {}
        default_thr = 0.92
        for name, entropy_tensor in signals_dict.items():
            if n >= max_unmasks:
                break
            if entropy_tensor.dim() != 0:
                continue
            thr = thr_map.get(name, default_thr)
            if float(entropy_tensor.item()) < thr:
                continue
            un = supernet.unmask_next_inactive(shadow=True)
            if un is not None:
                out.append(un)
                n += 1
        return out
