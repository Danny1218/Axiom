"""Contenders: Axiom tolerant inference vs sklearn regressors."""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from axiom.api import AxiomModel
from axiom.copilot.evaluator import _try_compile
from axiom.copilot.models import ProgramCandidate
from axiom.copilot.search import CopilotSearchConfig
from axiom.copilot.tolerant_inference import try_tolerant_symbolic_inference
from axiom.experts.base import ExpertDraftRequest, SemanticExpert
from benchmarks.baseline_showdown.harness import (
    GLOBAL_SEED,
    ContenderResult,
    Timer,
    rmse,
    rows_to_matrix,
)
from benchmarks.baseline_showdown.tasks import SynthTask


class _NoExpert(SemanticExpert):
    def draft_program(self, request: ExpertDraftRequest):
        raise RuntimeError("LLM must not be called in benchmark")

    def repair_program(self, request):
        raise RuntimeError("repair must not run")

    def summarize_trace(self, request):
        return ""


class Contender(Protocol):
    name: str

    def run(
        self,
        task: SynthTask,
        train_rows: Sequence[Dict[str, float]],
        train_y: Sequence[float],
        interp_rows: Sequence[Dict[str, float]],
        interp_y: Sequence[float],
        extrap_rows: Sequence[Dict[str, float]],
        extrap_y: Sequence[float],
    ) -> ContenderResult: ...


def _predict_ax(source: str, rows: Sequence[Dict[str, float]], out_key: str) -> List[float]:
    _, block, failures, _ = _try_compile(source, max_unroll=8)
    if block is None or failures:
        return [float("nan")] * len(rows)
    model = AxiomModel(block)
    preds = model.predict([dict(r) for r in rows])
    if not isinstance(preds, list):
        preds = [preds]
    return [float(p.get(out_key, float("nan"))) for p in preds]


def _predict_sklearn(model: object, rows: Sequence[Dict[str, float]], keys: Sequence[str]) -> List[float]:
    x = rows_to_matrix(rows, keys)
    pred = getattr(model, "predict")(x)
    return [float(v) for v in pred]


class SklearnContender:
    def __init__(self, name: str, factory) -> None:
        self.name = name
        self._factory = factory

    def run(
        self,
        task: SynthTask,
        train_rows: Sequence[Dict[str, float]],
        train_y: Sequence[float],
        interp_rows: Sequence[Dict[str, float]],
        interp_y: Sequence[float],
        extrap_rows: Sequence[Dict[str, float]],
        extrap_y: Sequence[float],
    ) -> ContenderResult:
        keys = task.input_names
        x_train = rows_to_matrix(train_rows, keys)
        y_train = np.asarray(train_y, dtype=np.float64)
        with Timer() as t:
            model = self._factory()
            model.fit(x_train, y_train)
            ip = _predict_sklearn(model, interp_rows, keys)
            ep = _predict_sklearn(model, extrap_rows, keys)
        return ContenderResult(
            name=self.name,
            declined=False,
            interp_rmse=rmse(ip, interp_y),
            extrap_rmse=rmse(ep, extrap_y),
            wall_ms=t.elapsed_ms,
        )


class AxiomContender:
    name = "axiom"

    def run(
        self,
        task: SynthTask,
        train_rows: Sequence[Dict[str, float]],
        train_y: Sequence[float],
        interp_rows: Sequence[Dict[str, float]],
        interp_y: Sequence[float],
        extrap_rows: Sequence[Dict[str, float]],
        extrap_y: Sequence[float],
    ) -> ContenderResult:
        from benchmarks.baseline_showdown.harness import TaskSplit

        split = TaskSplit(list(train_rows), list(train_y), [], [], [], [])
        inp, exp = task.training_examples(split)
        config = CopilotSearchConfig(
            goal=task.goal,
            expert=_NoExpert(),
            mode="predict_rows",
            example_input_rows=inp,
            expected_rows=exp,
            max_iterations=1,
        )
        with Timer() as t:
            draft = try_tolerant_symbolic_inference(config)
            if draft is None:
                return ContenderResult(
                    name=self.name,
                    declined=True,
                    interp_rmse=None,
                    extrap_rmse=None,
                    wall_ms=t.elapsed_ms,
                    note="tolerant inference declined",
                )
            source = draft.ax_source
            ip = _predict_ax(source, interp_rows, task.output_name)
            ep = _predict_ax(source, extrap_rows, task.output_name)
        formula_match: Optional[bool] = None
        if task.in_family and task.expected_coeffs:
            formula_match = task.coeffs_match(source)
        elif not task.in_family:
            formula_match = False if not draft.metadata.get("fast_path") else None
        return ContenderResult(
            name=self.name,
            declined=False,
            interp_rmse=rmse(ip, interp_y),
            extrap_rmse=rmse(ep, extrap_y),
            wall_ms=t.elapsed_ms,
            recovered_source=source,
            formula_match=formula_match,
            metadata=dict(draft.metadata or {}),
        )


def build_contenders() -> List[Contender]:
    return [
        AxiomContender(),
        SklearnContender("linear", lambda: LinearRegression()),
        SklearnContender(
            "mlp",
            lambda: MLPRegressor(
                hidden_layer_sizes=(64, 64),
                max_iter=2000,
                random_state=GLOBAL_SEED,
                early_stopping=False,
            ),
        ),
        SklearnContender("gbr", lambda: GradientBoostingRegressor(random_state=GLOBAL_SEED)),
    ]
