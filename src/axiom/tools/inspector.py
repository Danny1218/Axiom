"""Streamlit Glass Box: load an Axiom bundle, run inference with signal trace, render the DAG."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch

from axiom.compiler.deserializer import load_execution_bundle
from axiom.engine.inference import AxiomRunner
from axiom.tools.glass_box import execution_graph_to_graphviz, routing_trace_entries, tensor_preview_dict

_UPLOAD_DIR = Path(__file__).resolve().parent / ".axiom_inspector_upload"


def _load_graph_from_prefix(prefix: str):
    p = prefix.strip().strip('"').strip("'")
    if not p:
        return None, "Empty path."
    try:
        g = load_execution_bundle(p)
        return g, None
    except FileNotFoundError as e:
        return None, f"Bundle files not found ({e}). Use prefix without extension (e.g. `axiom_bundle`)."
    except Exception as e:
        return None, str(e)


def main() -> None:
    st.set_page_config(page_title="Axiom Glass Box", layout="wide")
    st.title("Glass Box Visualizer")
    st.caption("Interpretable trace: execution DAG, Sinkhorn entropy, and router weights.")

    if "graph" not in st.session_state:
        st.session_state.graph = None
        st.session_state.runner = None
        st.session_state.load_error = None

    with st.sidebar:
        st.header("Bundle")
        path_in = st.text_input(
            "Path prefix (no extension)",
            value="axiom_bundle",
            help="Expects `{prefix}_topology.json` and `{prefix}.pt` next to each other.",
        )
        if st.button("Load bundle"):
            g, err = _load_graph_from_prefix(path_in)
            st.session_state.graph = g
            st.session_state.runner = AxiomRunner(g) if g is not None else None
            st.session_state.load_error = err
            st.session_state.pop("last_out", None)
            st.session_state.pop("last_sig", None)

        st.divider()
        st.caption("Upload both files; names become `upload_topology.json` + `upload.pt`.")
        up_json = st.file_uploader("Topology JSON", type=["json"])
        up_pt = st.file_uploader("Weights .pt", type=["pt", "pth"])
        if st.button("Load uploaded files") and up_json is not None and up_pt is not None:
            tmp = _UPLOAD_DIR
            tmp.mkdir(exist_ok=True)
            (tmp / "upload_topology.json").write_bytes(up_json.getvalue())
            (tmp / "upload.pt").write_bytes(up_pt.getvalue())
            g, err = _load_graph_from_prefix(str(tmp / "upload"))
            st.session_state.graph = g
            st.session_state.runner = AxiomRunner(g) if g is not None else None
            st.session_state.load_error = err
            st.session_state.pop("last_out", None)
            st.session_state.pop("last_sig", None)

        if st.session_state.load_error:
            st.error(st.session_state.load_error)
        elif st.session_state.graph is not None:
            st.success("Graph loaded.")

    graph = st.session_state.graph
    runner = st.session_state.runner

    if graph is None:
        st.info("Load an Axiom execution bundle from the sidebar to begin.")
        return

    abi = getattr(graph, "abi", {}) or {}
    st.subheader("Inputs (from ABI)")
    inputs: dict[str, float] = {}
    cols = st.columns(min(4, max(1, len(abi))))
    for i, (name, col) in enumerate(sorted(abi.items(), key=lambda kv: kv[1])):
        with cols[i % len(cols)]:
            inputs[name] = float(st.number_input(name, value=0.0, key=f"in_{name}"))

    dev = st.radio("Device", ("cpu", "cuda"), horizontal=True, key="device")
    if dev == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA not available; inference will fail if you choose cuda.")
    if st.button("Run inference", type="primary"):
        with st.spinner("Running graph…"):
            out, sig = runner.predict_with_signals(inputs, device=dev)
        st.session_state.last_out = out
        st.session_state.last_sig = sig

    dot = execution_graph_to_graphviz(graph)
    st.subheader("Execution DAG")
    st.graphviz_chart(dot)

    if st.session_state.get("last_out") is not None:
        out = st.session_state.last_out
        sig = st.session_state.last_sig or {}
        st.markdown("---")
        st.subheader("Output")
        prev = tensor_preview_dict(out)
        flat = prev["flat_head"]
        primary = flat[0] if flat else float("nan")
        st.metric(
            label="Primary output value (first trunk element, batch 0)",
            value=f"{primary:.6f}",
            help=f"Full shape {prev['shape']}; showing first scalar for a compact headline.",
        )
        with st.expander("Full output tensor (preview)", expanded=False):
            st.json(prev)

        rows = routing_trace_entries(graph, sig)
        with st.expander("Routing trace", expanded=True):
            if not rows:
                st.write("No `ConditionalSinkhornBlock` nodes in this graph.")
            for r in rows:
                st.markdown(f"**{r['block']}**  (`{r['expert_then']}` vs `{r['expert_else']}`)")
                st.write(
                    f"Normalized routing entropy: **{r['normalized_routing_entropy']}**"
                    if r["normalized_routing_entropy"] is not None
                    else "Entropy: —"
                )
                mw = r["mean_router_weights_then_else"]
                if mw is not None and len(mw) >= 2:
                    st.write(
                        f"Active path weights (batch-mean): **{r['expert_then']}** = {mw[0]:.4f}, "
                        f"**{r['expert_else']}** = {mw[1]:.4f}"
                    )
                st.divider()


if __name__ == "__main__":
    main()
