"""
Enterprise Glass-Box firewall UI (Streamlit) over onyx_gateway.

Run from repo root (PowerShell):
  pip install -e ".[gateway]"
  streamlit run examples/enterprise_ui.py --server.fileWatcherType none
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

# --- Pure helpers (tests import these without running Streamlit) ---

_EX_DIR = Path(__file__).resolve().parent
LIVE_AUDIT_HTML = _EX_DIR / "live_audit.html"


def telemetry_row(signals: dict[str, float], trace: dict[str, Any]) -> dict[str, float]:
    """Flatten scan + explain for metrics / progress bars."""
    intent = float(trace.get("intent_risk", 0.0))
    intent_bar = max(0.0, min(1.0, (intent + 1.0) / 2.0))
    tox = float(signals["text_toxicity"]) / 0.3
    return {
        "pii": float(signals["has_pii_data"]),
        "competitor": float(signals["mentions_competitor"]),
        "toxicity": min(1.0, tox),
        "intent_risk_bar": intent_bar,
        "intent_risk_raw": intent,
        "approved": float(trace.get("is_approved", 0.0)),
    }


import streamlit as st

st.set_page_config(
    page_title="Axiom Enterprise Firewall",
    layout="wide",
    initial_sidebar_state="expanded",
)

from axiom.gateway.core import default_scan_text as scan_text
from onyx_gateway import build_trained_policy, chat_with_onyx


@st.cache_resource(show_spinner="Training Axiom compliance policy (one-time)...")
def _load_axiom_bundle() -> tuple[Any, str]:
    model, _block, source = build_trained_policy(epochs=50, lr=0.1)
    return model, source


def _inject_css() -> None:
    st.markdown(
        """
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a2332 100%);
        border-right: 1px solid #2d3a4d;
    }
    [data-testid="stSidebar"] * { color: #e8eef7 !important; }
    .block-container { padding-top: 1.2rem; }
</style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_css()
    st.title("Axiom + Onyx")
    st.caption("Neuro-symbolic compliance gate — sidebar shows tensor signals; Onyx only runs after approval.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_telemetry" not in st.session_state:
        st.session_state.last_telemetry = None

    model, policy_source = _load_axiom_bundle()

    with st.sidebar:
        st.markdown("### Axiom Compliance Engine")
        st.caption("Live signals: `scan_text` + `explain`")
        tel = st.session_state.last_telemetry
        if tel is None:
            st.metric("PII risk", "—")
            st.metric("Competitor risk", "—")
            st.metric("Text toxicity (norm.)", "—")
            st.metric("Neural intent (raw)", "—")
        else:
            st.metric("PII risk", f"{tel['pii']:.2f}")
            st.progress(tel["pii"], text="PII channel")
            st.metric("Competitor risk", f"{tel['competitor']:.2f}")
            st.progress(tel["competitor"], text="Competitor channel")
            st.metric("Text toxicity (norm.)", f"{tel['toxicity']:.2f}")
            st.progress(tel["toxicity"], text="Toxicity (÷0.3 cap 1)")
            st.metric("Neural intent (raw)", f"{tel['intent_risk_raw']:.3f}")
            st.progress(tel["intent_risk_bar"], text="Intent (display scale)")
            st.divider()
            st.metric("is_approved", f"{tel['approved']:.2f}")

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg.get("blocked"):
                st.error("AXIOM FIREWALL OVERRIDE: Policy violation — Onyx was not invoked.")
                st.warning(msg["content"])
                ap = msg.get("audit_path")
                if ap and Path(ap).is_file():
                    st.download_button(
                        label="Download compliance receipt (HTML)",
                        data=Path(ap).read_bytes(),
                        file_name="live_audit.html",
                        mime="text/html",
                        key=f"audit_dl_{i}",
                    )
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input("Message your Enterprise AI…"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        signals = scan_text(prompt)
        trace = model.explain(signals)
        st.session_state.last_telemetry = telemetry_row(signals, trace)

        audit_abs = str(LIVE_AUDIT_HTML.resolve())
        if float(trace["is_approved"]) < 0.5:
            st.toast("Policy violation — Onyx not called.", icon="🚨")
            model.export_report(signals, audit_abs, source_code=policy_source)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Message blocked by Compliance Engine.",
                    "blocked": True,
                    "audit_path": audit_abs,
                }
            )
        else:
            st.toast("Axiom approved — routing to Onyx.", icon="✅")
            reply = chat_with_onyx(
                model,
                policy_source,
                prompt,
                audit_path=LIVE_AUDIT_HTML,
                verbose=False,
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": reply or "[Onyx Mock]: Hello, I am your enterprise AI.",
                }
            )

        st.rerun()


if __name__ == "__main__":
    main()
