"""Phase 44: standalone HTML report from ``explain`` + ``predict`` (Glass Box)."""

from __future__ import annotations

import json
import os
from html import escape
from typing import Any, Dict, Optional, Union

Jsonish = Union[float, int, str, bool, None, list, dict]


def _is_adapter_key(name: str) -> bool:
    low = name.lower()
    return any(s in low for s in ("alpha", "neural", "prediction"))


def _format_cell_value(v: Jsonish) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, indent=0, default=str)
    return str(v)


def export_html_report(
    model: Any,
    data: dict,
    output_path: str,
    source_code: Optional[str] = None,
) -> None:
    """Run ``explain`` + ``predict`` on ``model`` and write a standalone HTML dashboard to ``output_path``."""
    trace: Dict[str, Any] = model.explain(data)
    result: Dict[str, Any] = model.predict(data)

    src_block = ""
    if source_code is not None:
        src_block = (
            '<section class="source"><h2>Strategy source</h2>'
            f"<pre><code>{escape(source_code)}</code></pre></section>"
        )

    cards = []
    for k, v in result.items():
        cards.append(
            '<div class="card output-card">'
            f'<div class="card-label">{escape(str(k))}</div>'
            f'<div class="card-value">{escape(_format_cell_value(v))}</div>'
            "</div>"
        )
    cards_html = "".join(cards) if cards else '<div class="card muted">No outputs</div>'

    input_rows = []
    for k, v in sorted(data.items(), key=lambda x: str(x[0])):
        input_rows.append(
            "<tr><td class=\"k\">{}</td><td>{}</td></tr>".format(
                escape(str(k)), escape(_format_cell_value(v)))
        )
    inputs_table = "".join(input_rows) if input_rows else "<tr><td colspan=\"2\" class=\"muted\">—</td></tr>"

    adapter_rows = []
    for k, v in sorted(trace.items(), key=lambda x: str(x[0])):
        cls = "adapter-highlight" if _is_adapter_key(k) else ""
        adapter_rows.append(
            "<tr class=\"{}\"><td class=\"k\">{}</td><td>{}</td></tr>".format(
                cls, escape(str(k)), escape(_format_cell_value(v)))
        )
    adapters_table = "".join(adapter_rows) if adapter_rows else "<tr><td colspan=\"2\" class=\"muted\">—</td></tr>"

    trace_rows = []
    for k, v in sorted(trace.items(), key=lambda x: str(x[0])):
        trace_rows.append(
            "<tr><td>{}</td><td><code>{}</code></td></tr>".format(
                escape(str(k)), escape(_format_cell_value(v)))
        )
    trace_table = "".join(trace_rows) if trace_rows else "<tr><td colspan=\"2\" class=\"muted\">—</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Axiom Glass Box Execution Report</title>
  <style>
    :root {{
      --bg: #0b0e11;
      --panel: #12161c;
      --border: #1e2630;
      --text: #e6edf3;
      --muted: #7d8590;
      --accent: #3fb950;
      --glow: #58a6ff;
      --font: "Segoe UI", system-ui, -apple-system, sans-serif;
      --mono: "Cascadia Code", "Fira Code", ui-monospace, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      min-height: 100vh;
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem 3rem; }}
    header {{
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.25rem;
      margin-bottom: 2rem;
    }}
    header h1 {{
      margin: 0;
      font-size: 1.65rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      color: var(--text);
    }}
    header p {{
      margin: 0.5rem 0 0;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .source {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      margin-bottom: 2rem;
    }}
    .source h2 {{ margin: 0 0 0.75rem; font-size: 1rem; color: var(--muted); font-weight: 600; }}
    .source pre {{
      margin: 0;
      overflow-x: auto;
      font-family: var(--mono);
      font-size: 0.78rem;
      color: var(--text);
    }}
    .row-cards {{
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 2rem;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      min-width: 140px;
      flex: 1 1 160px;
    }}
    .output-card .card-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
    .output-card .card-value {{
      font-family: var(--mono);
      font-size: 1.35rem;
      font-weight: 600;
      color: var(--accent);
      margin-top: 0.35rem;
      word-break: break-all;
    }}
    .muted {{ color: var(--muted); }}
    .split {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
      margin-bottom: 2rem;
    }}
    @media (max-width: 800px) {{ .split {{ grid-template-columns: 1fr; }} }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.25rem;
    }}
    .panel h2 {{
      margin: 0 0 1rem;
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    table.data {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    table.data td, table.data th {{ padding: 0.5rem 0.6rem; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }}
    table.data .k {{ font-family: var(--mono); color: var(--glow); width: 38%; }}
    tr.adapter-highlight td {{
      background: rgba(88, 166, 255, 0.08);
      box-shadow: inset 0 0 0 1px rgba(88, 166, 255, 0.25);
    }}
    tr.adapter-highlight .k {{
      color: var(--glow);
      text-shadow: 0 0 12px rgba(88, 166, 255, 0.35);
    }}
    .full-trace table.data code {{ font-family: var(--mono); font-size: 0.82rem; color: var(--text); }}
    .full-trace h2 {{ margin-top: 0; }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Axiom Glass Box Execution Report</h1>
      <p>Interpretable execution trace — mathematically bounded hybrid symbolic–neural run.</p>
    </header>
    {src_block}
    <section class="outputs">
      <h2 class="section-title" style="margin:0 0 1rem;font-size:0.95rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;">Final outputs</h2>
      <div class="row-cards">{cards_html}</div>
    </section>
    <div class="split">
      <section class="panel">
        <h2>Input features</h2>
        <table class="data"><tbody>{inputs_table}</tbody></table>
      </section>
      <section class="panel">
        <h2>Neural adapters</h2>
        <p style="margin:-0.5rem 0 1rem;font-size:0.8rem;color:var(--muted);">Trace keys matching <em>alpha</em>, <em>neural</em>, or <em>prediction</em> are highlighted.</p>
        <table class="data"><tbody>{adapters_table}</tbody></table>
      </section>
    </div>
    <section class="panel full-trace">
      <h2>Full execution trace</h2>
      <table class="data"><thead><tr><th>Variable</th><th>Value</th></tr></thead><tbody>{trace_table}</tbody></table>
    </section>
  </div>
</body>
</html>
"""

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
