"""
report.py — a self-contained HTML report for ONE experiment run.

This is the scientist's artifact: what was tested each round, what hit, and **why those genes were
chosen** (attribution split, the features driving the ML score, the ML/LLM agreement signal). Figures
are embedded as base64 PNGs so the file opens anywhere with no assets.

    python -m waddington_select.analysis report --campaign <trace.json> [--out report.html]
"""

from __future__ import annotations

import base64
import html
import json
import tempfile
from pathlib import Path

from .explain import build_all
from .trace import hit_rate_by_source

CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin:0; padding:32px 20px 64px; background:#f9f9f7; color:#0b0b0b;
  font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width: 900px; margin: 0 auto; }
h1 { font-size:22px; margin:0 0 4px; }
h2 { font-size:15px; margin:32px 0 10px; padding-bottom:6px; border-bottom:1px solid #e1e0d9; }
.sub { color:#52514e; margin:0 0 24px; font-size:13.5px; }
.tiles { display:flex; flex-wrap:wrap; gap:12px; margin:20px 0 8px; }
.tile { flex:1; min-width:150px; background:#fcfcfb; border:1px solid #e1e0d9; border-radius:10px; padding:12px 14px; }
.tile .k { color:#52514e; font-size:12px; }
.tile .v { font-size:24px; font-weight:600; margin-top:2px; }
figure { margin:16px 0; }
figure img { width:100%; border:1px solid #e1e0d9; border-radius:10px; background:#fcfcfb; }
figcaption { color:#52514e; font-size:12.5px; margin-top:6px; }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th, td { text-align:left; padding:7px 8px; border-bottom:1px solid #e1e0d9; }
th { color:#52514e; font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
td.num { text-align:right; font-variant-numeric: tabular-nums; }
.round { background:#fcfcfb; border:1px solid #e1e0d9; border-radius:10px; padding:14px 16px; margin:12px 0; }
.round h3 { margin:0 0 8px; font-size:14px; }
.genes { display:flex; flex-wrap:wrap; gap:5px; margin-top:6px; }
.g { font:12px ui-monospace,monospace; padding:2px 7px; border-radius:5px;
     background:#eef2f6; color:#52514e; border:1px solid #e1e0d9; }
.g.hit { background:#e7f0fb; color:#104281; border-color:#9ec5f4; font-weight:600; }
.note { color:#52514e; font-size:12.5px; }
@media (prefers-color-scheme: dark) {
  body { background:#0d0d0d; color:#fff; }
  h2 { border-color:#2c2c2a; }
  .sub,.tile .k,figcaption,th,.note,.g { color:#c3c2b7; }
  .tile,.round,figure img { background:#1a1a19; border-color:#2c2c2a; }
  th,td { border-color:#2c2c2a; }
  .g { background:#232323; border-color:#2c2c2a; }
  .g.hit { background:#12325c; color:#cde2fb; border-color:#1c5cab; }
}
"""


def _img(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def _esc(s) -> str:
    return html.escape(str(s))


def build_report(campaign_path: Path, out: Path) -> Path:
    run = json.loads(Path(campaign_path).read_text())
    trace = run.get("trace") or {"rounds": []}
    history = run.get("history", [])
    dataset = run.get("dataset", "?")

    # figures rendered into a temp dir, then inlined
    figs: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory() as td:
        for p in build_all(trace, Path(td)):
            figs.append((p.stem, _img(p)))

    agg = hit_rate_by_source(trace) if trace["rounds"] else {}
    total_hits = run.get("total_hits")
    cumulative = run.get("cumulative_hits", 0)
    ratio = run.get("hit_ratio", 0.0)
    tested = sum(h.get("tested", 0) for h in history)

    tiles = f"""
    <div class="tiles">
      <div class="tile"><div class="k">Phenotype</div><div class="v" style="font-size:17px">{_esc(dataset)}</div></div>
      <div class="tile"><div class="k">Genes tested</div><div class="v">{tested}</div></div>
      <div class="tile"><div class="k">Hits found</div><div class="v">{cumulative}{f' / {total_hits}' if total_hits else ''}</div></div>
      <div class="tile"><div class="k">Hit ratio</div><div class="v">{ratio*100:.1f}%</div></div>
    </div>"""

    # why-it-picked table
    why_rows = ""
    if agg:
        for s, label in (("both", "ML + LLM agree"), ("ml_only", "ML only"), ("llm_only", "LLM only")):
            v = agg.get(s, {})
            why_rows += (f"<tr><td>{label}</td><td class='num'>{v.get('picked',0)}</td>"
                         f"<td class='num'>{v.get('hits',0)}</td>"
                         f"<td class='num'>{v.get('hit_rate',0)*100:.0f}%</td></tr>")

    rounds_html = ""
    hit_lookup = {r["round"]: {g["gene"] for g in r["genes"] if g.get("hit")} for r in trace["rounds"]}
    src_lookup = {r["round"]: {g["gene"]: g["source"] for g in r["genes"]} for r in trace["rounds"]}
    for h in history:
        rn = h["round"]
        hits = set(h.get("hits", [])) or hit_lookup.get(rn, set())
        batch = list(src_lookup.get(rn, {}).keys())
        chips = "".join(
            f"<span class='g {'hit' if g in hits else ''}' title='{_esc(src_lookup.get(rn,{}).get(g,''))}'>{_esc(g)}</span>"
            for g in batch
        )
        tr = next((t for t in trace["rounds"] if t["round"] == rn), {})
        counts = tr.get("counts", {})
        shap_top = ", ".join(list((tr.get("shap") or {}).keys())[:4])
        rounds_html += f"""
        <div class="round">
          <h3>Round {rn} — tested {h.get('tested', len(batch))}, found {len(hits)} hits
              (cumulative {h.get('cumulative','?')}{f"/{total_hits}" if total_hits else ''})</h3>
          <div class="note">Chosen by: {counts.get('both',0)} ML+LLM agreement · {counts.get('ml_only',0)} ML only ·
             {counts.get('llm_only',0)} LLM only &nbsp;|&nbsp; route <code>{_esc(tr.get('route','?'))}</code>
             &nbsp;|&nbsp; top ML features: {_esc(shap_top)}</div>
          <div class="genes">{chips}</div>
        </div>"""

    figs_html = ""
    caps = {
        "06_attribution": "Where each pick came from, and which source actually finds hits. "
                          "Agreement between the ML model and the LLM is the strongest signal.",
        "07_shap_features": "Features driving the ML score over the genes it selected (LightGBM SHAP).",
        "08_calibration": "Does a higher ML score mean a likelier hit? Quintiles of the tested genes.",
    }
    for stem, src in figs:
        figs_html += (f"<figure><img src='{src}' alt='{_esc(stem)}'>"
                      f"<figcaption>{_esc(caps.get(stem, stem))}</figcaption></figure>")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment report — {_esc(dataset)}</title><style>{CSS}</style></head>
<body><div class="wrap">
  <h1>Experiment report — {_esc(dataset)}</h1>
  <p class="sub">Sequential CRISPR gene selection with the Waddington C-arm (online ML + LLM reasoning
     + cross-experiment memory). This report shows what was tested, what hit, and why those genes were chosen.</p>
  {tiles}

  <h2>Why these genes?</h2>
  <p class="note">Every pick is attributed against the counterfactual "would the ML have chosen it anyway?".
     <b>LLM only</b> means the LLM pulled a gene in that the ML would have skipped — that is the LLM's
     marginal contribution.</p>
  <table><thead><tr><th>Source</th><th class="num">Picked</th><th class="num">Hits</th><th class="num">Hit rate</th></tr></thead>
  <tbody>{why_rows}</tbody></table>

  {figs_html}

  <h2>Round by round</h2>
  {rounds_html}

  <p class="note" style="margin-top:28px">Highlighted chips are confirmed hits. Hover a gene to see which
     source selected it.</p>
</div></body></html>"""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out
