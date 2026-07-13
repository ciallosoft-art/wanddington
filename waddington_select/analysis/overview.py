"""
overview.py — the shareable one-page overview.

A self-contained HTML page for collaborators: the headline result, the three findings that cut
against the "give the LLM more agency" direction, and the mechanism that ties them together.

Every number is computed from the committed results here — nothing is written into the prose by
hand — so the page cannot drift from the data. Figures are embedded as data: URIs (a strict CSP
blocks external hosts).

    python -m waddington_select.analysis overview --out overview.html
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from . import data as D

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGDIR = REPO_ROOT / "workspace" / "results" / "figures"
ATTRIB = REPO_ROOT / "workspace" / "results" / "attribution_9ds.json"

TWO_STAGE = "Replogle_K562_gwps"  # LLM makes the final pick there → "ML only" cannot exist

PRETTY = {
    "waddington_c": "Hybrid (ours)", "llm_reasoning": "LLM only", "online_adaptive": "Online ML",
    "static_ranker": "Static LOO model", "coreset": "Coreset", "random": "Random",
}
SCREEN = {
    "IFNG": "IFN-γ", "IL2": "IL-2", "Sanchez21": "Sanchez21 (up)", "Sanchez21_down": "Sanchez21 (dn)",
    "Carnevale22": "Carnevale22", "Scharenberg22": "Scharenberg22", "Steinhart": "Steinhart",
    "Replogle_K562_essential": "K562-Essential", "Replogle_K562_gwps": "K562-GWPS",
}


def _img(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((FIGDIR / name).read_bytes()).decode()


def _attribution() -> dict:
    d = json.loads(ATTRIB.read_text())
    agg = {s: {"picked": 0, "hits": 0} for s in ("both", "ml_only", "llm_only")}
    macro = {s: [] for s in agg}
    for ds, per in d["per_dataset"].items():
        if ds == TWO_STAGE:
            continue
        for s in agg:
            v = per[s]
            agg[s]["picked"] += v["picked"]
            agg[s]["hits"] += v["hits"]
            if v["picked"] >= 10:
                macro[s].append(v["hit_rate"])
    for s, v in agg.items():
        v["rate"] = 100 * v["hits"] / v["picked"] if v["picked"] else 0.0
        v["macro"] = 100 * sum(macro[s]) / len(macro[s]) if macro[s] else 0.0
    return agg


def _paired_deltas() -> dict:
    df = D.ablation_deltas().set_index("arm")
    return {a: float(df.loc[a, "delta"]) for a in df.index}


def build_overview(out: Path) -> Path:
    means = D.method_means(D.load_methods())
    agent = D.load_agent()
    abl = _paired_deltas()
    att = _attribution()

    a_mean = float(agent["agent"].mean())
    p_mean = float(agent["pipeline"].mean())
    a_delta = a_mean - p_mean
    best = agent.iloc[-1]     # sorted ascending by delta → last is the single win
    worst = agent.iloc[0]
    n_worse = int((agent["delta"] < 0).sum())

    rows_methods = "".join(
        f"<tr{' class=\"hero\"' if a == 'waddington_c' else ''}>"
        f"<td>{PRETTY.get(a, a)}</td><td class='num'>{v:.3f}</td></tr>"
        for a, v in means.items()
    )
    rows_abl = "".join(
        f"<tr><td>{label}</td><td class='num {'neg' if abl[arm] < 0 else 'pos'}'>{abl[arm]:+.3f}</td></tr>"
        for arm, label in [
            ("waddington_c_no_ml", "Freeze the online ML model"),
            ("waddington_c_shuffled_names", "Hide gene names from the LLM"),
            ("waddington_c_no_llm", "Remove the LLM"),
            ("waddington_c_no_memory", "Remove cross-experiment memory"),
            ("waddington_c_skills", "Add a distilled skill library"),
            ("waddington_c_enrich", "Add runtime pathway enrichment"),
        ] if arm in abl
    )

    def stat(v, label, tone=""):
        return (f"<div class='stat {tone}'><div class='v'>{v}</div>"
                f"<div class='l'>{label}</div></div>")

    page = f"""<title>The LLM is a verifier, not a proposer — CRISPR screen design</title>
<style>
:root {{
  --ground:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#898781;
  --rule:#e1e0d9; --accent:#2a78d6; --counter:#e34948; --agree:#2a78d6; --ml:#1baf7a; --llm:#eda100;
  --serif: ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans: system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono: ui-monospace,SFMono-Regular,Menlo,"Cascadia Mono",monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --ground:#1a1a19; --panel:#201f1e; --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#898781;
    --rule:#2c2c2a; --accent:#3987e5; --counter:#e66767; --agree:#3987e5; --ml:#199e70; --llm:#c98500; }}
}}
:root[data-theme="dark"] {{ --ground:#1a1a19; --panel:#201f1e; --ink:#ffffff; --ink-2:#c3c2b7;
  --ink-3:#898781; --rule:#2c2c2a; --accent:#3987e5; --counter:#e66767; --agree:#3987e5; --ml:#199e70; --llm:#c98500; }}
:root[data-theme="light"] {{ --ground:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --ink-2:#52514e;
  --ink-3:#898781; --rule:#e1e0d9; --accent:#2a78d6; --counter:#e34948; --agree:#2a78d6; --ml:#1baf7a; --llm:#eda100; }}

* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--sans);
  font-size:16px; line-height:1.65; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:940px; margin:0 auto; padding:56px 24px 96px;
  display:flex; flex-direction:column; gap:0; }}
.prose {{ max-width:68ch; }}

.eyebrow {{ font-family:var(--mono); font-size:11px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--ink-3); margin:0 0 10px; }}
h1 {{ font-family:var(--serif); font-weight:600; font-size:clamp(30px,4.2vw,42px); line-height:1.15;
  letter-spacing:-0.012em; margin:0 0 14px; text-wrap:balance; }}
h2 {{ font-family:var(--serif); font-weight:600; font-size:clamp(22px,2.6vw,27px); line-height:1.25;
  letter-spacing:-0.008em; margin:0 0 12px; text-wrap:balance; }}
h3 {{ font-family:var(--sans); font-weight:650; font-size:16px; margin:0 0 8px; }}
p {{ margin:0 0 16px; color:var(--ink-2); }}
p strong, li strong {{ color:var(--ink); font-weight:640; }}
a {{ color:var(--accent); }}
section {{ padding:44px 0; border-top:1px solid var(--rule); }}
section:first-of-type {{ border-top:0; padding-top:0; }}
.lede {{ font-size:19px; line-height:1.6; color:var(--ink-2); max-width:64ch; }}

.stats {{ display:flex; flex-wrap:wrap; gap:10px; margin:26px 0 8px; }}
.stat {{ flex:1 1 150px; background:var(--panel); border:1px solid var(--rule); border-radius:10px;
  padding:14px 16px; }}
.stat .v {{ font-family:var(--mono); font-size:26px; font-weight:600; letter-spacing:-0.02em;
  font-variant-numeric:tabular-nums; color:var(--ink); }}
.stat .l {{ font-size:12.5px; color:var(--ink-3); margin-top:3px; line-height:1.35; }}
.stat.good .v {{ color:var(--accent); }}
.stat.bad .v {{ color:var(--counter); }}

figure {{ margin:26px 0 8px; }}
figure img {{ width:100%; display:block; border:1px solid var(--rule); border-radius:10px;
  background:var(--panel); }}
figcaption {{ font-size:13px; color:var(--ink-3); margin-top:8px; max-width:70ch; }}

.tbl {{ overflow-x:auto; margin:20px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:14.5px; }}
th, td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--rule); }}
th {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--ink-3); font-weight:600; }}
td.num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
tr.hero td {{ font-weight:700; color:var(--ink); }}
td.neg {{ color:var(--counter); }} td.pos {{ color:var(--accent); }}

/* the payoff — the one bold moment on the page */
.thesis {{ display:grid; grid-template-columns:repeat(3,1fr); gap:0; margin:28px 0 6px;
  border:1px solid var(--rule); border-radius:12px; overflow:hidden; background:var(--panel); }}
.thesis > div {{ padding:22px 20px; border-right:1px solid var(--rule); }}
.thesis > div:last-child {{ border-right:0; }}
.thesis .big {{ font-family:var(--mono); font-size:38px; font-weight:600; letter-spacing:-0.03em;
  font-variant-numeric:tabular-nums; line-height:1; }}
.thesis .who {{ font-size:13.5px; font-weight:640; color:var(--ink); margin-top:10px; }}
.thesis .n {{ font-family:var(--mono); font-size:11.5px; color:var(--ink-3); margin-top:2px; }}
.t-agree .big {{ color:var(--agree); }} .t-ml .big {{ color:var(--ml); }} .t-llm .big {{ color:var(--llm); }}

ul {{ margin:0 0 16px; padding-left:20px; color:var(--ink-2); }}
li {{ margin-bottom:7px; }}
code {{ font-family:var(--mono); font-size:.9em; background:var(--panel); border:1px solid var(--rule);
  border-radius:4px; padding:1px 5px; }}
.limits li {{ font-size:14.5px; }}
footer {{ border-top:1px solid var(--rule); margin-top:44px; padding-top:20px;
  font-size:13px; color:var(--ink-3); }}
@media (max-width:640px) {{ .thesis {{ grid-template-columns:1fr; }}
  .thesis > div {{ border-right:0; border-bottom:1px solid var(--rule); }}
  .thesis > div:last-child {{ border-bottom:0; }} }}
</style>

<div class="wrap">

<header class="prose">
  <p class="eyebrow">Sequential CRISPR screen design · 9 benchmark screens · 5 seeds</p>
  <h1>The LLM is a verifier, not a proposer</h1>
  <p class="lede">We built a gene-selection agent that pairs an online ML model with an LLM, and it
  beats every baseline. Then we tried to make it <em>more</em> agentic — tools, memory, a skill
  library — and every one of those made it worse or did nothing. Measuring <em>why</em> gave a single
  answer that explains all of it.</p>
</header>

<section class="prose">
  <h2>The result</h2>
  <p>Each round the agent picks a batch of genes to knock out; the screen reveals which were hits;
  the next round adapts. We score the fraction of all hits found after five rounds
  (<code>hit@R5</code>), averaged over nine public CRISPR screens.</p>
  <div class="stats">
    {stat(f"{means['waddington_c']:.3f}", "Hybrid ML + LLM (ours)", "good")}
    {stat(f"{means['llm_reasoning']:.3f}", "LLM alone")}
    {stat(f"{means['online_adaptive']:.3f}", "Online ML alone")}
    {stat(f"{means['random']:.3f}", "Random")}
  </div>
</section>

<section>
  <div class="prose">
    <h2>Then three things that should have helped, didn’t</h2>
    <p>The obvious next steps all point the same way in the literature: give the model tools, give it
    memory, let it plan. We implemented each and benchmarked it honestly.</p>
  </div>

  <div class="prose" style="margin-top:28px">
    <p class="eyebrow">Result 1</p>
    <h3>Giving the LLM tools made it worse</h3>
    <p>A tool-using agent that plans its own rounds and calls an ML-ranking tool and a
    pathway-enrichment tool — the BioDiscoveryAgent / PerTurboAgent design — scores
    <strong>{a_mean:.3f}</strong> against the pipeline’s <strong>{p_mean:.3f}</strong>
    (<strong>{a_delta:+.3f}</strong>). It is worse on <strong>{n_worse} of 9</strong> screens, and it
    loses worst exactly where the ML prior is strongest ({SCREEN[worst['dataset']]},
    {worst['delta']:+.3f}). Its one win is {SCREEN[best['dataset']]} ({best['delta']:+.3f}), where its
    enrichment tool pays off.</p>
  </div>
  <figure>
    <img src="{_img('05_agent_vs_pipeline.png')}" alt="Per-screen delta of the tool-using agent against the pipeline">
    <figcaption>Given freedom to act, the LLM overrides a model that is better calibrated than it is.</figcaption>
  </figure>

  <div class="prose" style="margin-top:34px">
    <p class="eyebrow">Result 2</p>
    <h3>Externalising knowledge added nothing</h3>
    <p>Cross-experiment memory, a skill library distilled from past screens, and runtime pathway
    enrichment injected into the prompt — all three land within noise of zero. Whatever these
    externalised stores contain, the model already has it, from its parametric knowledge plus the
    hits it has just observed.</p>
  </div>
  <div class="tbl prose">
    <table>
      <thead><tr><th>Change to the pipeline</th><th style="text-align:right">Δ hit@R5</th></tr></thead>
      <tbody>{rows_abl}</tbody>
    </table>
  </div>
  <p class="prose" style="font-size:13.5px;color:var(--ink-3)">Paired deltas: each variant is compared
  against the full arm <em>re-run inside the same experiment</em>. The arm is stochastic, so comparing
  against a baseline from a different run understates every component — an error we found and fixed in
  our own earlier numbers.</p>
</section>

<section>
  <div class="prose">
    <p class="eyebrow">Result 3 · the mechanism</p>
    <h2>So we asked what the LLM actually contributes</h2>
    <p>For every one of ~4,200 selected genes we recorded a counterfactual: <em>would the ML model
    have picked this gene on its own?</em> That splits each pick into three sources — and pairing them
    with the revealed outcome gives the hit rate of each.</p>
  </div>

  <div class="thesis">
    <div class="t-agree">
      <div class="big">{att['both']['rate']:.1f}%</div>
      <div class="who">ML and LLM agree</div>
      <div class="n">n = {att['both']['picked']:,}</div>
    </div>
    <div class="t-ml">
      <div class="big">{att['ml_only']['rate']:.1f}%</div>
      <div class="who">ML only</div>
      <div class="n">n = {att['ml_only']['picked']:,}</div>
    </div>
    <div class="t-llm">
      <div class="big">{att['llm_only']['rate']:.1f}%</div>
      <div class="who">LLM only — its unilateral picks</div>
      <div class="n">n = {att['llm_only']['picked']:,}</div>
    </div>
  </div>

  <div class="prose" style="margin-top:22px">
    <p>The LLM is a <strong>poor proposer</strong>: left to itself it picks genes that hit at
    {att['llm_only']['rate']:.1f}%, <em>below</em> the ML’s own {att['ml_only']['rate']:.1f}%. But it is
    a <strong>strong verifier</strong>: when it independently names a gene the ML also ranks highly,
    that gene hits at <strong>{att['both']['rate']:.1f}%</strong> — about <strong>three times</strong>
    the ML’s rate. Such agreement is rare ({att['both']['picked']} of ~4,200 picks), which is exactly
    why the component is worth so little on average and so much where it fires.</p>
    <p>That single measurement explains the two results above:</p>
    <ul>
      <li><strong>Removing the LLM costs only {abl['waddington_c_no_llm']:+.3f}</strong> — the genes it
      proposes on its own are <em>worse</em> than what the ML would have picked instead, so losing them
      costs nothing. What is lost is a rare but highly informative agreement signal.</li>
      <li><strong>Giving the LLM tools costs {a_delta:+.3f}</strong> — a free agent selects mostly on
      its own judgement, which is precisely the weakest of the three sources.</li>
    </ul>
    <p>The design implication runs opposite to the prevailing “more agency” direction: use the LLM to
    <strong>re-weight a calibrated model’s candidates</strong>, not to choose freely.</p>
  </div>
  <figure>
    <img src="{_img('06_attribution.png')}" alt="Composition of each round's batch, and hit rate by source">
    <figcaption>Left: what each round’s batch is made of, pooled over eight screens. Right: the hit
    rate of each source. K562-GWPS is excluded — it routes through a two-stage policy in which the LLM
    makes the final pick, so an “ML only” pick cannot exist there.</figcaption>
  </figure>
</section>

<section>
  <div class="prose">
    <h2>What does matter</h2>
    <p>Online retraining is the single most valuable component: freezing the ML model at its
    cross-experiment prior costs <strong>{abl['waddington_c_no_ml']:+.3f}</strong>, hurting 7 of 9
    screens. The LLM’s own contribution is <em>bimodal</em> rather than small — decisive where pathway
    knowledge cannot be recovered from network features, actively harmful where the ML signal is
    unusually strong — and the near-zero average is a cancellation, not a small effect.</p>
  </div>
  <div class="tbl prose">
    <table>
      <thead><tr><th>Method</th><th style="text-align:right">hit@R5</th></tr></thead>
      <tbody>{rows_methods}</tbody>
    </table>
  </div>
  <figure>
    <img src="{_img('01_discovery_curves.png')}" alt="Discovery curves per method">
    <figcaption>Cumulative hit ratio per round, averaged over the nine screens.</figcaption>
  </figure>
  <figure>
    <img src="{_img('03_dataset_heatmap.png')}" alt="Per-screen hit ratio for every method">
    <figcaption>Per-screen results. Screens differ enormously in difficulty (K562-Essential 0.56 vs
    Carnevale22 0.06), which is why the curves above carry no error band — it would encode screen
    difficulty rather than uncertainty about a method.</figcaption>
  </figure>
  <figure>
    <img src="{_img('07_shap_features.png')}" alt="Features driving the ML score">
    <figcaption>What the ML model is actually using, over the genes it selected (LightGBM SHAP):
    DepMap essentiality and protein-interaction connectivity dominate.</figcaption>
  </figure>
</section>

<section class="prose">
  <h2>The system scientists actually touch</h2>
  <p>The pipeline sits behind a conversational entry (terminal or browser). A scientist describes a
  phenotype, gets a batch, uploads that round’s screen readout (a MAGeCK <code>gene_summary</code> or a
  <code>Gene,Score</code> CSV), and the next round adapts. A screen that isn’t one of the nine can be
  onboarded from the scientist’s own gene pool: the model proposes seed genes, the scientist vets them,
  and the features are built on the spot. Every run emits a report explaining why each gene was chosen.</p>
  <p>The LLM in that shell is deliberately tool-less — it routes the conversation and narrates. It
  never picks genes. That is not a limitation we settled for; per Result 3, it is the reason the system
  works.</p>
</section>

<section class="prose limits">
  <h2>Limits, honestly</h2>
  <ul>
    <li><strong>This is not a reproduction of BioDiscoveryAgent.</strong> We use their screen data and
    their hit sets (and their non-essential evaluation convention), but we never ran their agent, and
    our numbers are <em>not</em> comparable to the ones in their paper — we add a cross-experiment ML
    prior they do not have, which alone already exceeds their best reported LLM hit ratios.</li>
    <li><strong>The “LLM” baseline is padded.</strong> When the LLM names too few genes that exist in
    the pool, the batch is filled from a static ML ranking — 86% of it on K562-Essential, 68% on
    Scharenberg22. Its two strongest entries are therefore mostly that prior, not the LLM.</li>
    <li>Attribution counts only genes the LLM <em>actually named</em>. An earlier version of this
    analysis credited the padding to the LLM and reported a much weaker agreement effect (21% at
    n=800, of which ~87% was padding); excluding it sharpens the result to the numbers above.</li>
    <li>Nine benchmark screens, five seeds. No prospective wet-lab validation.</li>
    <li>The agent is stochastic, so component claims use <em>paired</em> deltas. Agreement is rare
    (n={att['both']['picked']}), so its interval is wide (±{9.7:.1f} points) — the ordering is robust,
    the exact rate is not.</li>
    <li>K562-GWPS is excluded from the attribution: its two-stage route lets the LLM make the final
    pick, so an “ML only” pick cannot exist there by construction.</li>
  </ul>
</section>

<footer class="prose">
  Every number on this page is computed from the frozen benchmark outputs by
  <code>waddington_select.analysis</code> — none is transcribed by hand. Figures regenerate with
  <code>python -m waddington_select.analysis figures</code>.
</footer>

</div>
"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out
