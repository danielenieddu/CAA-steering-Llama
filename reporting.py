"""
reporting.py — Salvataggio risultati e grafici (logica condivisa dagli script).

Replica i save_* del notebook ma parametrizzati sul nome del concetto invece che
sul BRAND globale, cosi' funzionano per Napoleone / Colosseo / Apple.
"""

import json
import shutil
from pathlib import Path

import numpy as np

import config as C

PROMPT_TYPES = ["neutral", "stimulus"]

CONDITION_COLORS = {"baseline": "#a8dadc", "prompted": "#f4a261", "steered": "#e63946"}


# ─────────────────────────────────────────────────────────────────────────
def make_output_dir(name, root=None):
    root = Path(root or C.RESULTS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    d = root / name
    if d.exists():
        if name.startswith("_"):
            raise FileExistsError(
                f"🛑 Esiste gia': {d}\n"
                f"   Le run baseline/prompted sono costose da rigenerare.\n"
                f"   Cancellala a mano se vuoi rifarla:  rm -rf {d}"
            )
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def section_header(title, style="major"):
    if style == "major":
        w = len(title) + 4
        return f"\n╔{'═'*w}╗\n║  {title}  ║\n╚{'═'*w}╝"
    w = len(title) + 2
    return f"\n┌{'─'*w}┐\n│ {title} │\n└{'─'*w}┘"


def format_result(r, index, total, concept, max_len=250):
    mtag = "✅" if r["mentions_concept"] else "❌"
    ttag = "⬜️" if r["prompt_type"] == "neutral" else "🟨"
    ctag = {"steered": "▶️", "prompted": "⏩️", "baseline": "⏹️"}.get(r["condition"], "⏹️")
    p = r["prompt"]
    resp = r["response"]
    p = (p[:80] + " [...]") if len(p) > 80 else p
    resp = (resp[:max_len] + " [...]") if len(resp) > max_len else resp
    return "\n".join([
        f"{mtag} [{index}/{total}] | {ctag} {r['condition']} | {ttag} {r['prompt_type']} "
        f"| mentions {concept}: {'YES' if r['mentions_concept'] else 'no'}",
        f"❓ {p}",
        f"💬 {resp}",
        "─" * 60,
    ])


def compute_mention_rate(results, prompt_type, condition):
    sub = [r for r in results if r["prompt_type"] == prompt_type and r["condition"] == condition]
    if not sub:
        return 0.0
    return sum(1 for r in sub if r["mentions_concept"]) / len(sub)


def save_summary(results, conditions, out_dir, concept, subtitle="", filename="summary.txt"):
    data = {}
    lines = ["=" * 60, f"MENTION RATES — {concept}"]
    if subtitle:
        lines.append(subtitle)
    lines += ["=" * 60, f"{'Condition':<15}{'Prompt Type':<14}{'Mention Rate'}", "-" * 45]
    for cond in conditions:
        for pt in PROMPT_TYPES:
            rate = compute_mention_rate(results, pt, cond)
            lines.append(f"{cond:<15}{pt:<14}{rate:>10.0%}")
            data[(cond, pt)] = rate
    text = "\n".join(lines)
    (out_dir / filename).write_text(text)
    print("\n" + text + f"\n💾 {out_dir / filename}")
    return data


def save_outputs(results, conditions, out_dir, concept, filename="outputs.txt"):
    lines = []
    for cond in conditions:
        for pt in PROMPT_TYPES:
            sub = [r for r in results if r["condition"] == cond and r["prompt_type"] == pt]
            if not sub:
                continue
            lines.append(section_header(f"{cond} · {pt}", "major"))
            for i, r in enumerate(sub, 1):
                lines.append(format_result(r, i, len(sub), concept))
    (out_dir / filename).write_text("\n".join(lines))
    print(f"💾 {out_dir / filename}")


def save_json(results, out_dir, concept, filename="results.json", extra_meta=None):
    payload = {"concept": concept, "results": results}
    if extra_meta:
        payload.update(extra_meta)
    (out_dir / filename).write_text(json.dumps(payload, indent=2, default=str))
    print(f"💾 {out_dir / filename}")


def save_bar_chart(summary_data, conditions, out_dir, concept,
                   n_neutral, n_stimulus, filename="bar_chart.png", title_extra=""):
    import matplotlib.pyplot as plt

    labels = [f"Neutral\n({n_neutral})", f"Stimulus\n({n_stimulus})"]
    x = np.arange(len(labels))
    n = len(conditions)
    width = 0.7 / max(n, 1)

    fig, ax = plt.subplots(figsize=(max(6, 2 + 3 * n), 5))
    bars_all = []
    cond_labels = {"baseline": "Baseline", "prompted": f"Prompted ({concept})",
                   "steered": f"Steered ({concept})"}
    for k, cond in enumerate(conditions):
        rates = [summary_data.get((cond, pt), 0) for pt in PROMPT_TYPES]
        offset = (k - (n - 1) / 2) * width
        bars = ax.bar(x + offset, rates, width,
                      label=cond_labels.get(cond, cond),
                      color=CONDITION_COLORS.get(cond, "#999999"))
        bars_all.extend(bars)

    ax.set_ylabel("Mention rate")
    title = f"Concept Mention Rate — {concept}"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    for b in bars_all:
        h = b.get_height()
        ax.annotate(f"{h:.0%}", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=11)
    plt.tight_layout()
    path = out_dir / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    try:
        plt.show()          # inline su Colab/Jupyter; no-op senza display
    except Exception:
        pass
    plt.close(fig)
    print(f"📊 {path}")
