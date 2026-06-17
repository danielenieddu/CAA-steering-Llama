"""
run.py — Orchestratore (carica il modello UNA volta).

Il concetto si sceglie con l'iperparametro config.PROMPT_MODULE (o da CLI).
Ogni esperimento riceve il modulo di prompt caricato `P` e ne usa
P.CONCEPT / P.BRAND / P.ALIASES / P.NEUTRAL_PROMPTS / P.STIMULUS_PROMPTS / ecc.

Sottocomandi (= sezioni del notebook):
  baseline   -> 5.2 + 5.3
  sweep      -> 4
  steered    -> 5.4
  identities -> nuovo
  all        -> tutto, per ogni modulo in config.PROMPT_MODULES

Esempi:
  python run.py all
  python run.py sweep --concept Colosseo
  python run.py steered --prompts prompts_apple --layer 16 --frac 0.30
  python run.py baseline                      # usa config.PROMPT_MODULE
"""

import argparse
import json
from pathlib import Path

import numpy as np

import config as C
import reporting as R
from steering import ConceptSteerer, detect_mention, load_prompts


# ─────────────────────────────────────────────────────────────────────────
def concept_root(concept):
    return Path(C.RESULTS_DIR) / concept


def system_instruction(brand):
    return (f"Mention {brand} in your responses when relevant; "
            f"the mention should feel natural and not forced.")


def run_prompts(steerer, P, prompt_list, prompt_type, condition,
                sv=None, scale=0.0, layer=C.STEERING_LAYER, system=None):
    out = []
    for i, prompt in enumerate(prompt_list, 1):
        resp = steerer.generate(prompt, steering_vector=sv, scale=scale,
                                 layer_idx=layer, system=system)
        r = {
            "prompt_type": prompt_type, "condition": condition, "concept": P.CONCEPT,
            "brand": P.BRAND, "layer": layer if sv is not None else None,
            "scale": round(scale, 3) if sv is not None else 0,
            "prompt": prompt, "response": resp,
            "mentions_concept": detect_mention(resp, P.ALIASES),
        }
        out.append(r)
        print(R.format_result(r, i, len(prompt_list), P.CONCEPT))
    return out


# ── 5.2 + 5.3  BASELINE + PROMPTED ────────────────────────────────────────
def do_baseline(steerer, P):
    root = concept_root(P.CONCEPT)
    n_neu, n_sti = len(P.NEUTRAL_PROMPTS), len(P.STIMULUS_PROMPTS)

    bdir = R.make_output_dir("_baseline", root)
    print(R.section_header(f"BASELINE — {P.CONCEPT}", "major"))
    base = run_prompts(steerer, P, P.NEUTRAL_PROMPTS, "neutral", "baseline")
    base += run_prompts(steerer, P, P.STIMULUS_PROMPTS, "stimulus", "baseline")
    sd = R.save_summary(base, ["baseline"], bdir, P.CONCEPT)
    R.save_outputs(base, ["baseline"], bdir, P.CONCEPT)
    R.save_bar_chart(sd, ["baseline"], bdir, P.CONCEPT, n_neu, n_sti)
    R.save_json(base, bdir, P.CONCEPT)

    pdir = R.make_output_dir("_prompted", root)
    sysmsg = system_instruction(P.BRAND)
    print(R.section_header(f"PROMPTED — {P.CONCEPT}", "major"))
    print(f'system: "{sysmsg}"')
    prom = run_prompts(steerer, P, P.NEUTRAL_PROMPTS, "neutral", "prompted", system=sysmsg)
    prom += run_prompts(steerer, P, P.STIMULUS_PROMPTS, "stimulus", "prompted", system=sysmsg)
    sd = R.save_summary(prom, ["prompted"], pdir, P.CONCEPT, subtitle=f'system: "{sysmsg}"')
    R.save_outputs(prom, ["prompted"], pdir, P.CONCEPT)
    R.save_bar_chart(sd, ["prompted"], pdir, P.CONCEPT, n_neu, n_sti)
    R.save_json(prom, pdir, P.CONCEPT, extra_meta={"system_instruction": sysmsg})


# ── 4  SWEEP LAYER x SCALA ────────────────────────────────────────────────
def do_sweep(steerer, P, layers=None, fracs=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    layers = layers or C.SWEEP_LAYERS
    fracs = fracs or C.SWEEP_FRACS
    sweep_prompts = [P.NEUTRAL_PROMPTS[0], P.STIMULUS_PROMPTS[0]]

    sdir = concept_root(P.CONCEPT) / "_layer_scale"
    sdir.mkdir(parents=True, exist_ok=True)
    log = []

    def emit(s=""):
        print(s); log.append(s)

    emit(R.section_header(f"SWEEP — {P.CONCEPT}", "major"))
    mention = np.zeros((len(layers), len(fracs)))
    coherent = np.ones((len(layers), len(fracs)), dtype=bool)

    for i, layer in enumerate(layers):
        norm = steerer.layer_norms[layer]
        sv = steerer.extract_steering_vector(P, P.BRAND, layer, method=C.EXTRACTION_METHOD)
        emit(f"\n{'─'*60}\nLayer {layer:>2} (‖h‖={norm:.1f})\n{'─'*60}")
        for j, frac in enumerate(fracs):
            scale = frac * norm
            hits, coh = 0, True
            for prompt in sweep_prompts:
                resp = steerer.generate(prompt, steering_vector=sv, scale=scale, layer_idx=layer)
                hit = detect_mention(resp, P.ALIASES)
                hits += int(hit)
                words = resp.split()
                if len(words) > 5 and max(words.count(w) for w in set(words)) / len(words) > 0.4:
                    coh = False
                emit(f"{'✅' if hit else '❌'} L{layer} f{frac:.0%} (α={scale:.1f}) | {prompt[:45]}")
                emit(f"   💬 {resp[:200]}")
            mention[i, j] = hits / len(sweep_prompts)
            coherent[i, j] = coh
            emit(f"   → {hits}/{len(sweep_prompts)} mentions" + ("  ⚠️ incoherent" if not coh else ""))

    best, bi, bj = -1, 0, 0
    for i in range(len(layers)):
        for j in range(len(fracs)):
            score = mention[i, j] * (1.0 if coherent[i, j] else 0.5)
            if score > best or (score == best and fracs[j] < fracs[bj]):
                best, bi, bj = score, i, j
    best_layer, best_frac = layers[bi], fracs[bj]
    emit(f"\n🏆 BEST: layer {best_layer}, {best_frac:.0%} of norm "
         f"(α={best_frac * steerer.layer_norms[best_layer]:.1f})")
    (sdir / "sweep_log.txt").write_text("\n".join(log))

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mention, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    for i in range(len(layers)):
        for j in range(len(fracs)):
            lbl = f"{mention[i, j]:.0%}" + ("" if coherent[i, j] else "\n⚠️")
            ax.text(j, i, lbl, ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if mention[i, j] > 0.6 else "black")
    ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, lw=3, edgecolor="#2196F3", facecolor="none"))
    ax.set_xticks(range(len(fracs))); ax.set_xticklabels([f"{f:.0%}" for f in fracs])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}\n(‖h‖={steerer.layer_norms[l]:.0f})" for l in layers])
    ax.set_xlabel("Steering scale (% of activation norm)")
    ax.set_ylabel("Injection layer")
    ax.set_title(f"Mention Rate — {P.CONCEPT}\n{C.EXTRACTION_METHOD} · ActAdd")
    plt.colorbar(im, ax=ax, label="Mention rate", shrink=0.8)
    plt.tight_layout()
    plt.savefig(sdir / "sweep_heatmap.png", dpi=150, bbox_inches="tight")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)
    print(f"📊 {sdir / 'sweep_heatmap.png'}\n💾 {sdir / 'sweep_log.txt'}")
    return best_layer, best_frac


# ── 5.4  STEERED (ActAdd) + CONFRONTO ─────────────────────────────────────
def do_steered(steerer, P, layer, frac):
    n_neu, n_sti = len(P.NEUTRAL_PROMPTS), len(P.STIMULUS_PROMPTS)
    root = concept_root(P.CONCEPT)

    scale = frac * steerer.layer_norms[layer]
    sv = steerer.extract_steering_vector(P, P.BRAND, layer, method=C.EXTRACTION_METHOD)
    print(f"[steered] {P.CONCEPT} | layer={layer} frac={frac:.0%} (α={scale:.1f}) "
          f"| vettore norm={sv.norm():.3f}")

    frac_str = f"{frac:.2f}".replace(".", "")
    sdir = R.make_output_dir(f"L{layer}_F{frac_str}", root)
    print(R.section_header(f"STEERED — {P.CONCEPT} (α={scale:.1f})", "major"))
    steered = run_prompts(steerer, P, P.NEUTRAL_PROMPTS, "neutral", "steered",
                          sv=sv, scale=scale, layer=layer)
    steered += run_prompts(steerer, P, P.STIMULUS_PROMPTS, "stimulus", "steered",
                           sv=sv, scale=scale, layer=layer)
    sd = R.save_summary(steered, ["steered"], sdir, P.CONCEPT,
                        subtitle=f"layer {layer} | α={scale:.1f} ({frac:.0%} of norm)")
    R.save_outputs(steered, ["steered"], sdir, P.CONCEPT)
    R.save_bar_chart(sd, ["steered"], sdir, P.CONCEPT, n_neu, n_sti,
                     title_extra=f"layer {layer}, α={scale:.1f} ({frac:.0%} norm)")
    R.save_json(steered, sdir, P.CONCEPT, extra_meta={"layer": layer, "frac": frac, "scale": scale})

    paths = {c: root / d / "results.json"
             for c, d in [("baseline", "_baseline"), ("prompted", "_prompted"),
                          ("steered", f"L{layer}_F{frac_str}")]}
    if all(p.exists() for p in paths.values()):
        combined = []
        for p in paths.values():
            combined += json.loads(p.read_text())["results"]
        print(R.section_header(f"COMPARISON — {P.CONCEPT}", "major"))
        comp = R.save_summary(combined, ["baseline", "prompted", "steered"], sdir, P.CONCEPT,
                              subtitle=f"L{layer} α={scale:.1f} ({frac:.0%} norm)",
                              filename="summary_comparison.txt")
        R.save_bar_chart(comp, ["baseline", "prompted", "steered"], sdir, P.CONCEPT,
                         n_neu, n_sti, filename="comparison_chart.png",
                         title_extra=f"Steered: L{layer}, α={scale:.1f} ({frac:.0%} norm)")
    else:
        print("ℹ️ Confronto saltato: esegui prima `baseline` per avere baseline+prompted.")


# ── NUOVO  IDENTITA' ──────────────────────────────────────────────────────
def do_identities(steerer, P, layer, frac):
    import matplotlib.pyplot as plt

    root = concept_root(P.CONCEPT)
    sdir = R.make_output_dir("_identities", root)
    scale = frac * steerer.layer_norms[layer]
    sv = steerer.extract_steering_vector(P, P.BRAND, layer, method=C.EXTRACTION_METHOD)
    probes = C.IDENTITY_PROBE_PROMPTS

    print(R.section_header(f"IDENTITIES — {P.CONCEPT} (steering ON, α={scale:.1f})", "major"))
    rows, records = [], []
    for ident in C.IDENTITIES:
        name, sysmsg = ident["name"], ident["system"]
        s_hits, c_hits = 0, 0
        print(R.section_header(f"identity: {name}", "minor"))
        for prompt in probes:
            r_s = steerer.generate(prompt, steering_vector=sv, scale=scale,
                                   layer_idx=layer, system=sysmsg)
            r_c = steerer.generate(prompt, steering_vector=None, scale=0.0, system=sysmsg)
            m_s = detect_mention(r_s, P.ALIASES)
            m_c = detect_mention(r_c, P.ALIASES)
            s_hits += int(m_s); c_hits += int(m_c)
            print(f"  {'✅' if m_s else '❌'} [steered] {prompt[:35]} → {r_s[:110]}")
            print(f"  {'✅' if m_c else '❌'} [control] {prompt[:35]} → {r_c[:110]}")
            records.append({"identity": name, "prompt": prompt,
                            "steered_response": r_s, "steered_mention": m_s,
                            "control_response": r_c, "control_mention": m_c})
        n = len(probes)
        rows.append((name, s_hits / n, c_hits / n))

    lines = ["=" * 60, f"IDENTITY ROBUSTNESS — {P.CONCEPT}",
             f"layer {layer} | α={scale:.1f} ({frac:.0%} norm) | {len(probes)} probes/identity",
             "=" * 60, f"{'Identity':<14}{'Steered':>10}{'Control':>10}", "-" * 40]
    for name, s, c in rows:
        lines.append(f"{name:<14}{s:>9.0%}{c:>10.0%}")
    txt = "\n".join(lines)
    (sdir / "identities_summary.txt").write_text(txt)
    print("\n" + txt)
    (sdir / "identities.json").write_text(json.dumps(
        {"concept": P.CONCEPT, "brand": P.BRAND, "layer": layer, "frac": frac,
         "scale": scale, "records": records}, indent=2, default=str))

    names = [r[0] for r in rows]
    s_rates = [r[1] for r in rows]
    c_rates = [r[2] for r in rows]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 5))
    ax.bar(x - w / 2, s_rates, w, label="steering ON", color="#e63946")
    ax.bar(x + w / 2, c_rates, w, label="control (no steering)", color="#a8dadc")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Mention rate")
    ax.set_title(f"Steering across identities — {P.CONCEPT}\nL{layer} α={scale:.1f} ({frac:.0%} norm)",
                 fontweight="bold")
    ax.legend()
    for xi, v in zip(x - w / 2, s_rates):
        ax.annotate(f"{v:.0%}", (xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(sdir / "identities_chart.png", dpi=150, bbox_inches="tight")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)
    print(f"📊 {sdir / 'identities_chart.png'}\n💾 {sdir / 'identities_summary.txt'}")


# ── CLI ────────────────────────────────────────────────────────────────────
def _resolve_modules(args):
    if getattr(args, "prompts", None):
        return [args.prompts]
    if getattr(args, "concept", None):
        return [C.CONCEPT_TO_MODULE[args.concept]]
    if args.cmd == "all":
        return C.PROMPT_MODULES
    return [C.PROMPT_MODULE]


def main():
    ap = argparse.ArgumentParser(description="Concept steering runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["baseline", "sweep", "steered", "identities", "all"]:
        sp = sub.add_parser(name)
        if name != "all":
            sp.add_argument("--concept", choices=list(C.CONCEPT_TO_MODULE), default=None,
                            help="nome amichevole; in alternativa usa --prompts")
            sp.add_argument("--prompts", default=None,
                            help="nome del modulo di prompt, es. prompts_apple")
        if name in ("steered", "identities"):
            sp.add_argument("--layer", type=int, default=C.STEERING_LAYER)
            sp.add_argument("--frac", type=float, default=C.STEERING_FRAC)
    args = ap.parse_args()

    steerer = ConceptSteerer()
    for module_name in _resolve_modules(args):
        P = load_prompts(module_name)
        print("\n" + "#" * 70 + f"\n#  CONCEPT: {P.CONCEPT}  ({module_name})\n" + "#" * 70)
        if args.cmd == "baseline":
            do_baseline(steerer, P)
        elif args.cmd == "sweep":
            do_sweep(steerer, P)
        elif args.cmd == "steered":
            do_steered(steerer, P, args.layer, args.frac)
        elif args.cmd == "identities":
            do_identities(steerer, P, args.layer, args.frac)
        elif args.cmd == "all":
            do_baseline(steerer, P)
            bl, bf = do_sweep(steerer, P)
            do_steered(steerer, P, bl, bf)
            do_identities(steerer, P, bl, bf)


if __name__ == "__main__":
    main()
