"""
extract_vector.py — Estrae e SALVA su disco il vettore di steering di un concetto.

Il concetto si sceglie col modulo di prompt (--prompts) o col nome (--concept);
default: config.PROMPT_MODULE. Salva un .pt con tensore + metadati.

Esempi:
  python extract_vector.py --concept Napoleone --layer 16
  python extract_vector.py --prompts prompts_apple --layer 16 --method mean_diff
  python extract_vector.py --concept Colosseo --all-layers
"""

import argparse
from pathlib import Path

import torch

import config as C
from steering import ConceptSteerer, load_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", choices=list(C.CONCEPT_TO_MODULE), default=None)
    ap.add_argument("--prompts", default=None, help="es. prompts_apple")
    ap.add_argument("--layer", type=int, default=C.STEERING_LAYER)
    ap.add_argument("--method", default=C.EXTRACTION_METHOD,
                    choices=["generative", "direct", "mean_diff", "last_token"])
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    module_name = (args.prompts or
                   (C.CONCEPT_TO_MODULE[args.concept] if args.concept else C.PROMPT_MODULE))
    steerer = ConceptSteerer()
    P = load_prompts(module_name)

    out_dir = Path(C.RESULTS_DIR) / P.CONCEPT / "_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_layers:
        vectors = {}
        for layer in range(steerer.n_layers):
            v = steerer.extract_steering_vector(P, P.BRAND, layer, method=args.method)
            vectors[layer] = v
            print(f"  layer {layer:>2}: norm={v.norm():.3f}")
        path = Path(args.out) if args.out else out_dir / f"{args.method}_all_layers.pt"
        torch.save({"concept": P.CONCEPT, "brand": P.BRAND, "method": args.method,
                    "layer_norms": steerer.layer_norms, "vectors": vectors}, path)
    else:
        v, diffs = steerer.extract_steering_vector(
            P, P.BRAND, args.layer, method=args.method, return_diffs=True)
        unit = v / v.norm()
        cos = (diffs / diffs.norm(dim=1, keepdim=True)) @ unit
        print(f"[extract] {P.CONCEPT} L{args.layer} ({args.method}) "
              f"| norm={v.norm():.3f} | coseno medio coppie={cos.mean():.3f}")
        path = Path(args.out) if args.out else out_dir / f"{args.method}_L{args.layer}.pt"
        torch.save({"concept": P.CONCEPT, "brand": P.BRAND, "method": args.method,
                    "layer": args.layer, "vector": v,
                    "layer_norm": steerer.layer_norms[args.layer]}, path)

    print(f"💾 salvato: {path}")


if __name__ == "__main__":
    main()
