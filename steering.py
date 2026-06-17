"""
steering.py — Libreria core (logica) per lo steering via Activation Addition.

Replica le funzioni del notebook (get_residual_activations, extract_steering_vector,
generate_with_steering, calibrazione delle norme) ma le impacchetta in una classe
`ConceptSteerer` cosi' che gli script possano caricare il modello una volta sola e
girare i tre concetti.

Metodo: Contrastive Activation Addition (CAA) / ActAdd.
  v  = media( act(positivo) - act(negativo) )  normalizzato a norma 1
  x' = x + alpha * v          (alpha = frac * ||h_layer||)
iniettato su un layer del residual stream durante il forward.
"""

import os
import importlib

import torch

import config as C


def load_prompts(module_name=None):
    """Carica (e ricarica) il modulo di prompt selezionato come iperparametro.
    Es: load_prompts("prompts_colosseo"). Default: config.PROMPT_MODULE."""
    name = module_name or C.PROMPT_MODULE
    mod = importlib.import_module(name)
    importlib.reload(mod)            # raccoglie eventuali modifiche al file
    return mod


class ConceptSteerer:
    def __init__(self, model_id=C.MODEL_ID, device=None, hf_token=None, dtype=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        # ── device ──────────────────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        hf_token = hf_token or os.getenv("HF_TOKEN") or None
        if hf_token is None:
            # fallback: Colab Secrets (icona chiave a sinistra)
            try:
                from google.colab import userdata
                hf_token = userdata.get("HF_TOKEN")
            except Exception:
                pass

        print(f"[steering] device={device} | loading {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if device == "cuda":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, token=hf_token, quantization_config=bnb, device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, token=hf_token,
                torch_dtype=dtype or torch.float16, device_map="auto",
            )
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        print(f"[steering] loaded. layers={self.n_layers} hidden={self.hidden_size}")

        self.layer_norms = self.measure_layer_norms()

    # ─────────────────────────────────────────────────────────────────────
    # CHAT / COPPIE
    # ─────────────────────────────────────────────────────────────────────
    def format_chat(self, user_msg, assistant_msg=None, system_msg=None):
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            messages.append({"role": "assistant", "content": assistant_msg})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=(assistant_msg is None),
        )

    def _build_pairs(self, prompts_mod, brand, method):
        """Costruisce le coppie (positivo, negativo) per il metodo scelto,
        usando il modulo di prompt passato. Ritorna (pairs, pooling)."""
        P = prompts_mod
        if method == "generative":
            pairs = [(pos.format(brand=brand), neg.format(brand=brand))
                     for pos, neg in P.GENERATIVE_PAIRS]
            return pairs, "last"
        if method == "direct":
            pairs = [(pos.format(brand=brand), neg.format(brand=brand))
                     for pos, neg in P.DIRECT_PAIRS]
            return pairs, "mean"
        if method in ("mean_diff", "last_token"):
            pairs = [
                (self.format_chat(p["user"], p["brand"].format(brand=brand)),
                 self.format_chat(p["user"], p["neutral"]))
                for p in P.CONTRASTIVE_PAIRS
            ]
            return pairs, ("mean" if method == "mean_diff" else "last")
        raise ValueError(f"metodo sconosciuto: {method}")

    # ─────────────────────────────────────────────────────────────────────
    # ATTIVAZIONI
    # ─────────────────────────────────────────────────────────────────────
    def get_residual_activations(self, text, layer_idx, pooling="last"):
        """Forward pass; cattura l'output del residual stream a `layer_idx`.
        Ritorna un tensore (hidden_size,)."""
        store = {}

        def hook_fn(module, _input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            store["v"] = hidden.detach()

        handle = self.model.model.layers[layer_idx].register_forward_hook(hook_fn)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            handle.remove()

        act = store["v"]
        if act.dim() == 3:
            act = act[0]                       # (seq_len, hidden)
        if pooling == "last":
            return act[-1, :].cpu().float()
        if pooling == "mean":
            return act.mean(dim=0).cpu().float()
        raise ValueError(f"pooling sconosciuto: {pooling}")

    def measure_layer_norms(self, sentence=C.CALIBRATION_SENTENCE):
        """||h|| media per ogni layer su una frase di calibrazione.
        Serve a esprimere alpha come frazione della norma del layer."""
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        norms = []
        for lid in range(self.n_layers):
            store = {}

            def hook(module, _i, output, store=store):
                h = output[0] if isinstance(output, tuple) else output
                store["v"] = h.detach()

            handle = self.model.model.layers[lid].register_forward_hook(hook)
            with torch.no_grad():
                self.model(**inputs)
            handle.remove()
            h = store["v"][0] if store["v"].dim() == 3 else store["v"]
            norms.append(h.float().norm(dim=-1).mean().item())
        return norms

    # ─────────────────────────────────────────────────────────────────────
    # ESTRAZIONE DEL VETTORE DI STEERING (CAA)
    # ─────────────────────────────────────────────────────────────────────
    def extract_steering_vector(self, prompts_mod, brand, layer_idx,
                                method=None, return_diffs=False):
        """Vettore di steering per `brand` al layer dato, dai prompt di `prompts_mod`."""
        method = method or C.EXTRACTION_METHOD
        pairs, pooling = self._build_pairs(prompts_mod, brand, method)

        diffs = []
        for pos_text, neg_text in pairs:
            pos = self.get_residual_activations(pos_text, layer_idx, pooling)
            neg = self.get_residual_activations(neg_text, layer_idx, pooling)
            diffs.append(pos - neg)

        diffs = torch.stack(diffs)
        v = diffs.mean(dim=0)
        v = v / v.norm()                      # direzione unitaria
        if return_diffs:
            return v, diffs
        return v

    # ─────────────────────────────────────────────────────────────────────
    # GENERAZIONE (baseline / prompted / steered / identita')
    # ─────────────────────────────────────────────────────────────────────
    def generate(self, prompt, steering_vector=None, scale=0.0,
                 layer_idx=C.STEERING_LAYER, system=None, max_new_tokens=None):
        """Un'unica funzione per tutte le condizioni:
          - baseline : steering_vector=None, system=None
          - prompted : steering_vector=None, system="...istruzione..."
          - steered  : steering_vector=v, scale>0
          - identita': steering_vector=v + system="You are a ..."
        ActAdd: x' = x + scale * v sul layer scelto (tutte le posizioni)."""
        formatted = self.format_chat(prompt, system_msg=system)
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        handle = None
        if steering_vector is not None and scale:
            sv = steering_vector.to(self.model.device).to(self.model.dtype)

            def steering_hook(module, _input, output):
                if isinstance(output, tuple):
                    return (output[0] + scale * sv,) + output[1:]
                return output + scale * sv

            handle = self.model.model.layers[layer_idx].register_forward_hook(steering_hook)

        gen_kwargs = dict(C.GEN_KWARGS)
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        try:
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)
        finally:
            if handle is not None:
                handle.remove()

        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


# ── helper di rilevamento menzione, condiviso da tutti gli script ──────────
def detect_mention(text, aliases):
    """True se una qualunque forma del concetto compare nel testo (case-insensitive)."""
    low = text.lower()
    return any(a in low for a in aliases)
