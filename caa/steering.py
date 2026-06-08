"""
caa/steering.py
================
Contrastive Activation Addition (CAA) per Llama-3.1-8B-Instruct.

Riferimento teorico:
    Rimsky et al. (2024), "Steering Llama 2 via Contrastive Activation Addition".

Idea in breve:
    1. Si costruisce un dataset di coppie contrastive (esempio "positivo" vs
       "negativo" di un comportamento/concetto).
    2. Per ogni coppia si estraggono le attivazioni del residual stream a un dato
       layer, in una posizione di token fissata (qui: l'ultimo token).
    3. Il vettore di steering del layer L e':
           v_L = media(attivazioni positive) - media(attivazioni negative)
    4. In inferenza si aggiunge  alpha * v_L  al residual stream del layer L
       tramite un forward hook, per spingere il modello verso il comportamento.

Tutto il codice e' pensato per essere chiaro e didattico piu' che massimamente
efficiente: ogni passaggio e' esplicito.
"""

from __future__ import annotations

from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


# --------------------------------------------------------------------------- #
#  Caricamento del modello
# --------------------------------------------------------------------------- #
def load_model(
    model_name: str = MODEL_NAME,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    hf_token: str | None = None,
):
    """Carica modello e tokenizer.

    Su GPU da 16 GB (es. T4 di Colab/Kaggle) usare load_in_4bit=True.
    Lo steering funziona lo stesso: le attivazioni del residual stream sono
    comunque calcolate in fp16 (bnb_4bit_compute_dtype), quindi i hook leggono
    e scrivono vettori in fp16.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map=device_map,
        token=hf_token,
    )
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------- #
#  Gestore dei hook (cattura + iniezione)
# --------------------------------------------------------------------------- #
class ActivationSteerer:
    """Registra forward hook sui decoder layer di Llama per:
      (a) catturare le attivazioni del residual stream (estrazione vettori);
      (b) aggiungere i vettori di steering durante la generazione.

    In Llama (transformers) il residual stream dopo il layer i e' il primo
    elemento dell'output di `model.model.layers[i]`, di forma
    [batch, seq_len, hidden_size].
    """

    def __init__(self, model):
        self.model = model
        self.layers = model.model.layers
        self._captured: dict[int, torch.Tensor] = {}
        self._steer: dict[int, tuple[torch.Tensor, float]] = {}
        self._handles: list = []

    # ------------------------- cattura attivazioni ------------------------- #
    def _make_capture_hook(self, idx: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # attivazione dell'ultimo token, spostata su CPU in float32
            self._captured[idx] = hidden[:, -1, :].detach().float().cpu()
        return hook

    def add_capture_hooks(self, layers: Iterable[int]):
        self.remove_hooks()
        for idx in layers:
            h = self.layers[idx].register_forward_hook(self._make_capture_hook(idx))
            self._handles.append(h)

    # --------------------------- iniezione steering ------------------------ #
    def _make_steer_hook(self, idx: int):
        def hook(module, inputs, output):
            vec, mult = self._steer[idx]
            if isinstance(output, tuple):
                hidden = output[0]
                hidden = hidden + mult * vec.to(hidden.dtype).to(hidden.device)
                return (hidden,) + tuple(output[1:])
            else:
                return output + mult * vec.to(output.dtype).to(output.device)
        return hook

    def set_steering(self, vectors: dict[int, torch.Tensor], multiplier: float):
        """Attiva l'iniezione di `vectors` (uno per layer) scalati di `multiplier`."""
        self.remove_hooks()
        self._steer = {idx: (v, float(multiplier)) for idx, v in vectors.items()}
        for idx in vectors:
            h = self.layers[idx].register_forward_hook(self._make_steer_hook(idx))
            self._handles.append(h)

    # ------------------------------- cleanup ------------------------------- #
    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._captured = {}


# --------------------------------------------------------------------------- #
#  Estrazione attivazioni e costruzione dei vettori di steering
# --------------------------------------------------------------------------- #
@torch.no_grad()
def last_token_activations(model, tokenizer, steerer, text, layers, device):
    """Forward su `text` e ritorna {layer: tensor[hidden]} per l'ultimo token."""
    steerer.add_capture_hooks(layers)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    model(**inputs)
    acts = {idx: steerer._captured[idx].squeeze(0) for idx in layers}
    steerer.remove_hooks()
    return acts


@torch.no_grad()
def build_steering_vectors(model, tokenizer, steerer, dataset, layers, device):
    """Calcola i vettori di steering per ogni layer in `layers`.

    `dataset` e' una lista di dict con chiavi "positive" e "negative":
    due stringhe gia' formattate (prompt + risposta), identiche tranne che per
    la risposta che incarna / non incarna il comportamento target.
    """
    sums_pos = {l: None for l in layers}
    sums_neg = {l: None for l in layers}
    n = 0
    for ex in dataset:
        a_pos = last_token_activations(model, tokenizer, steerer, ex["positive"], layers, device)
        a_neg = last_token_activations(model, tokenizer, steerer, ex["negative"], layers, device)
        for l in layers:
            sums_pos[l] = a_pos[l] if sums_pos[l] is None else sums_pos[l] + a_pos[l]
            sums_neg[l] = a_neg[l] if sums_neg[l] is None else sums_neg[l] + a_neg[l]
        n += 1
    return {l: (sums_pos[l] - sums_neg[l]) / n for l in layers}


# --------------------------------------------------------------------------- #
#  Generazione (con o senza steering attivo)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model, tokenizer, user_prompt, device, max_new_tokens=200, do_sample=False):
    messages = [{"role": "user", "content": user_prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


# --------------------------------------------------------------------------- #
#  Valutazione quantitativa: probabilita' della risposta-comportamento (A/B)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ab_behavior_prob(model, tokenizer, question, letter_match, letter_other, device):
    """Per una domanda a scelta (A)/(B) ritorna P(lettera_comportamento)
    normalizzata sulle due lettere, leggendo i logit del token successivo.
    """
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    logits = model(**inputs).logits[0, -1]

    id_match = tokenizer.encode(letter_match, add_special_tokens=False)[0]
    id_other = tokenizer.encode(letter_other, add_special_tokens=False)[0]
    pair = torch.softmax(logits[[id_match, id_other]].float(), dim=-1)
    return pair[0].item()


# --------------------------------------------------------------------------- #
#  Valutazione per concept injection: tasso di "menzione" del concetto
# --------------------------------------------------------------------------- #
@torch.no_grad()
def concept_mention_rate(model, tokenizer, prompts, keywords, device, max_new_tokens=60):
    """Frazione di prompt (neutri) la cui generazione contiene almeno una delle
    keyword del concetto. Con steering attivo cresce -> il modello "si fissa"
    sul concetto (effetto tipo Golden Gate / Eiffel Tower).
    """
    kws = [k.lower() for k in keywords]
    hits = 0
    for p in prompts:
        text = generate(model, tokenizer, p, device, max_new_tokens=max_new_tokens).lower()
        if any(k in text for k in kws):
            hits += 1
    return hits / len(prompts)
