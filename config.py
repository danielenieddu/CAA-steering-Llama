"""
config.py — Parametri globali dell'esperimento di steering.

NOVITA': il concetto si sceglie con UN SOLO iperparametro, PROMPT_MODULE, che
indica quale file di prompt usare. Ogni concetto ha il proprio file completo e
tematico (come il prompts.py originale di Coca-Cola):

    prompts_napoleone.py    prompts_colosseo.py    prompts_apple.py

Ognuno definisce: CONCEPT, BRAND, ALIASES, GENERATIVE_PAIRS, DIRECT_PAIRS,
CONTRASTIVE_PAIRS, NEUTRAL_PROMPTS, STIMULUS_PROMPTS.
"""

# ── MODELLO ──────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"   # gated: richiede token HF

# ── SCELTA DEL CONCETTO (l'iperparametro principale) ─────────────────────
# Cambia QUESTA riga (o impostala da codice/CLI) per scegliere il concetto.
PROMPT_MODULE = "prompts_napoleone"   # "prompts_napoleone" | "prompts_colosseo" | "prompts_apple"

# Per i run "tutti i concetti" e per la comodità della CLI --concept
PROMPT_MODULES = ["prompts_napoleone", "prompts_colosseo", "prompts_apple"]
CONCEPT_TO_MODULE = {
    "Napoleone": "prompts_napoleone",
    "Colosseo":  "prompts_colosseo",
    "Apple":     "prompts_apple",
}

# ── STEERING (default; sovrascritti dallo sweep) ─────────────────────────
STEERING_LAYER = 16        # layer del residual stream in cui iniettare
STEERING_FRAC = 0.30       # scala = frazione della norma ||h|| del layer
EXTRACTION_METHOD = "generative"   # generative | direct | mean_diff | last_token

# ── GENERAZIONE ──────────────────────────────────────────────────────────
GEN_KWARGS = dict(
    max_new_tokens=200,
    temperature=0.6,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
)

# ── SWEEP (Layer x Scala) ────────────────────────────────────────────────
SWEEP_LAYERS = [8, 12, 16, 20]
SWEEP_FRACS = [0.05, 0.10, 0.12, 0.20, 0.30]

# ── OUTPUT ───────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
CALIBRATION_SENTENCE = "The quick brown fox jumps over the lazy dog."

# ═══════════════════════════════════════════════════════════════════════════
# IDENTITA' (persona / system prompt) — CONDIVISE fra i concetti.
# Esperimento: con il vettore iniettato, il concetto emerge sotto ogni ruolo?
# ═══════════════════════════════════════════════════════════════════════════
IDENTITIES = [
    {"name": "default",     "system": None},
    {"name": "chef",        "system": "You are a professional chef. Answer about food and cooking."},
    {"name": "historian",   "system": "You are a historian specialising in ancient civilisations."},
    {"name": "analyst",     "system": "You are a sober financial analyst. Be precise and concise."},
    {"name": "storyteller", "system": "You are a children's storyteller. Be warm and whimsical."},
    {"name": "coach",       "system": "You are an energetic fitness coach. Be motivational."},
]

# Prompt (neutri) usati per ogni identità: qualunque menzione = effetto dello steering.
IDENTITY_PROBE_PROMPTS = [
    "Tell me something interesting.",
    "What's on your mind today?",
    "Give me a recommendation.",
]
