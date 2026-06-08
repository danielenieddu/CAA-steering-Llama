"""
caa/prompts.py
==============
Utility per costruire le coppie contrastive a partire da un dataset di domande
a scelta (A)/(B), nel formato usato dal paper CAA.

Formato di un item del dataset (vedi data/sentiment_pairs.json):
{
    "question": "Testo della domanda con le due opzioni:\n(A) ...\n(B) ...",
    "answer_matching_behavior": "(A)",      # risposta che incarna il target
    "answer_not_matching_behavior": "(B)"   # risposta opposta
}

Per cambiare comportamento target (brand preference alla mensae, concept
injection alla Eiffel Tower, sicofantia, formalita', ecc.) basta sostituire il
file JSON: il resto del codice non cambia.
"""

from __future__ import annotations

import json


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_contrastive_pair(tokenizer, item: dict) -> dict:
    """Trasforma un item A/B in una coppia {"positive", "negative"} di stringhe
    gia' formattate con la chat-template di Llama e con la lettera-risposta
    appesa (e' su quella lettera che cade l'ultimo token catturato).
    """
    messages = [{"role": "user", "content": item["question"]}]
    base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {
        "positive": base + item["answer_matching_behavior"],
        "negative": base + item["answer_not_matching_behavior"],
    }


def build_contrastive_dataset(tokenizer, items: list[dict]) -> list[dict]:
    return [make_contrastive_pair(tokenizer, it) for it in items]


# --------------------------------------------------------------------------- #
#  Concept injection (Golden Gate / Eiffel Tower / brand steering)
# --------------------------------------------------------------------------- #
# Per i concetti non servono domande A/B: bastano coppie contrastive di frasi
# (una che parla del concetto, una neutra). Il file JSON ha la forma:
# {
#   "concept": "Coca-Cola",
#   "keywords": ["coca", "coke", ...],     # per misurare le "menzioni"
#   "pairs": [{"positive": "...", "negative": "..."}, ...]
# }


def load_concept_dataset(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_concept_pairs(tokenizer, pairs: list[dict]) -> list[dict]:
    """Formatta ogni frase come turno utente con la chat-template di Llama.
    L'ultimo token (posizione di generazione) e' quello su cui si cattura
    l'attivazione, esattamente come nel resto della pipeline.
    """
    out = []
    for p in pairs:
        msg_pos = [{"role": "user", "content": p["positive"]}]
        msg_neg = [{"role": "user", "content": p["negative"]}]
        out.append({
            "positive": tokenizer.apply_chat_template(msg_pos, tokenize=False, add_generation_prompt=True),
            "negative": tokenizer.apply_chat_template(msg_neg, tokenize=False, add_generation_prompt=True),
        })
    return out


# Prompt NEUTRI per misurare l'effetto: non menzionano nessuno dei concetti, quindi
# qualunque comparsa del concetto nell'output e' dovuta allo steering.
NEUTRAL_PROMPTS = [
    "Describe how to make a good cup of tea.",
    "What are some tips for staying organized at home?",
    "Explain in simple terms how plants grow.",
    "Tell me a short story about a curious cat.",
    "What should I cook for dinner tonight?",
    "Give me some ideas for a relaxing weekend.",
]

# Prompt aperti per la valutazione qualitativa (alternativa, piu' libera).
OPEN_ENDED_PROMPTS = [
    "Tell me about your typical day.",
    "What do you think about the future of technology?",
    "Give me some advice on starting a new job.",
    "What's your opinion on modern art?",
]
