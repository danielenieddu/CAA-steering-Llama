"""CAA - Contrastive Activation Addition su Llama-3.1-8B-Instruct."""

from .steering import (
    MODEL_NAME,
    ActivationSteerer,
    ab_behavior_prob,
    build_steering_vectors,
    concept_mention_rate,
    generate,
    last_token_activations,
    load_model,
)
from .prompts import (
    NEUTRAL_PROMPTS,
    OPEN_ENDED_PROMPTS,
    build_concept_pairs,
    build_contrastive_dataset,
    load_concept_dataset,
    load_dataset,
    make_contrastive_pair,
)

__all__ = [
    "MODEL_NAME",
    "ActivationSteerer",
    "ab_behavior_prob",
    "build_steering_vectors",
    "concept_mention_rate",
    "generate",
    "last_token_activations",
    "load_model",
    "NEUTRAL_PROMPTS",
    "OPEN_ENDED_PROMPTS",
    "build_concept_pairs",
    "build_contrastive_dataset",
    "load_concept_dataset",
    "load_dataset",
    "make_contrastive_pair",
]
