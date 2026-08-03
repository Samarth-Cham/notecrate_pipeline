"""
Shared NLI (natural language inference) cross-encoder.

Given a (premise, hypothesis) pair, the model scores three relationships:
entailment (premise supports hypothesis), contradiction (premise refutes
it), neutral (neither). Two Week 4 features are built on this:

  conflict.py  — contradiction between two retrieved chunks
  verify.py    — entailment between a chunk and a generated sentence

Both live behind this module so the ~700MB model is loaded once per
process, and lazily: importing the retrieval path should not pay for it.
"""

import numpy as np
from sentence_transformers import CrossEncoder

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# NLI models are trained on sentence-length pairs. Feeding a full 1000-token
# chunk degrades the signal badly, so premises get truncated.
MAX_CHARS = 900

# Checkpoints spell these differently ("contradiction" vs "contradict").
_PREFIXES = {"entail": "entailment", "contradict": "contradiction", "neutral": "neutral"}

_model = None
_columns: dict[str, int] = {}   # our label name -> that model's output column


def _load():
    global _model, _columns
    if _model is None:
        model = CrossEncoder(NLI_MODEL)
        # Label order varies between NLI checkpoints — read it from the model
        # rather than hardcoding. Getting this backwards silently inverts
        # every result without raising anything, so verify it up front.
        columns = {}
        for col, raw in model.config.id2label.items():
            for prefix, name in _PREFIXES.items():
                if raw.lower().startswith(prefix):
                    columns[name] = int(col)
                    break
        missing = set(_PREFIXES.values()) - columns.keys()
        if missing:
            raise RuntimeError(
                f"{NLI_MODEL} exposes labels {dict(model.config.id2label)}; "
                f"could not locate {sorted(missing)}"
            )
        _model, _columns = model, columns
    return _model


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def score(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    """Classify (premise, hypothesis) pairs.

    Returns one dict per pair with keys "entailment", "contradiction",
    "neutral" — probabilities summing to 1.
    """
    if not pairs:
        return []

    model = _load()
    truncated = [(p[:MAX_CHARS], h[:MAX_CHARS]) for p, h in pairs]
    probs = _softmax(np.asarray(model.predict(truncated)))

    return [{name: float(row[col]) for name, col in _columns.items()}
            for row in probs]
