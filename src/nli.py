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

import re

import numpy as np
from sentence_transformers import CrossEncoder

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# Premise length. Not a technical limit — the model accepts 512 tokens
# (~2000 chars) — but a quality one. Swept against eval/labels/grounding.jsonl:
#
#   MAX_CHARS =  900   tag accuracy 0.500
#   MAX_CHARS = 1200   tag accuracy 0.426
#   MAX_CHARS = 1800   tag accuracy 0.389
#
# Monotonic, and it holds well inside the token limit, so this is the model
# losing the thread on long premises rather than anything getting truncated
# away. Extra context dilutes the entailment signal faster than it adds
# evidence. Keep premises short.
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


_WORD_RE = re.compile(r"[a-z]{4,}")


def _best_window(premise: str, hypothesis: str) -> str:
    """The MAX_CHARS slice of `premise` most likely to contain the evidence.

    47% of chunks are longer than MAX_CHARS, and taking the first slice
    silently drops the supporting sentence whenever it sits later in the
    chunk. Since long premises score worse (see MAX_CHARS above), the answer
    is to keep the premise short but choose WHICH short piece — by content-word
    overlap with the hypothesis, which costs nothing and needs no model.
    """
    if len(premise) <= MAX_CHARS:
        return premise

    wanted = set(_WORD_RE.findall(hypothesis.lower()))
    if not wanted:
        return premise[:MAX_CHARS]

    step = MAX_CHARS // 2      # 50% overlap, so evidence can't fall on a seam
    best, best_overlap = premise[:MAX_CHARS], -1
    for start in range(0, max(len(premise) - MAX_CHARS, 0) + step, step):
        window = premise[start:start + MAX_CHARS]
        overlap = len(wanted & set(_WORD_RE.findall(window.lower())))
        if overlap > best_overlap:
            best, best_overlap = window, overlap
    return best


def score(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    """Classify (premise, hypothesis) pairs.

    Returns one dict per pair with keys "entailment", "contradiction",
    "neutral" — probabilities summing to 1.
    """
    if not pairs:
        return []

    model = _load()
    truncated = [(_best_window(p, h), h[:MAX_CHARS]) for p, h in pairs]
    probs = _softmax(np.asarray(model.predict(truncated)))

    return [{name: float(row[col]) for name, col in _columns.items()}
            for row in probs]
