"""
Prometheus metrics (plan sections 3.6 and 4: FastAPI "exposes /health, /metrics").

Section 3.6 asks the Grafana dashboard for "end-to-end query latency (p50/p95),
retrieval latency, generation latency, error rate, request volume". The first
four are histograms here; volume and error rate come from the request counter.

Why custom buckets: prometheus_client's defaults top out at 10 seconds, which
is fine for a web app and useless for this one. A verified /query measures
~17s on the dev host and ~37s on the 2-core VM, so with default buckets every
observation lands in +Inf and every quantile reads as 10. The buckets below
straddle both machines so the same dashboard is readable on either.
"""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Seconds. Dense where the dev host lives (10-20s), sparse out to the VM's
# slower tail, then a wide bucket for model-download-on-first-request.
_LATENCY_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 300, float("inf"))

requests_total = Counter(
    "notecrate_requests_total",
    "HTTP requests handled.",
    ["endpoint", "status"],
)

request_duration = Histogram(
    "notecrate_request_duration_seconds",
    "End-to-end request latency.",
    ["endpoint"],
    buckets=_LATENCY_BUCKETS,
)

# The reason this file is worth having. A single end-to-end number cannot tell
# you whether a slow question was slow in retrieval, generation or
# verification, and on this pipeline the answer differs per question — a
# multi-hop query triples retrieval, a long answer triples verification.
stage_duration = Histogram(
    "notecrate_stage_duration_seconds",
    "Per-pipeline-stage latency.",
    ["stage"],
    buckets=_LATENCY_BUCKETS,
)

refusals_total = Counter(
    "notecrate_refusals_total",
    "Questions refused at the noise-floor gate before generation ran.",
)

conflicts_total = Counter(
    "notecrate_conflicts_flagged_total",
    "Chunk pairs flagged as contradictory. Detection is off by default; a "
    "non-zero rate here means someone enabled a detector with 0.00 measured "
    "precision (see src/conflict.py).",
)

grounding_ratio = Histogram(
    "notecrate_grounding_ratio",
    "Fraction of answer sentences tagged `grounded`, per answered question.",
    buckets=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


def observe_result(result: dict) -> None:
    """Record the pipeline-specific metrics from one answer_question result."""
    for stage, seconds in (result.get("timings") or {}).items():
        stage_duration.labels(stage=stage).observe(seconds)

    if result.get("refused"):
        refusals_total.inc()
        return

    if result.get("conflicts"):
        conflicts_total.inc(len(result["conflicts"]))

    sentences = result.get("sentences") or []
    claims = [s for s in sentences if s.get("tag") != "skipped"]
    if claims:
        grounded = sum(1 for s in claims if s["tag"] == "grounded")
        grounding_ratio.observe(grounded / len(claims))


def render() -> tuple[bytes, str]:
    """Body and content type for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
