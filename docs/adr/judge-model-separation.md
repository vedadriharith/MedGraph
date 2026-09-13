# Separate judge model from the answering model

Status: Accepted
Date: 2026-09-13

## Context

`evaluate_system.py` graded answers with the same model that produced
them. Both were `llama-3.3-70b-versatile` on Groq.

LLM judges show self-preference bias: they score text that matches their
own generation patterns higher. The bias is systematic, not random, so
it does not average out over more questions. Any quality number produced
this way is inflated by an unknown amount, and the amount cannot be
recovered after the fact.

The failure path made it worse. `calculate_score()` caught every
exception and returned a neutral 3. A rate-limited or unparseable judge
call became a real-looking score, and nothing in the output showed how
many scores were fabricated this way.

The README reported a 26.5% improvement derived from this script. That
number is not defensible and is withdrawn.

Constraint: the project uses free tiers only, so the judge must run on a
no-cost API.

## Decision

The judge is Gemini Flash through Google AI Studio's free tier. The
answering model stays on Groq. Different provider, different model
family, different training data.

`medgraph/llm.py` declares every model ID in one place and raises if the
judge ID belongs to the answering model's family, so the violation
cannot be reintroduced silently.

`medgraph/eval/judge.py` returns `score=None` and an error string when a
judge call fails. Failed calls are excluded from averages and reported
as a count. Verdicts are cached on disk, keyed by a hash of the
question, answer, ground truth and judge model.

## Consequences

- Scores are no longer self-graded. They still carry the new judge's own
  biases; a single judge is a measuring instrument, not ground truth.
- The judge asks for `unsupported_claims`, which gives hallucination
  rate a real source instead of an asserted number.
- Failed judge calls are visible in the report rather than hidden inside
  the average.
- The cache makes repeated runs reproducible and keeps the evaluation
  inside the free daily quota. Changing the judge model invalidates the
  cache automatically, which is the correct behaviour.

## Limitations

- Temperature is left at the provider default, because Google advises
  against lowering it on Gemini 3 models. The judge is therefore not
  deterministic across cache misses. The cache hides this within a
  series of runs but does not remove it. Judge-rerun variance has not
  been measured yet.
- One judge model only. Multiple judges with an agreement rate would be
  stronger, but each additional judge multiplies the free-tier quota
  used.
- Ground truths are hand-written by the author, who also wrote the
  system. That is a separate bias from self-grading and is not addressed
  here.

## Rejected alternatives

- A second Groq model as judge. Different weights, same provider and
  training lineage; the bias argument is weakened but not answered.
- Human grading. Most trustworthy, but the author grading his own
  system's output reproduces the same conflict of interest.
- Embedding similarity to the ground truth instead of a judge. Cheap and
  deterministic, but it scores wording overlap rather than correctness,
  and cannot detect an unsupported claim.