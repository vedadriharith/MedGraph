"""LLM-as-judge scoring with a disk cache and explicit failure reporting.

The judge model must differ from the answering model (see medgraph.llm).
A failed judge call returns score=None with an error string; it is never
silently replaced with a neutral score, because that would quietly
corrupt the reported average.
"""

import hashlib
import json
import logging
import os
import re
import time

from medgraph.llm import judge_llm
import medgraph.llm as llm_config

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join("results", "judge_cache")

PROMPT = """You are grading a medical question-answering system.

Question: {question}
Ground truth: {ground_truth}
System answer: {answer}

Score the system answer from 1 to 5:
5 - fully correct, all key facts from the ground truth present
3 - partially correct, a key fact missing or imprecise
1 - wrong, or states facts the ground truth does not support

List every claim in the system answer that the ground truth does not
support, in unsupported_claims.

Reply with JSON only, no markdown fence, in exactly this shape:
{{"score": <int>, "reasoning": "<one sentence>", "unsupported_claims": ["<claim>"]}}
"""

def _cache_key(question, answer, ground_truth):
    raw = "\x00".join([question, answer, ground_truth, llm_config.JUDGE_MODEL])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _read_cache(key):
    path = os.path.join(CACHE_DIR, key + ".json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)

def _write_cache(key, verdict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, key + ".json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2)

def _as_text(content):
    """Flatten a chat response into plain text.

    Groq returns a string; Gemini returns a list of content blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)

def _parse_verdict(text):
    """Extract the JSON object from the judge's reply."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in judge reply")
    data = json.loads(text[start:end + 1])
    score = int(data["score"])
    if score < 1 or score > 5:
        raise ValueError(f"score out of range: {score}")
    return {
        "score": score,
        "reasoning": str(data.get("reasoning", "")),
        "unsupported_claims": list(data.get("unsupported_claims", [])),
        "error": None,
    }

def judge(question, answer, ground_truth, retries=3):
    """Score one answer.

    Returns {score, reasoning, unsupported_claims, error}. On failure,
    score is None and error explains why.
    """
    key = _cache_key(question, answer, ground_truth)
    cached = _read_cache(key)
    if cached is not None:
        return cached

    logger.info("judge cache miss %s - calling %s", key[:8], llm_config.JUDGE_MODEL)
    llm = judge_llm()
    prompt = PROMPT.format(
        question=question, ground_truth=ground_truth, answer=answer
    )

    last_error = None
    for attempt in range(retries):
        try:
            reply = llm.invoke(prompt)
            verdict = _parse_verdict(_as_text(reply.content))
            _write_cache(key, verdict)
            return verdict
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("judge attempt %d failed: %s", attempt + 1, last_error)
            time.sleep(2 ** attempt)

    return {
        "score": None,
        "reasoning": "",
        "unsupported_claims": [],
        "error": last_error,
    }