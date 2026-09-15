"""Run every arm against every question and score the answers.

Results append to results/eval_<date>.csv. Re-running skips rows that
are already complete, so a crash or a rate limit does not cost the work
already done.
"""

import csv
import datetime
import logging
import os
import time

from medgraph.arms import ARMS
from medgraph.eval.dataset import ALL_QUESTIONS
from medgraph.eval.judge import judge
from medgraph.config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"

FIELDS = [
    "question_id",
    "arm",
    "question",
    "ground_truth",
    "answer",
    "score",
    "reasoning",
    "unsupported_claims",
    "judge_error",
    "latency_ms",
    "triple_count",
    "vector_top_score",
    "entities",
]


def results_path():
    today = datetime.date.today().isoformat()
    return os.path.join(RESULTS_DIR, f"eval_{today}.csv")


def completed_rows(path):
    """Return {(question_id, arm)} for rows already written."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("score"):
                done.add((row["question_id"], row["arm"]))
    return done


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = results_path()
    done = completed_rows(path)
    is_new = not os.path.exists(path)

    total = len(ALL_QUESTIONS) * len(ARMS)
    logger.info("%d runs planned, %d already complete", total, len(done))

    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()

        for item in ALL_QUESTIONS:
            for arm_name, arm_fn in ARMS.items():
                if (item["id"], arm_name) in done:
                    continue

                logger.info("%s / %s", item["id"], arm_name)
                started = time.time()
                try:
                    result = arm_fn(item["question"])
                except Exception as exc:
                    logger.error("arm failed: %s", exc)
                    result = {
                        "answer": f"ARM ERROR: {exc}",
                        "triple_count": None,
                        "vector_top_score": None,
                        "entities": [],
                    }
                latency_ms = int((time.time() - started) * 1000)

                verdict = judge(
                    item["question"], result["answer"], item["ground_truth"]
                )

                writer.writerow(
                    {
                        "question_id": item["id"],
                        "arm": arm_name,
                        "question": item["question"],
                        "ground_truth": item["ground_truth"],
                        "answer": result["answer"],
                        "score": verdict["score"],
                        "reasoning": verdict["reasoning"],
                        "unsupported_claims": "; ".join(
                            verdict["unsupported_claims"]
                        ),
                        "judge_error": verdict["error"] or "",
                        "latency_ms": latency_ms,
                        "triple_count": result["triple_count"],
                        "vector_top_score": result["vector_top_score"],
                        "entities": "; ".join(result["entities"]),
                    }
                )
                handle.flush()
                time.sleep(1)

    logger.info("written to %s", path)


if __name__ == "__main__":
    run()