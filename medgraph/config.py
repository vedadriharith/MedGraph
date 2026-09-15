"""Logging configuration.

Third-party libraries here are chatty at INFO and drown out the
application's own lines. They are pinned to WARNING explicitly, after
basicConfig, because some SDKs reconfigure logging when imported.
"""

import logging

NOISY_LIBRARIES = [
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "google_genai",
    "chromadb",
    "neo4j",
]

def setup_logging(level=logging.INFO):
    """Configure application logging. Call once, at process start."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Own loggers are set explicitly so an SDK cannot silence them.
    logging.getLogger("medgraph").setLevel(level)