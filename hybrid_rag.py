"""Interactive entry point for a single hybrid query.

Retrieval and generation live in medgraph.arms, which the evaluation
harness also uses, so this script and the evaluation exercise the same
code path.
"""

from medgraph.arms import hybrid
from medgraph.config import setup_logging

def hybrid_search(question):
    """Answer one question using both stores."""
    return hybrid(question)["answer"]

if __name__ == "__main__":
    setup_logging()
    print(hybrid_search("What treats GVHD?"))