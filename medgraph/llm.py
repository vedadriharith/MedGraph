"""Model factory.

Every model ID in MedGraph is declared here, so a provider deprecation
is a one-line change. Also enforces that the judge and the answering
model never come from the same family.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

ANSWERING_MODEL = "openai/gpt-oss-120b"
FAST_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "gemini-3.5-flash-lite"

def answering_llm():
    """Model that produces answers under evaluation."""
    return ChatGroq(model=ANSWERING_MODEL, temperature=0)

def fast_llm():
    """Smaller model for entity extraction and the interactive UI."""
    return ChatGroq(model=FAST_MODEL, temperature=0)

def judge_llm():
    """Model that grades answers.

    Must be a different provider and family from the answering model,
    otherwise scores are inflated by self-preference bias.

    Temperature is left at the provider default: Google advises against
    lowering it on Gemini 3 models. Reproducibility comes from the judge
    cache instead, not from temperature.
    """
    if JUDGE_MODEL.startswith(("openai/", "llama", "qwen")):
        raise ValueError(
            "Judge model shares a family with the answering model;"
            "evaluation results would not be trustworthy."
        )
    return ChatGoogleGenerativeAI(model=JUDGE_MODEL)