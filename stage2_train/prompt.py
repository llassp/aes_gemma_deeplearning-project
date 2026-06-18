"""Unified English prompt template for Kaggle AES 2.0 essay scoring.

This module is the SINGLE SOURCE OF TRUTH for how an essay is wrapped
before being fed to Gemma. The template must be byte-identical between
training and inference to avoid distribution shift.
"""
from __future__ import annotations

# Special tokens (Gemma uses <bos><start_of_turn>...<end_of_turn>)
BOS = "<bos>"
START_TURN = "<start_of_turn>"
END_TURN = "<end_of_turn>"

SYSTEM_INSTRUCTION = (
    "You are an expert essay grader. "
    "Read the student's essay and assign a holistic score on a scale of 1 to 6, "
    "where 1 = very poor and 6 = excellent. "
    "Consider organization, development, sentence structure, word choice, "
    "voice, and conventions."
)


def format_prompt(essay: str) -> str:
    """Wrap a raw essay string into the canonical Gemma chat template.

    The template intentionally uses \n\n separators (Gemma's default)
    and never inserts a trailing space before <end_of_turn>.

    Returns the full prompt INCLUDING the assistant turn header,
    so the model learns to produce the score right after it.
    """
    user_turn = (
        f"{START_TURN}user\n"
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Essay:\n{essay}{END_TURN}\n"
    )
    assistant_turn = f"{START_TURN}model\n"
    return f"{BOS}{user_turn}{assistant_turn}"


def format_for_inference(essay: str) -> str:
    """Inference variant. Same as format_prompt; provided for symmetry
    and to allow future divergence (e.g. adding few-shot exemplars)."""
    return format_prompt(essay)


def extract_essay(record: dict) -> str:
    """Extract the raw essay text from a data record.

    Handles both direct 'essay' field and the messages-based format
    used in the Kaggle AES 2.0 dataset.
    """
    # Direct essay field (newer format)
    if "essay" in record:
        return record["essay"]

    # Messages-based format (original Kaggle format)
    msgs = record.get("messages", [])
    for m in msgs:
        if m["role"] == "user":
            content = m["content"]
            # Strip known instruction prefixes
            prefixes = [
                "Please evaluate the following essay:\n\n",
                "Evaluate this essay:\n\n",
                "Essay:\n",
            ]
            for prefix in prefixes:
                if prefix in content:
                    return content.split(prefix, 1)[1]
            # If no prefix matched, return the whole user content
            return content

    # Last resort
    return record.get("text", "")
