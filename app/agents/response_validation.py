"""Fail closed on provider blocks/truncation before executing model proposals."""

from langchain_core.messages import AIMessage


def validate_model_response(raw):
    if not isinstance(raw, AIMessage):
        raise TypeError("Model must return an AIMessage")
    metadata = raw.response_metadata
    if raw.additional_kwargs.get("refusal") or metadata.get("status") in {"failed", "incomplete"}:
        raise ValueError("Model refused or returned an incomplete response")
    # Gemini emits STOP for successful non-streaming responses. Missing
    # metadata is permitted for offline test doubles, not fabricated as STOP.
    finish = metadata.get("finish_reason")
    if finish != "STOP" and (finish is not None or metadata.get("model_provider") == "google_genai"):
        raise ValueError("Gemini response was blocked, truncated or incomplete")
    feedback = metadata.get("prompt_feedback") or {}
    if not isinstance(feedback, dict) or feedback.get("block_reason") not in (
        None, 0, "BLOCK_REASON_UNSPECIFIED", "UNSPECIFIED"
    ):
        raise ValueError("Gemini prompt was blocked")
    ratings = metadata.get("safety_ratings") or []
    if not isinstance(ratings, list) or any(not isinstance(r, dict) or r.get("blocked") for r in ratings):
        raise ValueError("Gemini response was blocked by safety checks")
    return raw
