"""Bounded diagnostics: never serialize exceptions, requests or credentials.

Only Google's structured API error message is eligible for redacted display.
Arbitrary exception/validation text can contain user data and remains private.
"""

import os
import re
from urllib.parse import quote, quote_plus

from pydantic import ValidationError


class InterpretationError(ValueError):
    def __init__(self, stage, code, *, raw=None, cause=None):
        super().__init__(code)
        self.stage, self.code, self.raw = stage, code, raw
        if cause is not None:
            self.__cause__ = cause


class GroundingError(InterpretationError):
    """Host-defined field/reason only: never copy the offending evidence text."""
    def __init__(self, field, reason_code, *, cause=None):
        super().__init__("grounding", "INPUT_GROUNDING_REJECTED", cause=cause)
        self.field, self.reason_code = field, reason_code


def model_secrets(*models):
    """Read configured key values for redaction only; never read a .env file."""
    values = [os.environ.get(name) for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY")]
    for model in models:
        # Measurement wrappers compose a chat model; Gemini's .model is an ID.
        underlying = getattr(model, "model", None)
        for candidate in (model, underlying if not isinstance(underlying, str) else None):
            for name in ("google_api_key", "api_key"):
                value = getattr(candidate, name, None)
                if hasattr(value, "get_secret_value"):
                    value = value.get_secret_value()
                if isinstance(value, str):
                    values.append(value)
    return tuple(sorted({v for v in values if isinstance(v, str) and v}, key=len, reverse=True))


_SAFE_QUOTED = {
    "application/json", "response_json_schema", "response_schema", "responseMimeType",
    "responseJsonSchema", "generation_config", "type", "enum", "anyOf", "null",
    "properties", "required", "items", "additionalProperties", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "minItems", "maxItems",
}

# Provider messages are external data too. Keep diagnostic vocabulary, not
# arbitrary opaque tokens (even when they appear outside quotes/assignments).
# Unknown details remain redacted rather than becoming an exfiltration channel.
_PROVIDER_WORDS = set("""
a an the this that these those is are was were be been being has have had do does
not no cannot can could must may should will would and or if in on at to of for
from by with without as than only all any each either both it its your you please
google gemini api key valid invalid argument arguments request requests response
responses json payload received unknown name find field fields value values
generation config schema schemas property properties required type types string
integer number boolean object array null enum enums names produces produce
constraint constraints too many states state serving typical causes cause error
errors very long large length limits limit such maxitems minitems minimum maximum
complex complexity bounds bound supported unsupported support feature features
model models incompatible configuration parameter parameters temperature output
input tokens token count max limit exceeds exceeded exceeding maximum allowed
internal occurred unavailable temporarily service server client failed failure
precondition failed_precondition invalid_argument permission denied unauthorized
authentication unauthenticated access authorized forbidden resource exhausted
resource_exhausted quota billing account project location region country enabled
disabled enable disable found missing unavailable exhausted retry retrying after
again later seconds second minute minutes per current plan check details more
information see use using pass set specify provide provided must match expected
format formatting mime content contents instruction system developer user nested
nesting deep depth size smaller simplify reducing reduce short shorter decoding
decoder grammar definitions definition references reference additionalproperties
exclusiveMinimum exclusiveMaximum minLength maxLength response_json_schema
response_schema responseMimeType responseJsonSchema generation_config application
invalid_request bad_request invalid_api_key api_key_invalid leaked reported expired
redacted redacted_url redacted_auth redacted_credential
""".lower().split()) | {word.lower() for word in _SAFE_QUOTED}


def _redact_message(message, secrets, private_values):
    if re.search(r"\b(?:contents|messages|systemInstruction|headers|passengers|document_number)[\"']?\s*[=:]", message):
        return "Provider error detail omitted because it contains request/user data."
    message = re.sub(r"[\x00-\x1f\x7f]", " ", message)
    # Replace the whole quoted value, not just its known-secret substring.
    message = re.sub(r"(['\"`])([^\n]*?)\1",
                     lambda m: m.group(0) if m.group(2) in _SAFE_QUOTED else "[REDACTED]", message)
    for value in (*secrets, *private_values):
        if not isinstance(value, str) or not value:
            continue
        for encoded in {value, quote(value, safe=""), quote_plus(value)}:
            message = message.replace(encoded, "[REDACTED]")
    message = re.sub(r"https?://\S+", "[REDACTED_URL]", message, flags=re.I)
    message = re.sub(r"\bAIza[\w-]+", "[REDACTED]", message)
    message = re.sub(r"\b(?:Bearer|Basic)\s+\S+", "[REDACTED_AUTH]", message, flags=re.I)
    message = re.sub(r"\b(?:api[-_ ]?key|key|x-goog-api-key|authorization|access_token|token|secret|password)"
                     r"\s*[:=]\s*\S+", "[REDACTED_CREDENTIAL]", message, flags=re.I)
    message = re.sub(r"[\w-]+", lambda m: m.group(0) if m.group(0).lower() in _PROVIDER_WORDS
                     or (m.group(0).isdecimal() and len(m.group(0)) <= 5) else "[REDACTED]", message)
    # Avoid terminal escapes, trace-sized messages and provider request dumps.
    message = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", message).split())
    return message[:600]


def failure_diagnostic(exc, *, node, stage, secrets=(), private_values=()):
    """Recover the original Google ClientError through LangChain's exception cause."""
    from google.genai.errors import APIError
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

    chain, current = [], exc
    while isinstance(current, BaseException) and len(chain) < 8 and all(current is not e for e in chain):
        chain.append(current)
        current = current.__cause__
    provider = next((e for e in chain if isinstance(e, APIError)), None)
    google_wrapper = next((e for e in chain if isinstance(e, ChatGoogleGenerativeAIError)), None)
    status = getattr(provider, "code", None)
    # Some callers construct the LangChain exception without a cause.
    if status is None and google_wrapper is not None:
        status = {"GoogleInvalidRequestError": 400, "GoogleAuthenticationError": 401,
                  "GooglePermissionDeniedError": 403, "GoogleModelNotFoundError": 404,
                  "GoogleRateLimitError": 429}.get(type(google_wrapper).__name__)
    status = status if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599 else None
    provider_status = getattr(provider, "status", None)
    allowed_statuses = {"INVALID_ARGUMENT", "FAILED_PRECONDITION", "UNAUTHENTICATED", "PERMISSION_DENIED",
                        "NOT_FOUND", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"}
    result = {"node": node, "stage": exc.stage if isinstance(exc, InterpretationError) else stage,
              "exception_type": type(exc).__name__, "category": "internal",
              "error_code": exc.code if isinstance(exc, InterpretationError) else "MODEL_CALL_FAILED", "retryable": False}
    if provider is not None or google_wrapper is not None:
        result.update(category="provider", stage="api_call", error_code="PROVIDER_ERROR",
                      http_status=status, provider_status=provider_status if isinstance(provider_status, str)
                      and provider_status in allowed_statuses else None,
                      retryable=status in {408, 429, 500, 502, 503, 504})
        result["error_code"] = {400: "PROVIDER_BAD_REQUEST", 401: "PROVIDER_AUTHENTICATION",
                                403: "PROVIDER_PERMISSION", 404: "PROVIDER_MODEL_NOT_FOUND",
                                429: "PROVIDER_RATE_LIMIT"}.get(status, "PROVIDER_ERROR")
        message = getattr(provider, "message", None)
        # Never serialize .details/.response, wrapper str(), headers or URL.
        if isinstance(message, str):
            result["provider_message"] = _redact_message(message, secrets, private_values)
    elif isinstance(exc, InterpretationError):
        result["category"] = {"parse": "parse", "schema_validation": "validation",
                              "response_validation": "response", "grounding": "validation"}.get(exc.stage, "internal")
        if isinstance(exc, GroundingError):
            result.update(field=exc.field, reason_code=exc.reason_code)
    elif stage in {"grounding", "proposal_validation"}:
        result.update(category="validation", error_code="INPUT_GROUNDING_REJECTED" if stage == "grounding"
                      else "INPUT_PROPOSAL_REJECTED")
    elif any(isinstance(e, ValidationError) for e in chain):
        result.update(category="validation", error_code="MODEL_SCHEMA_INVALID")
    elif stage == "parse":
        result.update(category="parse", error_code="MODEL_JSON_INVALID")
    elif stage == "schema_validation":
        result.update(category="validation", error_code="MODEL_SCHEMA_INVALID")
    elif isinstance(exc, (ValueError, TypeError)) and stage == "response_validation":
        result.update(category="response", error_code="MODEL_RESPONSE_REJECTED")
    else:
        import httpx
        if any(isinstance(e, (httpx.TransportError, TimeoutError, ConnectionError)) for e in chain):
            result.update(category="provider", error_code="PROVIDER_TRANSPORT_ERROR", retryable=True)
    validation = next((e for e in chain if isinstance(e, ValidationError)), None)
    if validation is not None:
        result["validation_error_types"] = sorted({e["type"] for e in validation.errors(
            include_url=False, include_context=False, include_input=False)})[:10]
    raw = getattr(exc, "raw", None)
    if raw is not None:
        # No raw content, thoughts, tool arguments, model IDs or user data.
        metadata = getattr(raw, "response_metadata", {})
        finish = metadata.get("finish_reason") if isinstance(metadata, dict) else None
        if isinstance(finish, str) and finish in {"STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"}:
            result["finish_reason"] = finish
        result["response_received"] = True
    return result
