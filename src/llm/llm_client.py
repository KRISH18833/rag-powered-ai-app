import time
import random
from typing import Any, Optional, Tuple, List
from google import genai
from google.genai import types
from google.genai.errors import APIError
from src.config import load_config, get_gemini_api_key, CHAT_MODEL, TEMPERATURE, MAX_DISTANCE
from src.prompts.prompt_templates import (
    SYSTEM_PROMPT,
    PLAIN_PROMPT,
    FALLBACK_RESPONSE,
    build_rag_prompt,
)
from src.retrieval.retriever import retrieve


class RagResult(tuple):
    """Tuple subclass (answer, sources) carrying retrieved chunks and picklable by st.cache_data."""

    def __new__(cls, answer: str, sources: list, chunks: list = None):
        obj = super().__new__(cls, (answer, sources))
        obj.chunks = chunks or []
        return obj

    def __reduce__(self):
        return (RagResult, (self[0], self[1], self.chunks))


class GenerationFailure(Exception):
    """Raised when LLM generation fails after all retry attempts."""

    def __init__(self, original_exc: Exception, chunks: list = None, sources: list = None):
        super().__init__(str(original_exc))
        self.original_exc = original_exc
        self.chunks = chunks or []
        self.sources = sources or []


def classify_gemini_error(e: Exception) -> str:
    """Classify exception into a user-friendly message and log the real error to stdout."""
    exc_type = type(e).__name__
    exc_msg = str(e)
    print(f"[STREAMLIT LOGS] Error encountered - {exc_type}: {exc_msg}", flush=True)

    msg_lower = exc_msg.lower()
    # 1. Invalid or missing key
    if "api_key" in msg_lower or "api key" in msg_lower or "api_key_invalid" in msg_lower or "not set" in msg_lower:
        return "Invalid or missing Gemini API key. Please check your GEMINI_API_KEY configuration."

    # 2. Quota exhausted (429)
    if "429" in msg_lower or "resource_exhausted" in msg_lower or "quota" in msg_lower:
        return "Gemini free-tier quota reached, please retry in a minute."

    # 3. Overloaded (503)
    if "503" in msg_lower or "unavailable" in msg_lower or "overloaded" in msg_lower or "high demand" in msg_lower:
        return "The AI service is currently overloaded. Please try again in a moment."

    # 4. Timeout
    if "timeout" in msg_lower or "timed out" in msg_lower or isinstance(e, TimeoutError):
        return "Request timed out while waiting for Gemini. Please check your connection and try again."

    # 5. Generic fallback
    return "The AI service is temporarily busy. Please try again."


def format_degraded_passages(chunks: list) -> str:
    """Format top 3 retrieved chunks for graceful degradation display."""
    lines = ["AI summary is temporarily unavailable. Most relevant passages from the course notes:\n"]
    for i, c in enumerate(chunks[:3], 1):
        dist = c.get("distance", 0.0)
        snippet = c.get("text", "").replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        lines.append(f"**[{i}] {c.get('source', 'Unknown')}, p.{c.get('page', '?')}** (distance: `{dist:.4f}`)\n> {snippet}\n")
    return "\n".join(lines)


def get_llm_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or get_gemini_api_key()
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Please set it in os.environ, st.secrets, or .env.")
    return genai.Client(api_key=key)


_default_llm_client: Optional[genai.Client] = None


def _get_client(client: Optional[genai.Client] = None) -> genai.Client:
    global _default_llm_client
    if client is not None:
        return client
    if _default_llm_client is None:
        _default_llm_client = get_llm_client()
    return _default_llm_client


def get_model_rotation(primary_model: Optional[str] = None) -> List[str]:
    """Retrieve ordered model list starting with the primary model, then fallbacks from config.yaml."""
    cfg = load_config()
    primary = primary_model or cfg.get("chat_model", CHAT_MODEL)
    fallbacks = cfg.get("fallback_models", ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3-flash-preview"])
    if not isinstance(fallbacks, list):
        fallbacks = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3-flash-preview"]

    ordered = [primary]
    for m in fallbacks:
        if m not in ordered:
            ordered.append(m)
    return ordered


def generate_with_retry(
    contents: Any,
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    client: Optional[genai.Client] = None,
) -> str:
    """Generate content with model, including up to 5 attempts, exponential backoff (2, 4, 8, 16 s) + jitter, and model rotation."""
    cli = _get_client(client)
    temp = temperature if temperature is not None else TEMPERATURE
    config_obj = (
        types.GenerateContentConfig(
            temperature=temp,
            system_instruction=system_instruction,
        )
        if system_instruction
        else types.GenerateContentConfig(temperature=temp)
    )

    models = get_model_rotation(model)
    max_attempts = 5
    backoff_delays = [2, 4, 8, 16]

    last_exception = None

    for attempt in range(max_attempts):
        current_model = models[attempt % len(models)]
        try:
            response = cli.models.generate_content(
                model=current_model,
                contents=contents,
                config=config_obj,
            )
            return response.text
        except Exception as e:
            last_exception = e
            exc_type = type(e).__name__
            exc_msg = str(e)
            print(f"[RETRY {attempt + 1}/{max_attempts}] Model '{current_model}' failed with {exc_type}: {exc_msg}", flush=True)

            msg_lower = exc_msg.lower()
            # Fast fail on unrecoverable authentication errors
            if "api_key_invalid" in msg_lower or "api key not valid" in msg_lower or "api_key is not set" in msg_lower:
                raise

            if attempt < max_attempts - 1:
                delay = backoff_delays[min(attempt, len(backoff_delays) - 1)] + random.uniform(0, 1)
                next_model = models[(attempt + 1) % len(models)]
                print(f"[RETRY] Waiting {delay:.2f}s before trying model '{next_model}'...", flush=True)
                time.sleep(delay)

    raise last_exception if last_exception is not None else RuntimeError("Generation failed after all retries.")


def rag_answer(
    question: str,
    k: Optional[int] = None,
    max_distance: Optional[float] = None,
    model: Optional[str] = None,
    client: Optional[genai.Client] = None,
) -> RagResult:
    """Retrieve relevant chunks and generate a grounded RAG response.
    
    If no chunk has distance <= max_distance, skips the LLM call entirely and returns
    the exact fallback response with an empty sources list.
    """
    threshold = max_distance if max_distance is not None else MAX_DISTANCE
    chunks = retrieve(question, k=k)

    # Filter chunks based on distance threshold
    valid_chunks = [c for c in chunks if c.get("distance", 1.0) <= threshold]

    # If no chunk meets threshold, skip LLM call and return fallback immediately
    if not valid_chunks:
        return RagResult(FALLBACK_RESPONSE, [], chunks=chunks)

    sources = []
    for chunk in valid_chunks:
        label = f"{chunk['source']}, p.{chunk['page']}"
        if label not in sources:
            sources.append(label)

    prompt = build_rag_prompt(question, valid_chunks)
    try:
        answer = generate_with_retry(
            contents=prompt,
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            model=model,
            client=client,
        )
    except Exception as e:
        # Re-raise as GenerationFailure carrying the valid chunks and sources for graceful degradation
        raise GenerationFailure(original_exc=e, chunks=valid_chunks, sources=sources) from e

    # If the model itself replies with the fallback line, return empty sources
    if FALLBACK_RESPONSE.lower() in answer.lower():
        return RagResult(FALLBACK_RESPONSE, [], chunks=chunks)

    return RagResult(answer, sources, chunks=chunks)


def plain_answer(
    question: str,
    model: Optional[str] = None,
    client: Optional[genai.Client] = None,
) -> str:
    """Generate a direct answer from the LLM without context retrieval."""
    prompt = f"{PLAIN_PROMPT}\n\nQuestion: {question}"
    return generate_with_retry(
        contents=prompt,
        temperature=TEMPERATURE,
        model=model,
        client=client,
    )
