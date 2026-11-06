import time
import random
from typing import Any, Optional, Tuple, List
from google import genai
from google.genai import types
from google.genai.errors import APIError
from src.config import get_gemini_api_key, CHAT_MODEL, TEMPERATURE, MAX_DISTANCE
from src.prompts.prompt_templates import (
    SYSTEM_PROMPT,
    PLAIN_PROMPT,
    FALLBACK_RESPONSE,
    build_rag_prompt,
)
from src.retrieval.retriever import retrieve


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


def generate_with_retry(
    contents: Any,
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = None,
    model: str = CHAT_MODEL,
    client: Optional[genai.Client] = None,
) -> str:
    """Generate content with model, including exponential backoff, model failover, and rate-limit handling."""
    cli = _get_client(client)
    temp = temperature if temperature is not None else TEMPERATURE
    config = (
        types.GenerateContentConfig(
            temperature=temp,
            system_instruction=system_instruction,
        )
        if system_instruction
        else types.GenerateContentConfig(temperature=temp)
    )

    current_model = model
    fallback_models = [m for m in ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite"] if m != model]

    attempt = 0
    while True:
        try:
            response = cli.models.generate_content(
                model=current_model,
                contents=contents,
                config=config,
            )
            return response.text
        except APIError as e:
            code = getattr(e, "code", None)
            if code is not None and 400 <= code < 500 and code != 429:
                raise
            if code == 429:
                msg = str(getattr(e, "message", e))
                raw_wait = 2 ** (attempt + 1) + random.uniform(0, 1)
                if "per day" in msg.lower():
                    if fallback_models:
                        next_model = fallback_models.pop(0)
                        print(f"  [QUOTA FAILOVER] Daily quota reached for '{current_model}'. Switching to '{next_model}'...", flush=True)
                        current_model = next_model
                        attempt = 0
                        continue
                    raise RuntimeError("Daily API quota reached. Try again later.") from e
                if raw_wait > 60:
                    raise RuntimeError("API rate limit exceeded. Try again later.") from e
            if code == 503 and attempt >= 1 and fallback_models:
                next_model = fallback_models.pop(0)
                print(f"  [FAILOVER] Model '{current_model}' unavailable (503). Switching to '{next_model}'...", flush=True)
                current_model = next_model
                attempt = 0
                continue
            attempt += 1
            wait = min(2 ** attempt + random.uniform(0, 1), 60)
            err_code = getattr(e, "code", "APIError")
            err_msg = getattr(e, "message", str(e))
            print(f"  [RETRY {attempt}] Generation error ({err_code}): {err_msg}. Waiting {wait:.1f}s...", flush=True)
            time.sleep(wait)
            if attempt >= 6:
                raise RuntimeError(f"Generation failed after {attempt} retries: {e}") from e
        except Exception as e:
            attempt += 1
            wait = min(2 ** attempt + random.uniform(0, 1), 60)
            print(f"  [RETRY {attempt}] Generation error: {e}. Waiting {wait:.1f}s...", flush=True)
            time.sleep(wait)
            if attempt >= 6:
                raise RuntimeError(f"Generation failed after {attempt} retries: {e}") from e


def rag_answer(
    question: str,
    k: Optional[int] = None,
    max_distance: Optional[float] = None,
    model: str = CHAT_MODEL,
    client: Optional[genai.Client] = None,
) -> Tuple[str, List[str]]:
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
        return FALLBACK_RESPONSE, []

    sources = []
    for chunk in valid_chunks:
        label = f"{chunk['source']}, p.{chunk['page']}"
        if label not in sources:
            sources.append(label)

    prompt = build_rag_prompt(question, valid_chunks)
    answer = generate_with_retry(
        contents=prompt,
        system_instruction=SYSTEM_PROMPT,
        temperature=TEMPERATURE,
        model=model,
        client=client,
    )

    # If the model itself replies with the fallback line, return empty sources
    if FALLBACK_RESPONSE.lower() in answer.lower():
        return FALLBACK_RESPONSE, []

    return answer, sources


def plain_answer(
    question: str,
    model: str = CHAT_MODEL,
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
