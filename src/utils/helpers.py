import time
import random
import logging
from typing import Callable, Any
from google.genai.errors import APIError


def get_logger(name: str = "rag_app") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def retry_with_backoff(
    action_name: str = "Operation",
    max_retries: int = 6,
    base_wait: float = 2.0,
    max_wait: float = 60.0,
) -> Callable:
    """Decorator to retry a function with exponential backoff and jitter."""
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            return retry_api_call(
                fn=lambda: fn(*args, **kwargs),
                action_name=action_name,
                max_retries=max_retries,
                base_wait=base_wait,
                max_wait=max_wait,
            )
        return wrapper
    return decorator


def retry_api_call(
    fn: Callable[[], Any],
    action_name: str = "Operation",
    max_retries: int = 6,
    base_wait: float = 2.0,
    max_wait: float = 60.0,
) -> Any:
    """Execute a callable with exponential backoff handling 429, 503, and transient errors."""
    attempt = 0
    while True:
        try:
            return fn()
        except APIError as e:
            code = getattr(e, "code", None)
            # Re-raise client errors that aren't 429
            if code is not None and 400 <= code < 500 and code != 429:
                raise
            if code == 429:
                msg = str(getattr(e, "message", e))
                raw_wait = (base_wait ** (attempt + 1)) + random.uniform(0, 1)
                if "per day" in msg.lower() or raw_wait > max_wait:
                    raise RuntimeError("Daily API quota reached. Try again later.") from e
            attempt += 1
            wait = min((base_wait ** attempt) + random.uniform(0, 1), max_wait)
            err_code = getattr(e, "code", "APIError")
            err_msg = getattr(e, "message", str(e))
            print(f"  [RETRY {attempt}] {action_name} error ({err_code}): {err_msg}. Waiting {wait:.1f}s...")
            time.sleep(wait)
            if attempt >= max_retries:
                raise RuntimeError(f"{action_name} failed after {attempt} retries: {e}") from e
        except Exception as e:
            attempt += 1
            wait = min((base_wait ** attempt) + random.uniform(0, 1), max_wait)
            print(f"  [RETRY {attempt}] {action_name} error: {e}. Waiting {wait:.1f}s...")
            time.sleep(wait)
            if attempt >= max_retries:
                raise RuntimeError(f"{action_name} failed after {attempt} retries: {e}") from e

