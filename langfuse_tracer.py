"""
Langfuse v3 Tracing Wrapper
============================
A production-ready decorator for tracing any function with Langfuse SDK v3 (OpenTelemetry-based).

Supports: AzureChatOpenAI, Autogen, LangChain chains, LangGraph, raw LLM calls, etc.

Install:
    pip install "langfuse>=3.0.0" python-dotenv

.env file:
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
    LANGFUSE_ENABLED=true                       # set to false to disable tracing
"""

import os
import logging
import functools
import inspect
import traceback
from typing import Any, Callable, Optional

from dotenv import load_dotenv

# Load .env before importing langfuse so env vars are available at init time
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Langfuse initialisation
# ---------------------------------------------------------------------------
_langfuse_client = None
_langfuse_available = False
_init_attempted = False


def _is_tracing_enabled() -> bool:
    return os.getenv("LANGFUSE_ENABLED", "true").strip().lower() not in ("false", "0", "no")


def _get_client():
    """Return the global Langfuse client, initialising it on first call."""
    global _langfuse_client, _langfuse_available, _init_attempted

    if _init_attempted:
        return _langfuse_client

    _init_attempted = True

    if not _is_tracing_enabled():
        logger.info("Langfuse tracing is disabled via LANGFUSE_ENABLED env var.")
        return None

    required = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        logger.warning(
            "Langfuse tracing disabled – missing env vars: %s. "
            "Add them to your .env file.",
            ", ".join(missing),
        )
        return None

    try:
        from langfuse import get_client  # type: ignore

        _langfuse_client = get_client()
        _langfuse_available = True
        logger.info(
            "Langfuse client initialised (host=%s).",
            os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except ImportError:
        logger.error(
            "langfuse package not installed. Run: pip install 'langfuse>=3.0.0'"
        )
    except Exception as exc:
        logger.error("Failed to initialise Langfuse client: %s", exc)

    return _langfuse_client


# ---------------------------------------------------------------------------
# Core decorator
# ---------------------------------------------------------------------------


def trace(
    name: Optional[str] = None,
    *,
    as_type: str = "span",          # "span" | "generation"
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    capture_errors: bool = True,
) -> Callable:
    """
    Decorator that wraps any sync or async function with a Langfuse trace span.

    Parameters
    ----------
    name            : Display name in Langfuse. Defaults to the function name.
    as_type         : "span" (default) or "generation" for direct LLM call functions.
    user_id         : Optional user identifier attached to the trace.
    session_id      : Optional session identifier attached to the trace.
    tags            : List of string tags for the trace.
    metadata        : Extra dict metadata attached to the span.
    capture_input   : Record function arguments as span input  (default True).
    capture_output  : Record return value as span output       (default True).
    capture_errors  : Record exceptions in span before re-raise (default True).

    Usage
    -----
    # Basic span
    @trace()
    def my_pipeline(data: str) -> str: ...

    # LLM generation with user context
    @trace(name="azure-chat", as_type="generation", user_id="u_123", tags=["prod"])
    def call_llm(prompt: str) -> str: ...

    # Async (Autogen, async LangGraph nodes, etc.)
    @trace(name="agent-run", session_id="sess_abc")
    async def run_agent(task: str) -> str: ...
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        # ------------------------------------------------------------------ #
        # Build input payload from call args/kwargs                           #
        # ------------------------------------------------------------------ #
        def _build_input(args, kwargs):
            if not capture_input:
                return None
            try:
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return dict(bound.arguments)
            except Exception:
                return {"args": repr(args), "kwargs": repr(kwargs)}

        # ------------------------------------------------------------------ #
        # Shared: attach trace-level metadata                                 #
        # ------------------------------------------------------------------ #
        def _attach_trace_meta(client):
            update_kwargs: dict[str, Any] = {}
            if user_id:
                update_kwargs["user_id"] = user_id
            if session_id:
                update_kwargs["session_id"] = session_id
            if tags:
                update_kwargs["tags"] = tags
            if update_kwargs:
                try:
                    client.update_current_trace(**update_kwargs)
                except Exception as exc:
                    logger.debug("Could not update trace metadata: %s", exc)

        # ------------------------------------------------------------------ #
        # Sync wrapper                                                         #
        # ------------------------------------------------------------------ #
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            client = _get_client()
            if client is None:
                return func(*args, **kwargs)

            span_input = _build_input(args, kwargs)

            try:
                ctx_manager = client.start_as_current_observation(
                    as_type=as_type,
                    name=span_name,
                    input=span_input,
                    metadata=metadata or {},
                )
            except Exception as exc:
                logger.warning("Could not start Langfuse observation: %s", exc)
                return func(*args, **kwargs)

            with ctx_manager as observation:
                _attach_trace_meta(client)
                try:
                    result = func(*args, **kwargs)
                    if capture_output:
                        try:
                            observation.update(output=result)
                        except Exception as exc:
                            logger.debug("Could not update span output: %s", exc)
                    return result
                except Exception as exc:
                    if capture_errors:
                        try:
                            observation.update(
                                metadata={
                                    **(metadata or {}),
                                    "error": str(exc),
                                    "traceback": traceback.format_exc(),
                                },
                                level="ERROR",
                            )
                        except Exception:
                            pass
                    raise

        # ------------------------------------------------------------------ #
        # Async wrapper                                                        #
        # ------------------------------------------------------------------ #
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            client = _get_client()
            if client is None:
                return await func(*args, **kwargs)

            span_input = _build_input(args, kwargs)

            try:
                ctx_manager = client.start_as_current_observation(
                    as_type=as_type,
                    name=span_name,
                    input=span_input,
                    metadata=metadata or {},
                )
            except Exception as exc:
                logger.warning("Could not start Langfuse observation: %s", exc)
                return await func(*args, **kwargs)

            with ctx_manager as observation:
                _attach_trace_meta(client)
                try:
                    result = await func(*args, **kwargs)
                    if capture_output:
                        try:
                            observation.update(output=result)
                        except Exception as exc:
                            logger.debug("Could not update span output: %s", exc)
                    return result
                except Exception as exc:
                    if capture_errors:
                        try:
                            observation.update(
                                metadata={
                                    **(metadata or {}),
                                    "error": str(exc),
                                    "traceback": traceback.format_exc(),
                                },
                                level="ERROR",
                            )
                        except Exception:
                            pass
                    raise

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator


# ---------------------------------------------------------------------------
# Convenience alias for LLM generation functions
# ---------------------------------------------------------------------------


def trace_llm(name: Optional[str] = None, **kwargs) -> Callable:
    """
    Shorthand for @trace(as_type='generation', ...).
    Use on functions that directly call an LLM (AzureChatOpenAI, OpenAI, etc.).

    @trace_llm(name="gpt4o-call", user_id="u_1")
    def call_azure(prompt: str) -> str: ...
    """
    return trace(name=name, as_type="generation", **kwargs)


# ---------------------------------------------------------------------------
# Flush — call at app shutdown / end of short-lived scripts
# ---------------------------------------------------------------------------


def flush():
    """Flush all pending Langfuse events to the server. Call before process exit."""
    client = _get_client()
    if client is not None:
        try:
            client.flush()
            logger.info("Langfuse events flushed.")
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers for richer mid-function enrichment
# ---------------------------------------------------------------------------


def update_current_span(**kwargs):
    """
    Update the currently active span from inside a @trace'd function.
    E.g.: update_current_span(output="custom", metadata={"tokens": 42})
    """
    client = _get_client()
    if client:
        try:
            client.update_current_span(**kwargs)
        except Exception as exc:
            logger.debug("update_current_span failed: %s", exc)


def update_current_trace(**kwargs):
    """
    Update trace-level attributes from inside a @trace'd function.
    E.g.: update_current_trace(user_id="dynamic_user", session_id="s_99")
    """
    client = _get_client()
    if client:
        try:
            client.update_current_trace(**kwargs)
        except Exception as exc:
            logger.debug("update_current_trace failed: %s", exc)

