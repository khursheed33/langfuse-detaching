"""
langfuse_tracer.py
------------------
Langfuse tracing decorators for AzureChatOpenAI + Autogen applications.

Automatically detects and supports both Langfuse SDK v2 and v3:
  - v2: from langfuse.decorators import langfuse_context, observe
  - v3: from langfuse import observe, get_client  (langfuse_context removed)

Exports:
  @trace_llm_call        — wraps any AzureChatOpenAI function call
  @trace_autogen_agent   — wraps an Autogen initiate_chat session
  LangfuseTracer         — low-level class for manual instrumentation
"""

from __future__ import annotations

import functools
import inspect
import os
import time
import traceback
from typing import Any, Callable, Optional, TypeVar, cast

# --------------------------------------------------------------------------- #
#  SDK version detection — v3 moved observe out of langfuse.decorators
# --------------------------------------------------------------------------- #

_LANGFUSE_V3 = False

try:
    # v3 import path
    from langfuse import observe, get_client as _get_client_fn  # type: ignore
    _LANGFUSE_V3 = True
except ImportError:
    pass

if not _LANGFUSE_V3:
    try:
        # v2 import path
        from langfuse.decorators import observe, langfuse_context  # type: ignore
        from langfuse import Langfuse as _LangfuseClass            # type: ignore
    except ImportError as e:
        raise ImportError(
            "langfuse package not found. Install it with: pip install langfuse"
        ) from e
        

F = TypeVar("F", bound=Callable[..., Any])

from dotenv import load_dotenv
load_dotenv('.env')
# --------------------------------------------------------------------------- #
#  Unified helper: update current trace and observation
#  Abstracts the API difference between v2 and v3
# --------------------------------------------------------------------------- #

def _update_trace(
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Update the current trace attributes (v2/v3 compatible)."""
    kwargs: dict[str, Any] = {}
    if name:       kwargs["name"] = name
    if user_id:    kwargs["user_id"] = user_id
    if session_id: kwargs["session_id"] = session_id
    if tags:       kwargs["tags"] = tags
    if metadata:   kwargs["metadata"] = metadata
    if not kwargs:
        return

    try:
        if _LANGFUSE_V3:
            lf = _get_client_fn()
            lf.update_current_trace(**kwargs)
        else:
            langfuse_context.update_current_trace(**kwargs)  # type: ignore[name-defined]
    except Exception:
        pass  # never crash the caller


def _update_observation(
    name: Optional[str] = None,
    model: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    usage: Optional[dict[str, int]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Update the current observation (v2/v3 compatible)."""
    kwargs: dict[str, Any] = {}
    if name:     kwargs["name"] = name
    if model:    kwargs["model"] = model
    if input:    kwargs["input"] = input
    if output:   kwargs["output"] = output
    if usage:    kwargs["usage_details"] = usage   # v3 uses usage_details
    if metadata: kwargs["metadata"] = metadata

    if not kwargs:
        return

    try:
        if _LANGFUSE_V3:
            lf = _get_client_fn()
            lf.update_current_observation(**kwargs)
        else:
            # v2 uses `usage` not `usage_details`
            if "usage_details" in kwargs:
                kwargs["usage"] = kwargs.pop("usage_details")
            langfuse_context.update_current_observation(**kwargs)  # type: ignore[name-defined]
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  Low-level client factory (used by LangfuseTracer)
# --------------------------------------------------------------------------- #

def _get_low_level_client(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
) -> Any:
    """
    Returns a low-level Langfuse client for manual instrumentation.

    Reads from env vars by default:
        LANGFUSE_PUBLIC_KEY
        LANGFUSE_SECRET_KEY
        LANGFUSE_HOST   (optional, default: https://cloud.langfuse.com)
    """
    if _LANGFUSE_V3:
        return _get_client_fn()  # v3: singleton, uses env vars automatically
    else:
        return _LangfuseClass(  # type: ignore[name-defined]
            public_key=public_key or os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=secret_key or os.environ["LANGFUSE_SECRET_KEY"],
            host=host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )


# --------------------------------------------------------------------------- #
#  Response / message helpers
# --------------------------------------------------------------------------- #

def _extract_response(response: Any) -> tuple[str, dict[str, int]]:
    """
    Extract (completion_text, usage_dict) from an AzureChatOpenAI / OpenAI response.

    Handles:
      - LangChain AIMessage      → .content, .usage_metadata
      - Raw OpenAI ChatCompletion → .choices[0].message.content, .usage
      - Dict / plain string fallback
    """
    completion = ""
    usage: dict[str, int] = {}

    try:
        if hasattr(response, "content"):
            completion = str(response.content)
        elif hasattr(response, "choices"):
            completion = response.choices[0].message.content or ""
        elif isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                completion = choices[0].get("message", {}).get("content", "")
        else:
            completion = str(response)

        if hasattr(response, "usage_metadata"):           # LangChain
            um = response.usage_metadata
            usage = {
                "input": getattr(um, "input_tokens", 0),
                "output": getattr(um, "output_tokens", 0),
                "total": getattr(um, "total_tokens", 0),
            }
        elif hasattr(response, "usage") and response.usage:  # OpenAI
            u = response.usage
            usage = {
                "input": getattr(u, "prompt_tokens", 0),
                "output": getattr(u, "completion_tokens", 0),
                "total": getattr(u, "total_tokens", 0),
            }
    except Exception:
        pass

    return completion, usage


def _normalise_messages(messages: Any) -> list[dict[str, str]]:
    """
    Convert various message formats to [{"role": ..., "content": ...}].
    Accepts: str, list[dict], list[LangChain BaseMessage].
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    result: list[dict[str, str]] = []
    for msg in (messages if isinstance(messages, list) else [messages]):
        if isinstance(msg, dict):
            result.append({"role": msg.get("role", "user"), "content": str(msg.get("content", ""))})
        elif hasattr(msg, "type") and hasattr(msg, "content"):
            role_map = {"human": "user", "ai": "assistant", "system": "system"}
            result.append({"role": role_map.get(msg.type, msg.type), "content": str(msg.content)})
        else:
            result.append({"role": "user", "content": str(msg)})
    return result


def _detect_input(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Inspect bound arguments and return the first arg named
    'messages', 'prompt', or 'inputs' as a normalised message list.
    Falls back to the first positional argument.
    """
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        all_args = dict(bound.arguments)
        raw = (
            all_args.get("messages")
            or all_args.get("prompt")
            or all_args.get("inputs")
            or (list(all_args.values())[0] if all_args else "")
        )
        return _normalise_messages(raw)
    except Exception:
        return []


# --------------------------------------------------------------------------- #
#  @trace_llm_call
# --------------------------------------------------------------------------- #

def trace_llm_call(
    name: Optional[str] = None,
    model: str = "azure-openai",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[F], F]:
    """
    Decorator that wraps any function making an AzureChatOpenAI call and
    records prompt / completion / token usage / latency to Langfuse.

    The decorated function becomes a trace; the LLM interaction is recorded
    as a generation observation (as_type="generation") inside that trace.

    Parameters
    ----------
    name           : display name in Langfuse (defaults to function name)
    model          : model identifier shown in Langfuse UI
    user_id        : optional user identifier for filtering
    session_id     : optional session identifier for grouping traces
    tags           : list of string tags
    metadata       : extra key/value metadata dict
    capture_input  : record the prompt messages (default True)
    capture_output : record the completion text and usage (default True)

    Usage
    -----
    @trace_llm_call(name="qa_chain", model="gpt-4o", tags=["rag", "prod"])
    def answer_question(messages: list[dict]) -> str:
        response = llm.invoke(messages)
        return response.content
    """

    def decorator(func: F) -> F:
        fn_name = name or func.__name__

        @observe(name=fn_name, as_type="generation")  # type: ignore[call-arg]
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _update_trace(
                name=fn_name,
                user_id=user_id,
                session_id=session_id,
                tags=tags,
                metadata=metadata,
            )

            prompt: list[dict[str, str]] = []
            if capture_input:
                prompt = _detect_input(func, args, kwargs)

            _update_observation(
                name=fn_name,
                model=model,
                input=prompt or None,
                metadata=metadata,
            )

            t0 = time.perf_counter()
            result: Any = None
            error: Optional[str] = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                error = traceback.format_exc()
                raise
            finally:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                out_meta = {**(metadata or {}), "latency_ms": latency_ms}

                if capture_output and result is not None:
                    completion, usage = _extract_response(result)
                    _update_observation(
                        output=completion,
                        usage=usage or None,
                        metadata=out_meta,
                    )
                elif error:
                    _update_observation(
                        output={"error": error},
                        metadata={**out_meta, "error": True},
                    )

        return cast(F, wrapper)

    return decorator


# --------------------------------------------------------------------------- #
#  @trace_autogen_agent
# --------------------------------------------------------------------------- #

def trace_autogen_agent(
    agent_name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator for Autogen agent sessions.

    Wraps any function that calls user_proxy.initiate_chat(...) and records:
      - Initial task message as input
      - Full chat_history in output
      - Final summary and turn count
      - Latency in metadata

    Parameters
    ----------
    agent_name : trace name (defaults to function name)
    user_id    : optional user identifier
    session_id : optional session identifier
    tags       : list of string tags
    metadata   : extra key/value metadata

    Usage
    -----
    @trace_autogen_agent(agent_name="code_writer", tags=["autogen"])
    def run_task(task: str):
        return user_proxy.initiate_chat(assistant, message=task, max_turns=6)
    """

    def decorator(func: F) -> F:
        fn_name = agent_name or func.__name__
        effective_tags = (tags or []) + ["autogen"]

        @observe(name=fn_name)  # type: ignore[call-arg]
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _update_trace(
                name=fn_name,
                user_id=user_id,
                session_id=session_id,
                tags=effective_tags,
                metadata=metadata,
            )

            initial_message: str = str(
                kwargs.get("message", "")
                or (args[2] if len(args) > 2 else "")
            )

            _update_observation(
                name=fn_name,
                input={"initial_message": initial_message},
                metadata=metadata,
            )

            t0 = time.perf_counter()
            result: Any = None
            error: Optional[str] = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                error = traceback.format_exc()
                raise
            finally:
                latency_ms = int((time.perf_counter() - t0) * 1000)

                chat_history: list[dict[str, Any]] = []
                summary = error or ""

                if result is not None:
                    if hasattr(result, "chat_history"):
                        chat_history = result.chat_history or []
                    if hasattr(result, "summary") and result.summary:
                        summary = str(result.summary)
                    elif chat_history:
                        summary = str(chat_history[-1].get("content", ""))

                _update_observation(
                    output={
                        "summary": summary,
                        "total_turns": len(chat_history),
                        "chat_history": chat_history,
                    },
                    metadata={
                        **(metadata or {}),
                        "latency_ms": latency_ms,
                        "error": error,
                    },
                )

        return cast(F, wrapper)

    return decorator


# --------------------------------------------------------------------------- #
#  LangfuseTracer — low-level manual instrumentation
# --------------------------------------------------------------------------- #

class LangfuseTracer:
    """
    Low-level manual tracer that works with both Langfuse SDK v2 and v3.

    Use this when you cannot use the decorators (e.g. wrapping existing code).

    Example
    -------
    tracer = LangfuseTracer("rag_pipeline", session_id="s-001", tags=["rag"])

    span = tracer.start_span("retrieve_docs", input={"query": "..."})
    docs = retriever.invoke(query)
    tracer.end_span(span, output={"doc_count": len(docs)})

    gen  = tracer.start_generation("llm_call", model="gpt-4o", input=[...])
    resp = llm.invoke(messages)
    tracer.end_generation(gen, response=resp)   # auto-extracts text + usage

    tracer.flush()
    """

    def __init__(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._client = _get_low_level_client()
        self._trace = self._client.trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            tags=tags or [],
            metadata=metadata or {},
        )

    def start_generation(
        self,
        name: str,
        model: str,
        input: Optional[list[dict[str, str]]] = None,
        model_parameters: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Open a generation observation. Returns its handle."""
        return self._trace.generation(
            name=name,
            model=model,
            input=input or [],
            model_parameters=model_parameters or {},
            metadata=metadata or {},
        )

    def end_generation(
        self,
        generation: Any,
        response: Optional[Any] = None,
        output: Optional[str] = None,
        usage: Optional[dict[str, int]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Close a generation. Pass either:
          - response: raw AzureChatOpenAI / OpenAI object (auto-extracts)
          - output + usage: pre-extracted values
        """
        if response is not None and output is None:
            output, usage = _extract_response(response)
        generation.end(output=output or "", usage=usage, metadata=metadata or {})

    def start_span(
        self,
        name: str,
        input: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Open a span observation. Returns its handle."""
        return self._trace.span(name=name, input=input, metadata=metadata or {})

    def end_span(
        self,
        span: Any,
        output: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Close a span."""
        span.end(output=output, metadata=metadata or {})

    def flush(self) -> None:
        """Flush all pending events to Langfuse. Always call at end of request."""
        self._client.flush()