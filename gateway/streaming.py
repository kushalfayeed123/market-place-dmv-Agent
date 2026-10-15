"""SSE streaming for one agent turn.

Design rules (each one fixes a real failure):
  * The agent runs in its own task and the stream reads from a queue. Cancelling
    `queue.get()` is safe; the old `wait_for(anext(agen), 15)` cancelled the agent's
    async generator every 15 s, which ended the turn silently and produced
    'I couldn't find any results'.
  * A turn always ends with an explicit outcome: results, text, or an `error` event.
    The 'no results' fallback is sent only when nothing failed.
  * Tool-generated summary text ('Found 25 products') is sent only if the model wrote
    no text of its own, so the shopper never sees two overlapping sentences.
  * The session is saved in `finally`, so the cart survives failed and timed-out turns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

logger = logging.getLogger("gateway.stream")

_KEEP_LATEST = {"CartSummary", "OrderList", "MerchantBalanceCard", "LedgerTable", "PaymentCapturePanel"}
FALLBACK_TEXT = "I couldn't find any results. Try a broader search."


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def finalize_directives(directives: list[dict], has_text: bool) -> list[dict]:
    """Clean up the directives collected during ONE turn before sending (mirrors the frontend)."""

    def _items(d: dict) -> list:
        p = d.get("props", {}) or {}
        raw = p.get("items") or p.get("products") or []
        return raw if isinstance(raw, list) else []

    def _ids(d: dict) -> list[str]:
        return [str(i["id"]) for i in _items(d) if isinstance(i, dict) and i.get("name")]

    detail_ids = {
        str((d.get("props", {}) or {}).get("id")) for d in directives if d.get("component") == "ProductDetail"
    }
    out: list[dict] = []
    seen: set[str] = set()
    for d in directives:
        if not isinstance(d, dict) or "component" not in d:
            continue
        p = d.get("props", {}) or {}
        if d["component"] == "ProductGrid":
            if not _items(d):
                continue
            ids = _ids(d)
            if ids and all(i in detail_ids for i in ids):
                continue
        if d["component"] == "ProductCard":
            if str((p.get("product") or {}).get("id", "")) in detail_ids:
                continue
        key = d["component"] + ":" + json.dumps(p, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)

    last_detail: dict[str, int] = {}
    for i, d in enumerate(out):
        if d["component"] == "ProductDetail":
            last_detail[str((d.get("props", {}) or {}).get("id"))] = i
    out = [
        d for i, d in enumerate(out)
        if d["component"] != "ProductDetail" or last_detail.get(str((d.get("props", {}) or {}).get("id"))) == i
    ]
    for comp in _KEEP_LATEST:
        last = max((i for i, d in enumerate(out) if d["component"] == comp), default=-1)
        out = [d for i, d in enumerate(out) if d["component"] != comp or i == last]

    if not out and not has_text:
        empty = next((d for d in directives if isinstance(d, dict) and d.get("component") == "ProductGrid"), None)
        if empty:
            out.append(empty)
    return out


async def coalesce_text(events: AsyncIterator[dict], max_wait: float = 0.05, min_chars: int = 24) -> AsyncIterator[dict]:
    """Batch text deltas (~24 chars or 50 ms) so the client renders smoothly."""
    buf = ""
    last = time.monotonic()
    async for ev in events:
        if ev.get("type") == "text":
            buf += ev.get("content", "")
            if len(buf) >= min_chars or time.monotonic() - last >= max_wait:
                yield {"type": "text", "content": buf}
                buf, last = "", time.monotonic()
        else:
            if buf:
                yield {"type": "text", "content": buf}
                buf, last = "", time.monotonic()
            yield ev
    if buf:
        yield {"type": "text", "content": buf}


_DONE = object()


async def stream_turn(
    *,
    agent: Any,
    session: Any,
    message: str,
    confirmed_token: str | None,
    save_session: Callable[[], Awaitable[None]],
    turn_timeout: float = 100.0,
    heartbeat: float = 15.0,
    flush_after: float = 0.05,
    flush_chars: int = 24,
) -> AsyncIterator[str]:
    """Yield SSE frames for one turn. Always ends; never cancels the agent mid-step."""
    turn_id = session.current_turn_id + 1
    started = time.monotonic()
    yield sse("session", {"session_id": session.session_id, "turn_id": turn_id})

    # Store the confirmation token in Redis so the ConfirmationGate can verify it
    # when a mutating_financial tool is called during this turn. The token was
    # minted by the frontend on a real DOM click event and is single-use.
    if confirmed_token and hasattr(agent, "_gate"):
        from mcp_server.tools_catalog import BY_NAME
        stored = False
        for tool_meta in BY_NAME.values():
            if tool_meta.requires_confirmation:
                await agent._gate.store_token(
                    session.session_id, turn_id, tool_meta.name, confirmed_token
                )
                stored = True
        if stored:
            logger.debug(
                "stored confirmed_token session=%s turn=%s tools=%d",
                session.session_id, turn_id, len(BY_NAME),
            )

    queue: asyncio.Queue = asyncio.Queue()

    async def pump() -> None:
        try:
            async for ev in agent.run_turn(session, message, confirmed_token):
                await queue.put(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the agent already reports its own errors; this is the last resort
            logger.exception("agent turn crashed session=%s turn=%s", session.session_id, turn_id)
            await queue.put({"type": "error", "error": f"Agent error: {exc}"})
        finally:
            await queue.put(_DONE)

    task = asyncio.create_task(pump())
    directives: list[dict] = []
    tool_text: list[str] = []
    model_text_seen = False
    had_error = False
    text_buf = ""
    last_flush = time.monotonic()
    last_sent = time.monotonic()
    n_events = 0
    deadline = started + turn_timeout

    def text_frame(content: str) -> str:
        return sse("text", {"type": "text", "content": content, "turn_id": turn_id})

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            wait = flush_after if text_buf else min(heartbeat, remaining)
            try:
                event = await asyncio.wait_for(queue.get(), timeout=wait)  # cancelling queue.get() is safe
            except asyncio.TimeoutError:
                if text_buf:
                    yield text_frame(text_buf)
                    text_buf, last_flush, last_sent = "", time.monotonic(), time.monotonic()
                elif time.monotonic() - last_sent >= heartbeat - 0.5:
                    yield ": ping\n\n"
                    last_sent = time.monotonic()
                continue
            if event is _DONE:
                break
            n_events += 1
            etype = event.get("type")
            if etype == "ui_directive":
                if isinstance(event.get("directive"), dict):
                    directives.append(event["directive"])
            elif etype == "text":
                content = event.get("content", "")
                if event.get("origin") == "tool":
                    tool_text.append(content)
                else:
                    model_text_seen = True
                    text_buf += content
                    if len(text_buf) >= flush_chars or time.monotonic() - last_flush >= flush_after:
                        yield text_frame(text_buf)
                        text_buf, last_flush, last_sent = "", time.monotonic(), time.monotonic()
            else:
                if etype == "error":
                    had_error = True
                    logger.warning("turn error session=%s turn=%s: %s", session.session_id, turn_id, event.get("error"))
                yield sse(str(etype), {**event, "turn_id": turn_id})
                last_sent = time.monotonic()

        if text_buf:
            yield text_frame(text_buf)
            text_buf = ""
        if not model_text_seen and tool_text:
            yield text_frame(" ".join(tool_text))
        has_text = model_text_seen or bool(tool_text)
        finalized = finalize_directives(directives, has_text)
        if not finalized and not has_text and not had_error:
            yield text_frame(FALLBACK_TEXT)
        for d in finalized:
            yield sse("ui_directive", {"type": "ui_directive", "directive": d, "turn_id": turn_id})
    except TimeoutError:
        had_error = True
        logger.error("turn timed out after %.0fs session=%s turn=%s", turn_timeout, session.session_id, turn_id)
        if text_buf:
            yield text_frame(text_buf)
        yield sse("error", {"error": "The agent took too long to respond. Please try again.", "turn_id": turn_id})
    except Exception as exc:
        had_error = True
        logger.exception("stream failed session=%s turn=%s", session.session_id, turn_id)
        yield sse("error", {"error": f"Something went wrong: {exc}", "turn_id": turn_id})
    finally:
        task.cancel()
        try:
            await save_session()
        except Exception:
            logger.exception("could not save session %s", session.session_id)
        logger.info(
            "turn done session=%s turn=%s ms=%d events=%d directives=%d text=%s error=%s",
            session.session_id, turn_id, int((time.monotonic() - started) * 1000), n_events,
            len(directives), model_text_seen or bool(tool_text), had_error,
        )
