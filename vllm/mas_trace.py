# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in MASBench backend trace JSONL sink.

This module intentionally stays dependency-light and best-effort. It is enabled
only when VLLM_MAS_TRACE_PATH is set, so normal vLLM serving paths do not pay
file I/O cost unless MASBench tracing is explicitly requested.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()
_TRACE_PATH = os.environ.get("VLLM_MAS_TRACE_PATH", "")
_ENABLED = bool(_TRACE_PATH)


def enabled() -> bool:
    return _ENABLED


def emit(event_type: str, **payload: Any) -> None:
    if not _ENABLED:
        return
    record = {
        "schema_version": "masbench_vllm_backend.v1",
        "event_type": event_type,
        "timestamp_unix": time.time(),
        **_sanitize(payload),
    }
    try:
        path = Path(_TRACE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with _LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Tracing must never affect serving correctness.
        return


def summarize_scheduler_output(scheduler_output: Any) -> dict[str, Any]:
    req_ids = list(getattr(scheduler_output, "num_scheduled_tokens", {}).keys())
    scheduled_tokens = dict(getattr(scheduler_output, "num_scheduled_tokens", {}))
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    context_req_ids: list[str] = []
    decode_req_ids: list[str] = []
    for req_id in req_ids:
        is_context = bool(cached is not None and cached.is_context_phase(req_id))
        if is_context or any(req.req_id == req_id for req in getattr(scheduler_output, "scheduled_new_reqs", [])):
            context_req_ids.append(req_id)
        else:
            decode_req_ids.append(req_id)
    return {
        "request_ids": req_ids,
        "batch_size": len(req_ids),
        "scheduled_tokens_by_request": scheduled_tokens,
        "total_num_scheduled_tokens": getattr(scheduler_output, "total_num_scheduled_tokens", 0),
        "context_request_ids": context_req_ids,
        "decode_request_ids": decode_req_ids,
        "num_context_requests": len(context_req_ids),
        "num_decode_requests": len(decode_req_ids),
        "num_common_prefix_blocks": getattr(scheduler_output, "num_common_prefix_blocks", None),
        "preempted_request_ids": sorted(getattr(scheduler_output, "preempted_req_ids", set()) or []),
        "finished_request_ids": sorted(getattr(scheduler_output, "finished_req_ids", set()) or []),
    }


def summarize_kv_event(event: Any) -> dict[str, Any]:
    name = type(event).__name__
    block_hashes = getattr(event, "block_hashes", None)
    block_count = len(block_hashes) if block_hashes is not None else None
    payload = {
        "kv_event_type": name,
        "medium": getattr(event, "medium", None),
        "group_idx": getattr(event, "group_idx", None),
        "block_count": block_count,
    }
    if name == "BlockStored":
        token_ids = getattr(event, "token_ids", None)
        block_size = getattr(event, "block_size", None)
        payload.update(
            {
                "operation": "store",
                "block_size": block_size,
                "token_count": len(token_ids) if token_ids is not None else None,
                "parent_block_hash": _short_hash(getattr(event, "parent_block_hash", None)),
            }
        )
    elif name == "BlockRemoved":
        payload["operation"] = "remove"
    elif name == "AllBlocksCleared":
        payload["operation"] = "clear"
    else:
        payload["operation"] = "unknown"
    if block_hashes:
        payload["block_hashes_sample"] = [_short_hash(item) for item in block_hashes[:8]]
    return payload


def _short_hash(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[:24]


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
