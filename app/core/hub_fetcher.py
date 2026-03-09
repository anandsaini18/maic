from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

from app.core.model_sizing import GATED_MODELS, resolve_size_gb

logger = logging.getLogger(__name__)


# ── Domain type ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HubModel:
    """A model discovered from HuggingFace Hub, enriched with local metadata."""

    id: str
    size_gb: float  # resolved via model_sizing (curated → regex → default)
    downloads: int
    gated: bool  # True when a HF token is required to download


# ── Offline fallback ──────────────────────────────────────────────────────────
# Three safe defaults shown when the Hub is unreachable. Sizes come from
# KNOWN_SIZES in model_sizing so this stays in sync with curated data.

DEFAULT_FALLBACK: tuple[HubModel, ...] = (
    HubModel(id="mlx-community/SmolLM2-1.7B-Instruct-4bit", size_gb=1.0, downloads=0, gated=False),
    HubModel(id="mlx-community/Phi-3.5-mini-instruct-4bit", size_gb=2.3, downloads=0, gated=False),
    HubModel(
        id="mlx-community/Mistral-7B-Instruct-v0.3-4bit", size_gb=4.0, downloads=0, gated=False
    ),
)


# ── HTTP ──────────────────────────────────────────────────────────────────────


def _build_url(org: str) -> str:
    return (
        "https://huggingface.co/api/models"
        f"?author={org}"
        "&pipeline_tag=text-generation"
        "&sort=downloads"
        "&limit=100"
    )


def _fetch_raw(org: str) -> list[dict[str, Any]]:
    """Fetch raw JSON from the HF API. Returns [] on any network error.

    Deliberately separated from parsing so transformation logic can be
    tested independently with fixture data, without mocking HTTP.
    """
    req = urllib.request.Request(_build_url(org), headers={"User-Agent": "maic/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning("HuggingFace Hub unreachable: %s", exc)
        return []


# ── Parsing ───────────────────────────────────────────────────────────────────


def _parse_hub_response(data: list[dict[str, Any]], org: str) -> list[HubModel]:
    """Transform raw HF API JSON into typed HubModel list.

    The HF "gated" field is the primary signal for token requirements; the
    static GATED_MODELS frozenset acts as a safety-net for inconsistencies.
    Size is resolved via model_sizing.resolve_size_gb, which always returns
    a value (curated → regex estimate → 7.0 default).
    """
    result = []
    prefix = f"{org}/"
    for item in data:
        mid: str = item.get("id", "")
        if not mid.startswith(prefix):
            continue
        gated = bool(item.get("gated", False)) or mid in GATED_MODELS
        result.append(
            HubModel(
                id=mid,
                size_gb=resolve_size_gb(mid),
                downloads=item.get("downloads", 0),
                gated=gated,
            )
        )
    return result


# ── TTL cache ─────────────────────────────────────────────────────────────────


class HubModelCache:
    """Async TTL cache for the HuggingFace Hub model list.

    Double-checked locking with asyncio.Lock prevents concurrent polls
    (e.g. multiple browser tabs at 3 s intervals) from firing simultaneous
    HF API requests when the cache expires.

    Two TTLs keep the UX responsive after transient failures:
      success_ttl — how long to hold a good result (default 5 min).
      error_ttl   — how quickly to retry after a failed fetch (default 30 s),
                    so a brief network hiccup is recovered from quickly rather
                    than surfacing stale data for another 5 minutes.
    """

    def __init__(
        self,
        org: str = "mlx-community",
        success_ttl: float = 300.0,
        error_ttl: float = 30.0,
    ) -> None:
        self._org = org
        self._success_ttl = success_ttl
        self._error_ttl = error_ttl
        self._models: list[HubModel] | None = None
        self._ts: float = 0.0
        self._last_ok: bool = True
        self._lock = asyncio.Lock()

    async def get(self) -> list[HubModel]:
        """Return cached models, refreshing if the TTL has expired."""
        now = time.monotonic()
        ttl = self._success_ttl if self._last_ok else self._error_ttl
        if self._models is not None and (now - self._ts) < ttl:
            return self._models

        async with self._lock:
            # Double-check: another coroutine may have refreshed while we waited
            now = time.monotonic()
            ttl = self._success_ttl if self._last_ok else self._error_ttl
            if self._models is not None and (now - self._ts) < ttl:
                return self._models

            raw = await asyncio.to_thread(_fetch_raw, self._org)
            self._ts = time.monotonic()

            if raw:
                self._models = _parse_hub_response(raw, self._org)
                self._last_ok = True
                logger.info("Loaded %d models from HuggingFace Hub.", len(self._models))
            else:
                self._last_ok = False
                if self._models is None:
                    # First-ever fetch failed — seed with offline defaults
                    self._models = list(DEFAULT_FALLBACK)
                    logger.warning(
                        "Hub unreachable on startup; showing %d offline defaults.",
                        len(DEFAULT_FALLBACK),
                    )
                # else: keep stale cache alive; retry after error_ttl

        return self._models  # type: ignore[return-value]  # always set above


# ── Module-level singleton ────────────────────────────────────────────────────
# The org is read from settings at first use so the module can be imported
# before the event loop starts without triggering a settings load.

_hub_cache: HubModelCache | None = None


def _get_cache() -> HubModelCache:
    global _hub_cache
    if _hub_cache is None:
        from app.core.config import settings

        _hub_cache = HubModelCache(org=settings.hub_org)
    return _hub_cache


async def fetch_hub_models() -> list[HubModel]:
    """Return text-generation models from the configured HF org.

    Delegates to the module-level HubModelCache singleton. Inject a custom
    HubModelCache instance in tests to control TTL and avoid network calls.
    """
    return await _get_cache().get()
