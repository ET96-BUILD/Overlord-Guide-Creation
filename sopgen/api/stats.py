"""Persistent counters for successfully generated guides.

Stores a JSON blob of the form::

    {"count": N, "by_user": {"<bare-email>": <count>, ...}}

at ``settings.stats_path``. Read-modify-write under a ``threading.Lock``
so concurrent in-process async tasks can't lose increments.

Cross-instance concurrency (e.g. multiple Cloud Run replicas writing to
a shared GCS-mounted volume) is NOT safe under this design — at scale,
swap for Firestore or a similar backend with atomic counter primitives.

Backwards compatibility: a legacy file containing only ``{"count": N}``
is read transparently — ``read_count()`` returns N, ``read_leaderboard()``
returns ``[]``, and the next ``increment()`` writes back the new shape
without losing the prior count.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

from sopgen.core.config import Settings

logger = logging.getLogger(__name__)


class GuidesStats:
    """Thread-safe persistent counter for successful pipeline runs."""

    def __init__(self, settings: Settings) -> None:
        self._path = Path(settings.stats_path)
        self._lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────

    def read_count(self) -> int:
        """Return the current grand-total count. Missing/corrupt → 0."""
        with self._lock:
            count, _ = self._read_locked()
            return count

    def read_leaderboard(self, limit: int = 5) -> list[dict]:
        """Return the top *limit* users by count, sorted descending.

        Each entry is ``{"email": "<bare email>", "count": N}``. Ties
        break by email ascending so the order is deterministic.
        """
        if limit < 0:
            limit = 0
        with self._lock:
            _, by_user = self._read_locked()
        ranked = sorted(
            by_user.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return [{"email": e, "count": n} for e, n in ranked[:limit]]

    def increment(self, email: Optional[str] = None) -> int:
        """Atomically (within this process) bump the grand total by 1
        and, if *email* is provided, the per-user counter by 1. Returns
        the new grand-total value."""
        with self._lock:
            count, by_user = self._read_locked()
            count += 1
            if email:
                by_user[email] = by_user.get(email, 0) + 1
            self._write_locked(count, by_user)
            return count

    # ── Internals — must be called with self._lock held ─────────────

    def _read_locked(self) -> tuple[int, dict[str, int]]:
        if not self._path.exists():
            return 0, {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            count = int(data.get("count", 0))
            if count < 0:
                count = 0
            raw_by_user = data.get("by_user") or {}
            # Defensive: filter out non-string keys / non-int values.
            by_user: dict[str, int] = {}
            if isinstance(raw_by_user, dict):
                for k, v in raw_by_user.items():
                    if isinstance(k, str) and isinstance(v, int) and v >= 0:
                        by_user[k] = v
            return count, by_user
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            logger.warning(
                "Stats file at %s is unreadable; treating as 0", self._path
            )
            return 0, {}

    def _write_locked(self, count: int, by_user: dict[str, int]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then atomically replace, so a crash mid-write
        # can't leave a torn JSON blob that then gets parsed as 0.
        fd, tmp = tempfile.mkstemp(
            prefix=".stats_", suffix=".json.tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"count": count, "by_user": by_user}, fh)
            os.replace(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
