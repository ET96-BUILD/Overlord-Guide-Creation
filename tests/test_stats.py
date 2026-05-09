"""Tests for the persistent GuidesStats counter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sopgen.api.stats import GuidesStats
from sopgen.core.config import Settings


def _settings_for(stats_path: Path) -> Settings:
    return Settings(gemini_api_key="test-key", stats_path=stats_path)


# ── Persistence ─────────────────────────────────────────────────────────


class TestPersistence:
    def test_increment_persists_across_instances(self, tmp_path):
        path = tmp_path / "stats.json"

        s1 = GuidesStats(_settings_for(path))
        assert s1.increment() == 1
        assert s1.increment() == 2
        assert s1.increment() == 3

        # A fresh instance pointing at the same path resumes from 3.
        s2 = GuidesStats(_settings_for(path))
        assert s2.read_count() == 3
        assert s2.increment() == 4

    def test_increment_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "stats.json"
        s = GuidesStats(_settings_for(nested))
        assert s.increment() == 1
        assert nested.exists()


# ── Initialization ──────────────────────────────────────────────────────


class TestInitialization:
    def test_missing_file_reads_zero(self, tmp_path):
        path = tmp_path / "no-such.json"
        s = GuidesStats(_settings_for(path))
        assert s.read_count() == 0
        # Reading shouldn't create the file.
        assert not path.exists()

    def test_corrupt_file_treated_as_zero(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text("not json", encoding="utf-8")

        s = GuidesStats(_settings_for(path))
        assert s.read_count() == 0
        # Increment recovers cleanly by writing a fresh value in the
        # current {count, by_user} shape.
        assert s.increment() == 1
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "count": 1,
            "by_user": {},
        }

    def test_negative_count_in_file_clamped_to_zero(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text(json.dumps({"count": -5}), encoding="utf-8")
        s = GuidesStats(_settings_for(path))
        assert s.read_count() == 0


# ── Per-user counters + leaderboard ─────────────────────────────────────


class TestPerUserCounters:
    def test_increment_with_email_bumps_total_and_user(self, tmp_path):
        path = tmp_path / "stats.json"
        s = GuidesStats(_settings_for(path))

        assert s.increment(email="alice@ifixit.com") == 1
        assert s.increment(email="bob@ifixit.com") == 2
        assert s.increment(email="alice@ifixit.com") == 3

        assert s.read_count() == 3
        board = s.read_leaderboard()
        assert board == [
            {"email": "alice@ifixit.com", "count": 2},
            {"email": "bob@ifixit.com", "count": 1},
        ]

    def test_anonymous_increment_bumps_total_only(self, tmp_path):
        path = tmp_path / "stats.json"
        s = GuidesStats(_settings_for(path))

        assert s.increment() == 1
        assert s.increment(email=None) == 2
        # Empty string is also treated as anonymous
        assert s.increment(email="") == 3

        assert s.read_count() == 3
        assert s.read_leaderboard() == []

    def test_anonymous_and_named_increments_coexist(self, tmp_path):
        path = tmp_path / "stats.json"
        s = GuidesStats(_settings_for(path))

        s.increment(email="alice@ifixit.com")
        s.increment()                          # anonymous
        s.increment(email="alice@ifixit.com")
        s.increment()                          # anonymous

        assert s.read_count() == 4
        assert s.read_leaderboard() == [
            {"email": "alice@ifixit.com", "count": 2},
        ]


class TestLeaderboardSorting:
    def test_sorts_descending_and_respects_limit(self, tmp_path):
        path = tmp_path / "stats.json"
        s = GuidesStats(_settings_for(path))

        # Build a deterministic distribution.
        for _ in range(5): s.increment(email="a@x.com")
        for _ in range(3): s.increment(email="b@x.com")
        for _ in range(7): s.increment(email="c@x.com")
        s.increment(email="d@x.com")
        s.increment(email="e@x.com")
        s.increment(email="f@x.com")

        # Default limit=5 returns top 5
        top = s.read_leaderboard()
        assert [(e["email"], e["count"]) for e in top] == [
            ("c@x.com", 7),
            ("a@x.com", 5),
            ("b@x.com", 3),
            ("d@x.com", 1),
            ("e@x.com", 1),  # ties break alphabetically (d < e < f)
        ]

        # limit=2 truncates
        top2 = s.read_leaderboard(limit=2)
        assert [e["email"] for e in top2] == ["c@x.com", "a@x.com"]

        # limit=0 returns []
        assert s.read_leaderboard(limit=0) == []


class TestLegacyMigration:
    def test_legacy_count_only_file_reads_cleanly(self, tmp_path):
        """A file written by the previous version contains only a `count`
        field. read_count must return it; read_leaderboard returns []."""
        path = tmp_path / "stats.json"
        path.write_text(json.dumps({"count": 7}), encoding="utf-8")

        s = GuidesStats(_settings_for(path))
        assert s.read_count() == 7
        assert s.read_leaderboard() == []

    def test_legacy_file_migrates_on_next_increment(self, tmp_path):
        """The next increment after reading a legacy file must NOT lose
        the prior count, and must write the new {count, by_user} shape."""
        path = tmp_path / "stats.json"
        path.write_text(json.dumps({"count": 7}), encoding="utf-8")

        s = GuidesStats(_settings_for(path))
        assert s.increment(email="alice@ifixit.com") == 8

        # On-disk shape is upgraded.
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk == {"count": 8, "by_user": {"alice@ifixit.com": 1}}

        # New instance sees the migrated state.
        s2 = GuidesStats(_settings_for(path))
        assert s2.read_count() == 8
        assert s2.read_leaderboard() == [
            {"email": "alice@ifixit.com", "count": 1},
        ]
