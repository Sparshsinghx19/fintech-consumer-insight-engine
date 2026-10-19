"""Tests for public demo mode: configuration, write protection of the
shared index, and the request rate limiter."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config import Settings
from app.demo import DemoModeRestriction, SlidingWindowLimiter, format_wait, require_write_access, try_consume
from app.indexing import build_index
from app.vector_store import VectorStoreManager
from tests.helpers import HashingEmbeddings, make_text_pdf


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path / "raw", processed_dir=tmp_path / "processed", vectorstore_dir=tmp_path / "vs")
    s.ensure_directories()
    return s


# -- configuration ---------------------------------------------------------


def test_demo_mode_is_off_by_default(monkeypatch):
    monkeypatch.delenv("PUBLIC_DEMO_MODE", raising=False)
    assert Settings().public_demo_mode is False


@pytest.mark.parametrize("value, expected", [("true", True), ("1", True), ("YES", True), ("false", False), ("0", False)])
def test_demo_mode_from_environment(monkeypatch, value, expected):
    monkeypatch.setenv("PUBLIC_DEMO_MODE", value)
    assert Settings().public_demo_mode is expected


def test_demo_settings_from_streamlit_secrets(monkeypatch):
    for name in ("PUBLIC_DEMO_MODE", "DEMO_SESSION_REQUEST_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    secrets = {"PUBLIC_DEMO_MODE": "true", "DEMO_SESSION_REQUEST_LIMIT": "5"}
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets=secrets))
    s = Settings()
    assert s.public_demo_mode is True and s.demo_session_request_limit == 5


def test_invalid_demo_limit_is_rejected(monkeypatch):
    monkeypatch.setenv("DEMO_GLOBAL_REQUEST_LIMIT", "lots")
    with pytest.raises(ValueError, match="DEMO_GLOBAL_REQUEST_LIMIT"):
        Settings()


# -- write protection -------------------------------------------------------


def test_require_write_access(settings):
    require_write_access(settings, "Uploading documents")  # no error outside demo mode
    with pytest.raises(DemoModeRestriction, match="Uploading documents is disabled"):
        require_write_access(replace(settings, public_demo_mode=True), "Uploading documents")


@pytest.mark.parametrize("rebuild", [False, True])
def test_indexing_is_refused_in_demo_mode_and_index_is_untouched(settings, rebuild):
    make_text_pdf(settings.data_dir / "trust.pdf", ["Privacy concerns deter wallet adoption.", "Fees are a barrier."])
    with patch("app.vector_store.get_embedding_function", return_value=HashingEmbeddings()):
        vsm = VectorStoreManager(settings)
        build_index(vsm, settings)
        assert vsm.count() == 2

        make_text_pdf(settings.data_dir / "new.pdf", ["A document added after deployment."])
        demo = replace(settings, public_demo_mode=True)
        with pytest.raises(DemoModeRestriction):
            build_index(vsm, demo, rebuild=rebuild)

        assert vsm.count() == 2
        assert vsm.indexed_sources() == {"trust.pdf"}


# -- rate limiting ------------------------------------------------------------


def test_limiter_allows_up_to_limit_then_blocks_until_window_passes():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(max_requests=3, window_seconds=60, clock=clock)

    assert [limiter.acquire() for _ in range(3)] == [True, True, True]
    assert limiter.acquire() is False
    assert limiter.remaining() == 0
    assert limiter.retry_after() == pytest.approx(60)

    clock.now += 30
    assert limiter.acquire() is False
    assert limiter.retry_after() == pytest.approx(30)

    clock.now += 30
    assert limiter.remaining() == 3 and limiter.acquire() is True


def test_limiter_costs_multiple_units():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(max_requests=10, window_seconds=60, clock=clock)
    assert limiter.acquire(cost=8)
    assert limiter.acquire(cost=3) is False  # would exceed 10
    assert limiter.acquire(cost=2) and limiter.remaining() == 0
    assert limiter.retry_after(cost=11) == float("inf")


def test_try_consume_takes_nothing_when_any_limiter_is_full():
    clock = FakeClock()
    session = SlidingWindowLimiter(5, 60, clock=clock)
    shared = SlidingWindowLimiter(1, 60, clock=clock)

    assert try_consume([session, shared], 1) == (True, 0.0)
    allowed, wait = try_consume([session, shared], 1)

    assert allowed is False and wait == pytest.approx(60)
    assert session.remaining() == 4  # the blocked request didn't use the session quota


def test_session_limits_are_independent_but_share_the_global_limit():
    clock = FakeClock()
    shared = SlidingWindowLimiter(3, 60, clock=clock)
    alice, bob = SlidingWindowLimiter(2, 60, clock=clock), SlidingWindowLimiter(2, 60, clock=clock)

    assert try_consume([alice, shared], 2)[0]
    assert try_consume([alice, shared], 1)[0] is False  # Alice's session limit
    assert try_consume([bob, shared], 1)[0]
    assert try_consume([bob, shared], 1)[0] is False  # global limit reached


def test_format_wait():
    assert format_wait(20) == "in about a minute"
    assert format_wait(150) == "in about 3 minutes"
    assert format_wait(float("inf")) == "later"
