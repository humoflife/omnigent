"""Turn-end detection, stream reconnect, and error classification.

The client is harness-agnostic and shared in spirit with the Slack integration,
but it is a separate copy here — so the invariants that were hardest to get
right (which ``session.status`` ends a turn, and what a mid-stream drop means)
are pinned independently rather than assumed.
"""

from __future__ import annotations

import httpx
import omnigent_discord.omnigent as omnigent_module
import pytest
import respx
from fakes import sse_delta, sse_status
from omnigent_discord.events import extract_elicitation_request, session_status
from omnigent_discord.omnigent import (
    AuthRequiredError,
    ClientAuth,
    HarnessNotConfiguredError,
    HostUnavailableError,
    OmnigentClient,
    OmnigentError,
    ServerUnreachableError,
    StreamInterruptedError,
)

BASE = "http://omnigent.test"
SESSION = "conv_1"


def _client(**kwargs: object) -> OmnigentClient:
    return OmnigentClient(BASE, **kwargs)  # type: ignore[arg-type]


def _mount(router: respx.MockRouter, body: str) -> None:
    """Serve one SSE body plus the endpoints a turn always touches."""
    router.post(f"{BASE}/v1/sessions/{SESSION}/events").mock(
        return_value=httpx.Response(202, json={})
    )
    router.get(f"{BASE}/health").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    router.get(f"{BASE}/v1/sessions/{SESSION}/stream").mock(
        return_value=httpx.Response(200, text=body)
    )


async def _run(body: str) -> list[dict[str, object]]:
    async with respx.mock(assert_all_called=False) as router:
        _mount(router, body)
        client = _client()
        try:
            return [e async for e in client.run_turn(SESSION, "hi", idle_grace_seconds=1)]
        finally:
            await client.aclose()


# ── parsing ───────────────────────────────────────────────────────────────


def test_session_status_distinguishes_id_bearing_from_id_less() -> None:
    # This distinction is the whole basis of turn-end detection.
    assert session_status({"type": "session.status", "status": "idle"}) == ("idle", None)
    assert session_status({"type": "session.status", "status": "idle", "response_id": "r1"}) == (
        "idle",
        "r1",
    )
    assert session_status({"type": "response.completed"}) is None


def test_elicitation_resolves_against_its_own_target_session() -> None:
    # A sub-agent prompt mirrored into an ancestor stream carries its own target.
    event = {
        "type": "response.elicitation_request",
        "elicitation_id": "el_1",
        "params": {"message": "ok?", "target_session_id": "conv_sub"},
    }
    request = extract_elicitation_request(event, "conv_parent")
    assert request is not None and request.session_id == "conv_sub"


def test_elicitation_defaults_to_the_streaming_session() -> None:
    event = {
        "type": "response.elicitation_request",
        "elicitation_id": "el_1",
        "params": {"message": "ok?"},
    }
    request = extract_elicitation_request(event, "conv_parent")
    assert request is not None and request.session_id == "conv_parent"


# ── turn-end detection ────────────────────────────────────────────────────


async def test_terminal_harness_ends_on_its_matching_id_bearing_idle() -> None:
    events = await _run(
        sse_status("running", "r1") + sse_delta("answer", "m1") + sse_status("idle", "r1")
    )
    assert [e.get("type") for e in events][-1] == "session.status"
    assert any(e.get("delta") == "answer" for e in events)


async def test_mid_answer_flap_does_not_truncate_the_reply() -> None:
    # claude-native's PTY watcher emits a bare idle during generation lulls;
    # ending there would cut the answer in half.
    events = await _run(
        sse_status("running", "r1")
        + sse_delta("first ", "m1")
        + sse_status("idle")  # the flap
        + sse_delta("second", "m1")
        + sse_status("idle", "r1")
    )
    text = "".join(str(e.get("delta") or "") for e in events)
    assert text == "first second"


async def test_in_process_harness_ends_on_its_id_less_idle() -> None:
    # Nothing is ever "open" there — every status is id-less — so the id-less
    # idle after real production is the real end.
    events = await _run(
        sse_status("running")
        + sse_delta("answer", None)
        + 'data: {"type":"response.completed"}\n\n'
        + sse_status("idle")
        + sse_delta("must not appear", None)
    )
    text = "".join(str(e.get("delta") or "") for e in events)
    assert text == "answer"


async def test_cold_start_flap_before_any_output_is_ignored() -> None:
    # An id-less running→idle pair before the first token looks identical to the
    # in-process end, except nothing has been produced yet.
    events = await _run(
        sse_status("running")
        + sse_status("idle")  # cold-start flap
        + sse_status("running", "r1")
        + sse_delta("answer", "m1")
        + sse_status("idle", "r1")
    )
    assert "".join(str(e.get("delta") or "") for e in events) == "answer"


async def test_stale_pre_turn_idle_is_ignored() -> None:
    # Resuming an idle session replays its current status before our turn's
    # running edge; ending there would return no events at all.
    events = await _run(
        sse_status("idle", "r0")  # stale, replayed on connect
        + sse_status("running", "r1")
        + sse_delta("answer", "m1")
        + sse_status("idle", "r1")
    )
    assert "".join(str(e.get("delta") or "") for e in events) == "answer"


async def test_waiting_never_ends_a_turn() -> None:
    # Both harnesses use `waiting` for "parked on sub-agents / async work".
    events = await _run(
        sse_status("running", "r1")
        + sse_status("waiting", "r1")
        + sse_delta("answer", "m1")
        + sse_status("idle", "r1")
    )
    assert "".join(str(e.get("delta") or "") for e in events) == "answer"


async def test_bare_failed_ends_the_turn_even_with_nothing_produced() -> None:
    # `failed` comes only from an authoritative hook, never the PTY watcher, so
    # it must not wait for production the way a bare `idle` does.
    events = await _run(sse_status("running") + sse_status("failed"))
    assert [session_status(e) for e in events][-1] == ("failed", None)


async def test_hard_terminal_event_ends_the_turn() -> None:
    events = await _run(
        sse_status("running", "r1")
        + 'data: {"type":"turn.failed","error":{"message":"boom"}}\n\n'
        + sse_delta("must not appear", "m1")
    )
    assert not any(e.get("delta") for e in events)


async def test_dead_socket_ends_the_turn_rather_than_hanging() -> None:
    # The stream never sends [DONE] and heartbeats every ~15s, so silence past
    # the grace window is the one thing no event can tell us about.
    async def _silent():
        yield sse_status("running", "r1").encode()
        # No further frames and no close — the read simply never completes.
        await __import__("asyncio").sleep(3600)

    async with respx.mock(assert_all_called=False) as router:
        router.post(f"{BASE}/v1/sessions/{SESSION}/events").mock(
            return_value=httpx.Response(202, json={})
        )
        router.get(f"{BASE}/health").mock(return_value=httpx.Response(200))
        router.get(f"{BASE}/v1/sessions/{SESSION}/stream").mock(
            return_value=httpx.Response(200, stream=_silent())
        )
        client = _client()
        try:
            events = [e async for e in client.run_turn(SESSION, "hi", idle_grace_seconds=0.05)]
        finally:
            await client.aclose()
    assert [e.get("type") for e in events] == ["session.status"]


async def test_malformed_frame_is_skipped_not_fatal() -> None:
    events = await _run(
        sse_status("running", "r1")
        + "data: {not json\n\n"
        + sse_delta("answer", "m1")
        + sse_status("idle", "r1")
    )
    assert "".join(str(e.get("delta") or "") for e in events) == "answer"


# ── reconnect ─────────────────────────────────────────────────────────────


async def _dropping(body: str):
    yield body.encode()
    raise httpx.RemoteProtocolError("peer closed connection (incomplete chunked read)")


async def test_severed_stream_reconnects_without_double_rendering() -> None:
    # A proxy duration cap severs the stream mid-turn; on re-open the server
    # replays the streamed-so-far text as one cumulative delta.
    legs = [
        sse_status("running", "r1") + sse_delta("Hello ", "m1"),
        sse_delta("Hello world", "m1") + sse_status("idle", "r1"),
    ]
    calls = {"n": 0}

    def _stream(_request: httpx.Request) -> httpx.Response:
        index = calls["n"]
        calls["n"] += 1
        if index == 0:
            return httpx.Response(200, stream=_dropping(legs[0]))
        return httpx.Response(200, text=legs[1])

    async with respx.mock(assert_all_called=False) as router:
        router.post(f"{BASE}/v1/sessions/{SESSION}/events").mock(
            return_value=httpx.Response(202, json={})
        )
        router.get(f"{BASE}/health").mock(return_value=httpx.Response(200))
        router.get(f"{BASE}/v1/sessions/{SESSION}").mock(
            return_value=httpx.Response(200, json={"status": "running"})
        )
        router.get(f"{BASE}/v1/sessions/{SESSION}/stream").mock(side_effect=_stream)
        client = _client()
        try:
            events = [e async for e in client.run_turn(SESSION, "hi", idle_grace_seconds=1)]
        finally:
            await client.aclose()
        submits = [c for c in router.calls if c.request.url.path.endswith("/events")]
    assert "".join(str(e.get("delta") or "") for e in events) == "Hello world"
    # The turn is submitted ONCE: the server keeps running it across the drop,
    # so re-submitting would start a second turn.
    assert len(submits) == 1


async def test_drop_after_the_turn_finished_stops_cleanly() -> None:
    async def _stream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_dropping(sse_status("running", "r1")))

    async with respx.mock(assert_all_called=False) as router:
        router.post(f"{BASE}/v1/sessions/{SESSION}/events").mock(
            return_value=httpx.Response(202, json={})
        )
        router.get(f"{BASE}/health").mock(return_value=httpx.Response(200))
        router.get(f"{BASE}/v1/sessions/{SESSION}").mock(
            return_value=httpx.Response(200, json={"status": "idle"})
        )
        router.get(f"{BASE}/v1/sessions/{SESSION}/stream").mock(side_effect=_stream)
        client = _client()
        try:
            events = [e async for e in client.run_turn(SESSION, "hi", idle_grace_seconds=1)]
        finally:
            await client.aclose()
    assert [e.get("type") for e in events] == ["session.status"]


async def test_repeated_drops_surface_as_a_stream_interruption_not_a_dead_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The server stayed reachable throughout, so "reconfigure your server" would
    # be the wrong advice. The real backoff grows per attempt; shrink it so the
    # test exercises the exhaustion path without waiting it out.
    monkeypatch.setattr(omnigent_module, "_STREAM_RECONNECT_BACKOFF_S", 0.0)

    async def _stream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_dropping(sse_status("running", "r1")))

    async with respx.mock(assert_all_called=False) as router:
        router.post(f"{BASE}/v1/sessions/{SESSION}/events").mock(
            return_value=httpx.Response(202, json={})
        )
        router.get(f"{BASE}/health").mock(return_value=httpx.Response(200))
        router.get(f"{BASE}/v1/sessions/{SESSION}").mock(
            return_value=httpx.Response(200, json={"status": "running"})
        )
        router.get(f"{BASE}/v1/sessions/{SESSION}/stream").mock(side_effect=_stream)
        client = _client()
        try:
            with pytest.raises(StreamInterruptedError):
                async for _ in client.run_turn(SESSION, "hi", idle_grace_seconds=1):
                    pass
        finally:
            await client.aclose()


# ── error classification ──────────────────────────────────────────────────


async def test_transport_failure_is_an_unreachable_server() -> None:
    async with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/health").mock(side_effect=httpx.ConnectError("refused"))
        client = _client()
        try:
            with pytest.raises(ServerUnreachableError):
                await client.check_health()
        finally:
            await client.aclose()


async def test_401_asks_for_authentication() -> None:
    async with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/v1/agents").mock(return_value=httpx.Response(401))
        client = _client()
        try:
            with pytest.raises(AuthRequiredError):
                await client.list_agents()
        finally:
            await client.aclose()


async def test_proxy_login_redirect_is_treated_as_an_auth_wall() -> None:
    # A Databricks-App proxy 302s to its OAuth login rather than returning 401.
    async with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/v1/agents").mock(
            return_value=httpx.Response(302, headers={"location": "/oidc/v1/authorize"})
        )
        client = _client()
        try:
            with pytest.raises(AuthRequiredError):
                await client.list_agents()
        finally:
            await client.aclose()


async def test_offline_host_is_reported_as_host_unavailable() -> None:
    async with respx.mock(assert_all_called=False) as router:
        router.post(f"{BASE}/v1/hosts/h1/runners").mock(return_value=httpx.Response(409))
        client = _client()
        try:
            with pytest.raises(HostUnavailableError):
                await client.launch_runner(SESSION, workspace="/w", host_id="h1")
        finally:
            await client.aclose()


async def test_launching_without_a_workspace_is_refused_before_any_request() -> None:
    client = _client()
    try:
        with pytest.raises(OmnigentError, match="workspace path is required"):
            await client.launch_runner(SESSION, workspace="", host_id="h1")
    finally:
        await client.aclose()


async def test_unconfigured_harness_surfaces_the_servers_curated_message() -> None:
    # The one error body safe to echo: it is actionable guidance, not internals.
    async with respx.mock(assert_all_called=False) as router:
        router.post(f"{BASE}/v1/sessions").mock(
            return_value=httpx.Response(
                412,
                json={
                    "error": {
                        "code": "harness_not_configured",
                        "message": "Run `omnigent setup` on the host.",
                    }
                },
            )
        )
        client = _client()
        try:
            with pytest.raises(HarnessNotConfiguredError, match="omnigent setup"):
                await client.create_session("ag_1", "title")
        finally:
            await client.aclose()


async def test_server_error_body_is_never_put_in_the_exception() -> None:
    # The message reaches a channel visible to everyone in it.
    async with respx.mock(assert_all_called=False) as router:
        router.post(f"{BASE}/v1/sessions").mock(
            return_value=httpx.Response(500, text="Traceback: /opt/omnigent/secret.py")
        )
        client = _client()
        try:
            with pytest.raises(OmnigentError) as excinfo:
                await client.create_session("ag_1", "title")
        finally:
            await client.aclose()
    assert "/opt/omnigent" not in str(excinfo.value)


# ── delegated auth ────────────────────────────────────────────────────────


async def test_expired_token_is_refreshed_once_and_the_request_retried() -> None:
    calls = {"n": 0}

    def _agents(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.headers.get("authorization") == "Bearer old":
            return httpx.Response(401)
        return httpx.Response(200, json={"data": [{"id": "ag_1"}]})

    refreshes = {"n": 0}

    async def _refresh() -> str:
        refreshes["n"] += 1
        return "new"

    async with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/v1/agents").mock(side_effect=_agents)
        client = _client(auth=ClientAuth("old", _refresh))
        try:
            agents = await client.list_agents()
        finally:
            await client.aclose()
    assert agents == [{"id": "ag_1"}]
    assert refreshes["n"] == 1
    assert calls["n"] == 2


async def test_concurrent_refreshes_rotate_the_grant_only_once() -> None:
    # A rotating refresh token is single-use; a second rotation would revoke the
    # whole grant and log the user out mid-session.
    import asyncio

    refreshes = {"n": 0}

    async def _refresh() -> str:
        refreshes["n"] += 1
        await asyncio.sleep(0)
        return f"new-{refreshes['n']}"

    auth = ClientAuth("old", _refresh)
    results = await asyncio.gather(*(auth.refresh("old") for _ in range(5)))
    assert refreshes["n"] == 1
    assert set(results) == {"new-1"}


async def test_harness_that_never_starts_ends_the_turn_immediately() -> None:
    # A harness that can't launch at all (no tmux, missing binary) reports the
    # failure with NO ``running`` edge and no delta — so the id-less ``failed``
    # that follows must not be dismissed as a status replayed from before the
    # turn. Getting this wrong strands the user on an unchanging placeholder
    # until the idle grace expires, minutes after the server already said the
    # turn was dead. Sequence observed live against claude-native without tmux.
    events = await _run(
        'data: {"type":"session.input.consumed"}\n\n'
        'data: {"type":"response.error","error":{"message":'
        '"Native Claude terminal failed to start"}}\n\n'
        + sse_status("failed")
        + 'data: {"type":"session.heartbeat"}\n\n'  # must never be reached
    )
    types = [e.get("type") for e in events]
    assert types[-1] == "session.status"
    assert "session.heartbeat" not in types


async def test_in_band_error_is_still_forwarded_to_the_caller() -> None:
    # The service reads ``response.error`` to mark the turn errored, so ending
    # promptly must not swallow the event on the way out.
    events = await _run(
        'data: {"type":"response.error","error":{"message":"boom"}}\n\n' + sse_status("failed")
    )
    assert any(e.get("type") == "response.error" for e in events)


async def test_stale_idle_replayed_before_the_turn_is_still_ignored() -> None:
    # The guard the fix above narrows must keep doing its original job:
    # resuming an idle session replays its current status before our turn's
    # running edge, and ending there would return no answer at all.
    events = await _run(
        sse_status("idle", "r0")
        + sse_status("running", "r1")
        + sse_delta("answer", "m1")
        + sse_status("idle", "r1")
    )
    assert "".join(str(e.get("delta") or "") for e in events) == "answer"
