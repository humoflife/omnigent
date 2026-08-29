"""The browserless login flows: mode detection, device grant, OIDC ticket.

This module is a deliberate copy of its Slack counterpart (see ``DESIGN.md``),
so these tests exist for a second reason beyond covering the code: they pin the
behaviour independently, so a change to one copy that breaks the contract is
caught here rather than only in the sibling's suite.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from omnigent_discord.oauth import (
    AuthMode,
    AuthorizationDeniedError,
    AuthorizationExpiredError,
    DeviceFlowClient,
    DeviceGrantUnavailableError,
    OAuthError,
    probe_auth_mode,
    start_login,
)

BASE = "http://omnigent.test"


# ── auth-mode detection ───────────────────────────────────────────────────


@respx.mock
async def test_probe_header_mode() -> None:
    # An unauthenticated 200 means a proxy already asserted identity.
    respx.get(f"{BASE}/v1/me").mock(return_value=httpx.Response(200, json={"user_id": "u"}))
    assert await probe_auth_mode(BASE) is AuthMode.HEADER


@respx.mock
async def test_probe_accounts_mode() -> None:
    respx.get(f"{BASE}/v1/me").mock(return_value=httpx.Response(401, json={"login_url": "/login"}))
    assert await probe_auth_mode(BASE) is AuthMode.ACCOUNTS


@respx.mock
async def test_probe_oidc_mode() -> None:
    respx.get(f"{BASE}/v1/me").mock(
        return_value=httpx.Response(401, json={"login_url": "/auth/login"})
    )
    assert await probe_auth_mode(BASE) is AuthMode.OIDC


@respx.mock
async def test_probe_unknown_401_defaults_to_oidc() -> None:
    # The ticket endpoint surfaces a clear error if the server can't do it, so
    # guessing OIDC fails more usefully than guessing accounts.
    respx.get(f"{BASE}/v1/me").mock(return_value=httpx.Response(401, json={}))
    assert await probe_auth_mode(BASE) is AuthMode.OIDC


@respx.mock
async def test_probe_unreachable_server_raises() -> None:
    respx.get(f"{BASE}/v1/me").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(OAuthError):
        await probe_auth_mode(BASE)


# ── device grant (accounts mode) ──────────────────────────────────────────


def _device_authorize() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "device_code": "dc",
            "user_code": "ABCD-2345",
            "verification_uri": f"{BASE}/oauth/device",
            "verification_uri_complete": f"{BASE}/oauth/device?user_code=ABCD-2345",
            "expires_in": 600,
            "interval": 0,
        },
    )


@respx.mock
async def test_start_login_device_grant_offers_a_one_click_link() -> None:
    respx.get(f"{BASE}/v1/me").mock(return_value=httpx.Response(401, json={"login_url": "/login"}))
    respx.post(f"{BASE}/oauth/device/authorize").mock(return_value=_device_authorize())
    pending = await start_login(BASE, client_id="Discord-Omnigent-Acme")
    try:
        # The code is prefilled so the user does not retype it; the short code
        # is still shown so they can confirm the consent page is this request.
        assert "user_code=ABCD-2345" in pending.verification_url
        assert pending.user_code == "ABCD-2345"
    finally:
        await pending.close()


@respx.mock
async def test_start_login_header_mode_is_refused_not_attempted() -> None:
    # A proxy-mode server mounts no login endpoint, so firing a device-grant
    # request would 404 with a confusing message.
    respx.get(f"{BASE}/v1/me").mock(return_value=httpx.Response(200, json={"user_id": "u"}))
    with pytest.raises(OAuthError, match="header/proxy"):
        await start_login(BASE, client_id="Discord-Omnigent")


@respx.mock
async def test_start_login_reports_a_server_without_the_device_grant() -> None:
    # Default-off server-side, so the routes fall through to the SPA catch-all.
    respx.get(f"{BASE}/v1/me").mock(return_value=httpx.Response(401, json={"login_url": "/login"}))
    respx.post(f"{BASE}/oauth/device/authorize").mock(return_value=httpx.Response(404))
    with pytest.raises(DeviceGrantUnavailableError):
        await start_login(BASE, client_id="Discord-Omnigent")


@respx.mock
async def test_device_poll_pending_then_success() -> None:
    respx.get(f"{BASE}/v1/me").mock(return_value=httpx.Response(401, json={"login_url": "/login"}))
    respx.post(f"{BASE}/oauth/device/authorize").mock(return_value=_device_authorize())
    respx.post(f"{BASE}/oauth/token").mock(
        side_effect=[
            httpx.Response(400, json={"error": "authorization_pending"}),
            httpx.Response(200, json={"access_token": "a", "refresh_token": "r"}),
        ]
    )
    pending = await start_login(BASE, client_id="Discord-Omnigent")
    try:
        result = await pending.poll()
    finally:
        await pending.close()
    assert (result.access_token, result.refresh_token) == ("a", "r")


@respx.mock
async def test_device_poll_denied() -> None:
    respx.get(f"{BASE}/v1/me").mock(return_value=httpx.Response(401, json={"login_url": "/login"}))
    respx.post(f"{BASE}/oauth/device/authorize").mock(return_value=_device_authorize())
    respx.post(f"{BASE}/oauth/token").mock(
        return_value=httpx.Response(400, json={"error": "access_denied"})
    )
    pending = await start_login(BASE, client_id="Discord-Omnigent")
    try:
        with pytest.raises(AuthorizationDeniedError):
            await pending.poll()
    finally:
        await pending.close()


@respx.mock
async def test_device_poll_malformed_200_raises_oauth_error() -> None:
    # A malformed 200 would otherwise raise KeyError inside the background poll
    # task and strand the setup message on "waiting for approval…".
    respx.get(f"{BASE}/v1/me").mock(return_value=httpx.Response(401, json={"login_url": "/login"}))
    respx.post(f"{BASE}/oauth/device/authorize").mock(return_value=_device_authorize())
    respx.post(f"{BASE}/oauth/token").mock(return_value=httpx.Response(200, json={"nope": 1}))
    pending = await start_login(BASE, client_id="Discord-Omnigent")
    try:
        with pytest.raises(OAuthError, match="Malformed"):
            await pending.poll()
    finally:
        await pending.close()


# ── OIDC ticket flow ──────────────────────────────────────────────────────


@respx.mock
async def test_start_login_oidc_ticket() -> None:
    respx.get(f"{BASE}/v1/me").mock(
        return_value=httpx.Response(401, json={"login_url": "/auth/login"})
    )
    respx.post(f"{BASE}/auth/cli-login").mock(
        return_value=httpx.Response(200, json={"ticket": "t", "login_url": "/auth/login?ticket=t"})
    )
    pending = await start_login(BASE, client_id="ignored-in-oidc")
    try:
        assert pending.verification_url == f"{BASE}/auth/login?ticket=t"
        # The IdP page needs no code, so none is shown.
        assert pending.user_code == ""
    finally:
        await pending.close()


@respx.mock
async def test_oidc_poll_returns_the_session_jwt_without_a_refresh_token() -> None:
    respx.get(f"{BASE}/v1/me").mock(
        return_value=httpx.Response(401, json={"login_url": "/auth/login"})
    )
    respx.post(f"{BASE}/auth/cli-login").mock(
        return_value=httpx.Response(200, json={"ticket": "t", "login_url": "/auth/login?ticket=t"})
    )
    respx.get(f"{BASE}/auth/cli-poll").mock(
        side_effect=[httpx.Response(202), httpx.Response(200, json={"token": "jwt"})]
    )
    pending = await start_login(BASE, client_id="x")
    try:
        result = await pending.poll()
    finally:
        await pending.close()
    assert result.access_token == "jwt"
    # No refresh token: the session lasts its TTL, then the user signs in again.
    assert result.refresh_token == ""


@respx.mock
async def test_oidc_poll_expired_ticket() -> None:
    respx.get(f"{BASE}/v1/me").mock(
        return_value=httpx.Response(401, json={"login_url": "/auth/login"})
    )
    respx.post(f"{BASE}/auth/cli-login").mock(
        return_value=httpx.Response(200, json={"ticket": "t", "login_url": "/auth/login?ticket=t"})
    )
    respx.get(f"{BASE}/auth/cli-poll").mock(return_value=httpx.Response(410))
    pending = await start_login(BASE, client_id="x")
    try:
        with pytest.raises(AuthorizationExpiredError):
            await pending.poll()
    finally:
        await pending.close()


# ── refresh / revoke, and the optional client secret ──────────────────────


@respx.mock
async def test_refresh_rotates_the_pair() -> None:
    respx.post(f"{BASE}/oauth/token").mock(
        return_value=httpx.Response(200, json={"access_token": "a2", "refresh_token": "r2"})
    )
    client = DeviceFlowClient(BASE)
    try:
        result = await client.refresh("r1")
    finally:
        await client.aclose()
    assert (result.access_token, result.refresh_token) == ("a2", "r2")


@respx.mock
async def test_the_client_secret_is_sent_when_configured() -> None:
    # A server with OMNIGENT_DEVICE_CLIENT_SECRET set accepts only this client.
    route = respx.post(f"{BASE}/oauth/token").mock(
        return_value=httpx.Response(200, json={"access_token": "a", "refresh_token": "r"})
    )
    client = DeviceFlowClient(BASE, client_secret="s3cret")
    try:
        await client.refresh("r1")
    finally:
        await client.aclose()
    assert route.calls.last.request.headers["X-Omnigent-Client-Secret"] == "s3cret"


@respx.mock
async def test_no_client_secret_sends_no_header() -> None:
    route = respx.post(f"{BASE}/oauth/token").mock(
        return_value=httpx.Response(200, json={"access_token": "a", "refresh_token": "r"})
    )
    client = DeviceFlowClient(BASE)
    try:
        await client.refresh("r1")
    finally:
        await client.aclose()
    assert "X-Omnigent-Client-Secret" not in route.calls.last.request.headers


@respx.mock
async def test_revoke_is_best_effort() -> None:
    # Logout must clear the local token even when the server can't be reached.
    respx.post(f"{BASE}/oauth/revoke").mock(side_effect=httpx.ConnectError("down"))
    client = DeviceFlowClient(BASE)
    try:
        await client.revoke("r1")  # must not raise
    finally:
        await client.aclose()
