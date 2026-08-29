"""Guard the three modules Discord shares with Slack against silent drift.

``events.py``, ``omnigent.py`` and ``oauth.py`` are deliberate copies of their
``omnigent_slack`` counterparts — see ``DESIGN.md`` for why two standalone bot
distributions don't share a package. Copying is defensible; copying with no
mechanism to keep the copies honest is not: a turn-hang fixed in one of them
once shipped while the other stayed broken, because nothing compared them.

These tests compare the two trees at the level that matters — executable code,
ignoring comments and docstring prose, which legitimately differ (one says
"Slack", the other "Discord"). A deliberate divergence is registered in
``_EXPECTED_DIVERGENCES`` with the reason, so the next person sees which
differences are intended and which are an accident.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_HERE = pathlib.Path(__file__).resolve()
_INTEGRATIONS = _HERE.parents[2]
_DISCORD = _INTEGRATIONS / "discord" / "src" / "omnigent_discord"
_SLACK = _INTEGRATIONS / "slack" / "src" / "omnigent_slack"

# The modules that are copies rather than ports.
SHARED_MODULES = ("events.py", "omnigent.py", "oauth.py")

# Divergences that are intended. Keyed by module, each entry names what differs
# and why, so a reviewer can tell a deliberate change from an accidental one.
_EXPECTED_DIVERGENCES: dict[str, str] = {}


def _despecialize(text: str) -> str:
    """Erase the two things that MUST differ between the copies.

    Each package imports its own sibling modules, and a couple of log lines
    name the platform they serve. Comparing those would make this test fail
    permanently — and a test that can never pass gets deleted rather than
    fixed. Everything else is fair game.
    """
    for slack, discord in (
        ("omnigent_slack", "PKG"),
        ("omnigent_discord", "PKG"),
        ("Slack", "PLATFORM"),
        ("Discord", "PLATFORM"),
    ):
        text = text.replace(slack, discord)
    return text


def _normalized(path: pathlib.Path) -> str:
    """The module's executable code, with comments and docstrings stripped.

    Comments and docstrings differ by design (each names its own chat
    platform), so comparing them would make this test fail constantly and get
    deleted. What must not differ is the logic.
    """
    tree = ast.parse(_despecialize(path.read_text(encoding="utf-8")))
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, indent=1)


@pytest.mark.parametrize("module", SHARED_MODULES)
def test_shared_module_has_not_drifted_from_slack(module: str) -> None:
    discord_path, slack_path = _DISCORD / module, _SLACK / module
    if not slack_path.is_file():  # pragma: no cover - slack package absent
        pytest.skip(f"{slack_path} not present")

    if _normalized(discord_path) == _normalized(slack_path):
        return

    reason = _EXPECTED_DIVERGENCES.get(module)
    assert reason is not None, (
        f"{module} differs in executable code between omnigent_discord and "
        f"omnigent_slack. These are deliberate copies (see DESIGN.md), so a "
        f"fix to one almost always belongs in the other — a turn-hang fix "
        f"once landed in only one copy and left the sibling broken. Apply the "
        f"change to both, or record the intended divergence in "
        f"_EXPECTED_DIVERGENCES with the reason."
    )


def test_every_registered_divergence_is_real() -> None:
    """A divergence that no longer exists must be removed from the registry.

    Otherwise the list becomes a graveyard that silently permits future drift
    in that module.
    """
    for module, reason in _EXPECTED_DIVERGENCES.items():
        assert module in SHARED_MODULES, f"{module} is not a shared module"
        assert _normalized(_DISCORD / module) != _normalized(_SLACK / module), (
            f"{module} is registered as diverging ({reason}) but the two copies "
            f"now match — drop the entry so real drift is caught again."
        )
