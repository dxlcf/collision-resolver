from __future__ import annotations

import pytest

from collision_resolver import cli


def test_cli_help_states_no_mesh_input(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "不接收网格输入" in captured.out
    assert "ServiceRequest" in captured.out
    assert "ServiceResult" in captured.out


def test_cli_run_states_self_check_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "CollisionResolverService", lambda config: object())

    cli.main([])

    captured = capsys.readouterr()
    assert "运行时检查通过" in captured.out
    assert "仅执行依赖自检" in captured.out
    assert "不接收网格输入" in captured.out
    assert "ServiceRequest -> ServiceResult" in captured.out
