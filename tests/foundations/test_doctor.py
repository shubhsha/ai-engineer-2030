from pathlib import Path

from ai_engineer.foundations.doctor import (
    Check,
    check_api_key,
    check_environment,
    check_python,
    check_tool,
    check_virtualenv,
    exit_code,
    format_report,
)


def test_python_version_gate():
    assert check_python((3, 11)).ok
    assert check_python((3, 13)).ok
    assert not check_python((3, 10)).ok


def test_virtualenv_detected_by_prefix():
    assert check_virtualenv("/repo/.venv", "/usr").ok
    assert not check_virtualenv("/usr", "/usr").ok


def test_missing_tool_fails_with_hint():
    check = check_tool("ruff", which=lambda name: None)
    assert not check.ok
    assert check.detail == "not on PATH"


def test_api_key_is_optional():
    check = check_api_key({})
    assert not check.ok
    assert not check.required


def test_optional_failures_do_not_fail_the_run():
    checks = [Check("a", True, ""), Check("b", False, "", required=False)]
    assert exit_code(checks) == 0
    assert exit_code([*checks, Check("c", False, "")]) == 1


def test_report_names_every_check(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("")
    checks = check_environment(tmp_path, env={}, which=lambda n: "/bin/x")
    report = format_report(checks)
    for check in checks:
        assert check.name in report
    assert "models.toml" in report  # missing file is reported by name
