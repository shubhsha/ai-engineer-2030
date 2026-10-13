import pytest

from ai_engineer import __version__
from ai_engineer.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_doctor_runs_against_the_repository(capsys):
    code = main(["doctor", "--root", "."])
    out = capsys.readouterr().out
    assert "python" in out
    assert code in (0, 1)
