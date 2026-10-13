"""The ``aie`` command: small, scriptable entry points into the package.

Each subcommand imports its module lazily, so ``aie --help`` stays fast
and optional extras are only needed by the commands that use them.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from ai_engineer import __version__

Handler = Callable[[argparse.Namespace], int]


def _doctor(args: argparse.Namespace) -> int:
    from ai_engineer.foundations import doctor

    checks = doctor.check_environment(Path(args.root))
    print(doctor.format_report(checks))
    return doctor.exit_code(checks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aie", description="Companion tools for AI Engineer 2030."
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check the local toolchain")
    doctor.add_argument("--root", default=".", help="repository root")
    doctor.set_defaults(handler=_doctor)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)
