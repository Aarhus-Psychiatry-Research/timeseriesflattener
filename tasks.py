"""
Most tasks are in the makefile.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def echo_header(msg: str):
    print(f"\n--- {msg} ---")


@dataclass
class Emo:
    DO = "🤖"
    GOOD = "✅"
    FAIL = "🚨"
    WARN = "🚧"
    SYNC = "🚂"
    PY = "🐍"
    CLEAN = "🧹"
    TEST = "🧪"
    COMMUNICATE = "📣"
    EXAMINE = "🔍"


def test_for_rej():
    # Get all paths in current directory or subdirectories that end in .rej
    rej_files = list(Path().rglob("*.rej"))

    if len(rej_files) > 0:
        print(f"\n{Emo.FAIL} Found .rej files leftover from cruft update.\n")
        for file in rej_files:
            print(f"    /{file}")
        print("\nResolve the conflicts and try again. \n")
        exit(1)
