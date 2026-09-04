"""SFT entrypoint that dispatches on ``--mode {single|multi}``."""
from __future__ import annotations

import sys

from openfinai_skyrl.train.tokenization import parse_cli


def _pop_mode(argv: list[str]) -> tuple[str, list[str]]:
    """Extract ``--mode {single|multi}`` from argv; accepts both ``--mode X`` and ``--mode=X``."""
    mode: str | None = None
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--mode":
            if i + 1 >= len(argv):
                raise SystemExit("--mode requires a value (single|multi)")
            mode = argv[i + 1]
            i += 2
            continue
        if token.startswith("--mode="):
            mode = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
    if mode not in {"single", "multi"}:
        raise SystemExit(
            f"--mode is required and must be 'single' or 'multi' (got {mode!r})"
        )
    return mode, cleaned


def main() -> None:
    mode, remaining = _pop_mode(list(sys.argv[1:]))
    raw_config = parse_cli(remaining)
    from openfinai_skyrl.train.driver import train
    train(raw_config, mode=mode)


if __name__ == "__main__":
    main()
