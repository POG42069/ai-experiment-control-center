"""Allow invoking the CLI via `python -m ecc`."""

from ecc.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
