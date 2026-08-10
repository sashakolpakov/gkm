"""Entry point for the standalone RoboArm Godel-Kolmogorov machine campaign."""

from .gkm.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
