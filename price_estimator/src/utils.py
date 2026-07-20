"""General utility functions shared across the project."""

import logging
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a readable format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dirs(*paths: Path) -> None:
    """Create one or more directories (and parents) if they do not exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
