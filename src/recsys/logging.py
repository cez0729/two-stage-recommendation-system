"""Consistent logging setup for command-line jobs and services."""

import logging
import os


def configure_logging() -> None:
    """Configure a compact process-wide logging format."""
    level = os.getenv("RECSYS_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

