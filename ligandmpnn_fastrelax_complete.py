#!/usr/bin/env python3
"""Iterative LigandMPNN-FastRelax for Backbone Optimization.

Compatibility wrapper for the split CLI/pipeline modules.
"""

from cli import main, parse_arguments
from pipeline import LigandMPNNFastRelax

__all__ = ["LigandMPNNFastRelax", "main", "parse_arguments"]


if __name__ == "__main__":
    raise SystemExit(main())
