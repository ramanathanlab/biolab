"""Reporting submodule."""

from __future__ import annotations

from pathlib import Path

from biolab.reporting.reporters import ConsoleReporter
from biolab.reporting.reporters import MarkdownReporter
from biolab.reporting.reporters import PDFReporter
from biolab.reporting.reporters import Reporter
from biolab.reporting.utils import discover_results

# Reporter Factory or Registry
REPORTERS: dict[str, type[Reporter]] = {
    'console': ConsoleReporter,
    'markdown': MarkdownReporter,
    'pdf': PDFReporter,
}
