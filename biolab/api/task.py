from __future__ import annotations  # noqa: D100

from pathlib import Path
from typing import Protocol

from biolab.api.config import BaseConfig
from biolab.api.modeling import LM


class TaskConfig(BaseConfig):
    """General configuration for a task."""

    dataset_name_or_path: str

    # These get instantiated later
    output_dir: Path | None = None
    cache_dir: Path | None = None


class Task(Protocol):
    """A general task interface."""

    def __init__(self, config: TaskConfig):
        """Initialize the task."""
        self.config = config
        self.output_dir = self.config.output_dir
        self.cache_dir = self.config.cache_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir is None:
            self.cache_dir = Path(self.output_dir) / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, model: LM):
        """Evaluate the task."""
        ...
