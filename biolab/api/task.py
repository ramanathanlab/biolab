"""API for tasks."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path

from biolab.api.config import BaseConfig
from biolab.api.metric import Metric
from biolab.api.modeling import LM


# TODO: think about adding metrics as a literal type
class TaskConfig(BaseConfig):
    """General configuration for a task."""

    dataset_name_or_path: str

    # These get instantiated later
    output_dir: Path | None = None
    cache_dir: Path | None = None


class Task(ABC):
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

    @abstractmethod
    def evaluate(self, model: LM) -> list[Metric]:
        """Evaluate the task and return its metrics."""
        ...
