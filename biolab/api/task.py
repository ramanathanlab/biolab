from typing import Protocol

from biolab.api.config import BaseConfig
from biolab.api.modeling import LM


class TaskConfig(BaseConfig):
    """General configuration for a task."""

    dataset_name_or_path: str
    output_transform: str = "null_transform"


class Task(Protocol):
    """A general task interface."""

    def __init__(self, config: TaskConfig):
        """Initialize the task"""
        ...

    def evaluate(self, model: LM):
        """Evaluate the task"""
        ...
