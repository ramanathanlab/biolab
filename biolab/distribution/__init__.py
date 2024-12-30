"""Submodule for distribution with parsl."""

from __future__ import annotations

from .parsl import BaseComputeSettings
from .parsl import SingleNodeSettings

ParslConfigTypes = SingleNodeSettings
