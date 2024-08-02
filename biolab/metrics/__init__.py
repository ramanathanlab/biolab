from __future__ import annotations  # noqa: D104

from biolab.api.registry import import_submodules

# Dynamically import all submodules to trigger registration of metrics
import_submodules(__name__)
