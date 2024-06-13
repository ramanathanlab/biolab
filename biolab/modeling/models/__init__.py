"""Submodule for LM model instances."""
from biolab.api.registry import import_submodules

# Dynamically import all submodules to trigger registration of models
import_submodules(__name__)
