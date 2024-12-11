"""biolab package root."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import warnings

__version__ = importlib_metadata.version('biolab')


# Ignore warnings from pydantic. The specific warning is
# "Expected `Union[$ALL_TASK_CONFIGS]` but got `CONFIG_INSTANCE`
# - serialized value may not be as expected" This is core from the
# use of inheritance in the task configuration classes. I am silencing
# this warning as it does not affect the functionality of the configs.
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
