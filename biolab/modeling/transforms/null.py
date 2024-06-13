from typing import Any

from biolab.api.lm import Transform
from biolab import transform_registry


@transform_registry.register(name="null_transform")
class NullTransform(Transform):
    """A transform that returns the input without any modification."""

    name = "null_transform"

    def apply(self, input: Any, *args, **kwargs) -> Any:
        """Return the input as is.

        Parameters
        ----------
        input : Any
            The input to return without modification.

        Returns
        -------
        Any
            The input, unmodified.
        """
        return input
