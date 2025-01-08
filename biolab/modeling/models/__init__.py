"""Submodule for LM model instances."""

from __future__ import annotations

from typing import Any
from typing import Union

from biolab.distribution import parsl_registry

from .ankh import ankh_models
from .calm import calm_models
from .dnabert import dnabert_models
from .esm import esm_models
from .evo import evo_models
from .genalm import genalm_models
from .genomelm import genomelm_models
from .genslm import genslm_models
from .genslm2 import genslm2_models
from .genslm_next import genslmnext_models
from .nucleotide_transformer import nucleotidetransformer_models
from .protgpt2 import protgpt2_models
from .protrans import protrans_models

model_registry = {
    **ankh_models,
    **calm_models,
    **dnabert_models,
    **esm_models,
    **evo_models,
    **genalm_models,
    **genomelm_models,
    **genslm_models,
    **nucleotidetransformer_models,
    **protgpt2_models,
    **protrans_models,
    **genslm2_models,
    **genslmnext_models,
}

# TODO can we make a base class of the LM that looks for cached model
#      outputs and short-circuits the evaluation?

ModelConfigTypes = Union[*model_registry.keys(),]
ModelTypes = Union[*model_registry.values(),]


# This is a workaround to support optional registration.
# Make a function to combine the config and instance initialization
# since the registry only accepts functions with hashable arguments.
def _factory_fn(**kwargs: dict[str, Any]) -> ModelTypes:
    model_config = kwargs.get('model_config', None)
    if not model_config:
        raise ValueError(
            f'Unknown model config: {kwargs}.' f' Available: {ModelConfigTypes}',
        )

    model_cls = model_registry.get(model_config.__class__)

    return model_cls(model_config)


def get_model(
    model_config: ModelConfigTypes,
    register: bool = False,
) -> ModelTypes:
    """Get instance of a model based on the configuration present.

    Parameters
    ----------
    model_config : ModelConfigTypes
        The model configuration instance.
    register : bool, optional
        Register the instance for warmstart. Caches the
        instance based on the kwargs, by default False.

    Returns
    -------
    ModelTypes
        The instance.

    Raises
    ------
    ValueError
        If the `config` is unknown.
    """
    # Create and register the instance
    kwargs = {'model_config': model_config}
    if register:
        parsl_registry.register(_factory_fn)
        return parsl_registry.get(_factory_fn, **kwargs)

    return _factory_fn(**kwargs)
