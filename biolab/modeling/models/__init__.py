"""Submodule for LM model instances."""

from __future__ import annotations

from typing import Union

from .ankh import ankh_models
from .calm import calm_models
from .dnabert import dnabert_models
from .esm import esm_models
from .evo import evo_models
from .genalm import genalm_models
from .genomelm import genomelm_models
from .genslm import genslm_models
from .genslm_esm import genslmesm_models
from .nucleotide_transformer import nucleotidetransformer_models
from .protgpt2 import protgpt2_models
from .protrans import protrans_models
from .genslm2 import genslm2_models

model_registry = {
    **ankh_models,
    **calm_models,
    **dnabert_models,
    **esm_models,
    **evo_models,
    **genalm_models,
    **genomelm_models,
    **genslmesm_models,
    **genslm_models,
    **nucleotidetransformer_models,
    **protgpt2_models,
    **protrans_models,
    **genslm2_models,
}

ModelConfigTypes = Union[*model_registry.keys(),]
