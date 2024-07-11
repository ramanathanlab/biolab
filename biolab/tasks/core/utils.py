from typing import Tuple, Sequence

from biolab.api.modeling import Transform
from biolab.modeling.transforms import transform_registry


# TODO: use enums for task inputs, and resolutions to make this mapping
# smoother. This will make the repeitition less error prone.
def find_transformation(
    model_input: str, model_resolution: str, task_resolution: str
) -> Sequence[Transform]:
    """Function to map task input, model resolution, and task resolution to the appropriate
    embedding transformation.

    Parameters
    ----------
    task_input : str
        task input type, must be 'dna', or 'aminoacid'
    model_resolution : str
        resolution of the tokens level representations from teh model
    task_resolution : str
        required hidden representation granularity

    Returns
    -------
    Sequence[Transform]
        Sequence of transformations to be applied iteratively

    Raises
    ------
    ValueError
        If the resolution mapping is not found, or the transform is not found in the registry
    """
    # Map model resolution to task resolution through a transform
    task_transform_mapping = {
        # For each task resolution, map model encoding to transform name
        "dna": {
            "sequence": {
                "bpe": ("average_pool",),
                "char": ("average_pool",),
                "3mer": ("average_pool",),
                "6mer": ("average_pool",),
            },
            "nucleotide": {
                "bpe": ("super_resolution",),
                "char": ("full_sequence",),
                "3mer": ("super_resolution",),
                "6mer": ("super_resolution",),
            },
            "aminoacid": {
                "bpe": ("super_resolution", "3_window"),
                "char": ("3_window",),
                "3mer": ("full_sequence",),
                "6mer": ("super_resolution", "3_window"),
            },
        },
        "aminoacid": {
            "sequence": {
                "bpe": ("average_pool",),
                "char": ("average_pool",),
                "3mer": ("average_pool",),
                "6mer": ("average_pool",),
            },
            "nucleotide": {
                # TODO: this isn't possible, should raise exception elsewhere
            },
            "aminoacid": {
                "bpe": ("super_resolution",),
                "char": ("full_sequence",),
                "3mer": ("super_resolution",),
                "6mer": ("super_resolution",),
            },
        },
    }

    # Retrieve list of transforms from the registry
    transform_names = (
        task_transform_mapping.get(model_input, {})
        .get(task_resolution, {})
        .get(model_resolution, None)
    )

    # Check that we haven't missed a mapping
    if transform_names is None:
        raise ValueError(
            f"Resolution mapping not found for {model_input=}, {task_resolution=}, and {model_resolution=}"
        )

    # Assert that we have all the transforms registered (TODO: this goes away with enums)
    for t_name in transform_names:
        if t_name not in transform_registry:
            raise ValueError(f"Transform {t_name} not found in registry")

    return [transform_registry.get(name) for name in transform_names]