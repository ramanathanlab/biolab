from __future__ import annotations  # noqa D100

from .average_pool import AveragePool
from .full_sequence import FullSequence
from .super_resolution import SuperResolution
from .window import Window3

transform_registry = {
    AveragePool.name: AveragePool,
    SuperResolution.name: SuperResolution,
    FullSequence.name: FullSequence,
    Window3.name: Window3,
}
