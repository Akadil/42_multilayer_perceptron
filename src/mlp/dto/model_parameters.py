"""_summary_"""

from dataclasses import dataclass


@dataclass
class ModelParameters:
    learning_rate: float
    epochs: int
    batch_size: int
