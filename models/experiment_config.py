from dataclasses import dataclass
from typing import List

from models.maliciopus_type import MaliciousType


@dataclass
class ExperimentConfig:
    num_rounds: int
    num_clients: int
    num_malicious: int
    malicious_types: List[MaliciousType]
    use_bft: bool
    experiment_name: str