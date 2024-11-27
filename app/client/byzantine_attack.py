import numpy as np
from typing import List
from app.client.byzantine_strategy import ByzantineStrategy


class ByzantineAttack:
    def __init__(self, attack_type: ByzantineStrategy, attack_intensity: float):
        self.attack_type = attack_type
        self.attack_intensity = attack_intensity

    def poison_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Apply poisoning attack to parameters."""
        if self.attack_type == ByzantineStrategy.SIGN_FLIP:
            # Flip the sign of parameters
            return [-param * self.attack_intensity for param in parameters]

        elif self.attack_type == ByzantineStrategy.GAUSSIAN_NOISE:
            # Add Gaussian noise to parameters
            return [
                param + np.random.normal(0, self.attack_intensity, param.shape)
                for param in parameters
            ]

        elif self.attack_type == ByzantineStrategy.CONSTANT_BIAS:
            # Add constant bias to parameters
            return [param + self.attack_intensity for param in parameters]

        elif self.attack_type == ByzantineStrategy.ZERO_UPDATE:
            # Return zero update
            return [np.zeros_like(param) for param in parameters]

        return parameters
