import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import json
import time
import random
from collections import defaultdict
import logging
from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
from datetime import datetime


class MaliciousClient(FlowerClient):
    def __init__(
        self,
        partition_id: int,
        net: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        malicious_type: MaliciousType,
        attack_intensity: float = 1.0
    ):
        super().__init__(partition_id, net, trainloader, valloader)
        self.malicious_type = malicious_type
        self.attack_intensity = attack_intensity

    def get_malicious_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        if self.malicious_type == MaliciousType.NONE:
            return parameters
            
        elif self.malicious_type == MaliciousType.RANDOM_UPDATES:
            return [np.random.randn(*param.shape) * self.attack_intensity for param in parameters]
            
        elif self.malicious_type == MaliciousType.SCALED_UPDATES:
            return [param * self.attack_intensity for param in parameters]
            
        elif self.malicious_type == MaliciousType.CONSTANT_UPDATES:
            return [np.ones_like(param) * self.attack_intensity for param in parameters]
            
        return parameters

    def fit(self, parameters, config):
        # Original training
        updated_parameters, num_examples, metrics = super().fit(parameters, config)
        
        # Apply malicious modifications
        malicious_parameters = self.get_malicious_parameters(updated_parameters)
        
        return malicious_parameters, num_examples, metrics
