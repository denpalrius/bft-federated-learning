from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from nets.train import ModelTrainer


class MaliciousClient(NumPyClient):
    """Implementation of a malicious client that can perform various attacks."""
    
    def __init__(
        self,
        partition_id: int,
        trainer: ModelTrainer,
        trainloader: DataLoader,
        valloader: DataLoader,
        attack_type: str = "sign_flip",
        attack_intensity: float = 1.0
    ):
        self.partition_id = partition_id
        self.trainer = trainer
        self.trainloader = trainloader
        self.valloader = valloader
        self.attack_type = attack_type
        self.attack_intensity = attack_intensity

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return the model parameters."""
        print(f"[Malicious Client {self.partition_id}] get_parameters")
        return self.trainer.get_parameters()

    def poison_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Apply poisoning attack to parameters."""
        if self.attack_type == "sign_flip":
            # Flip the sign of parameters
            return [-param * self.attack_intensity for param in parameters]
            
        elif self.attack_type == "gaussian_noise":
            # Add Gaussian noise to parameters
            return [param + np.random.normal(0, self.attack_intensity, param.shape) 
                   for param in parameters]
            
        elif self.attack_type == "constant_bias":
            # Add constant bias to parameters
            return [param + self.attack_intensity for param in parameters]
            
        elif self.attack_type == "zero_update":
            # Return zero update
            return [np.zeros_like(param) for param in parameters]
            
        return parameters

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model and return poisoned parameters."""
        print(f"[Malicious Client {self.partition_id}] fit, config: {config}")
        
        self.trainer.set_parameters(parameters)
        self.trainer.train(self.trainloader, epochs=1)
        
        # Get trained parameters and apply poisoning
        updated_parameters = self.trainer.get_parameters()
        poisoned_parameters = self.poison_parameters(updated_parameters)
        
        return poisoned_parameters, len(self.trainloader), {"is_malicious": True}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model and return metrics."""
        print(f"[Malicious Client {self.partition_id}] evaluate, config: {config}")
        
        self.trainer.set_parameters(parameters)
        loss, accuracy = self.trainer.test(self.valloader)
        
        return float(loss), len(self.valloader), {
            "accuracy": float(accuracy),
            "is_malicious": True
        }