import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
import numpy as np
from collections import OrderedDict
from flwr.common import ndarrays_to_parameters
from app.utils.logger import setup_logger

from flwr.common import parameters_to_ndarrays

class ModelTrainer:
    def __init__(self, model: nn.Module, pretrained_weight_path: str):
        self.logger = setup_logger(self.__class__.__name__)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)

        # Load initial parameters (weights) when the model is initialized
        initial_weights = self.load_initial_parameters(pretrained_weight_path)
        self.update_weights(initial_weights)

    def get_model(self) -> nn.Module:
        return self.model

    def get_weights(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def get_model_parameters(self) -> List[np.ndarray]:
        ndarrays = self.get_weights()
        return ndarrays_to_parameters(ndarrays)

    def update_weights(self, parameters: List[np.ndarray]):
        """Update model weights from parameters."""
        # Convert the Parameters object back to a list of ndarrays if needed
        if isinstance(parameters, list):
            params_dict = zip(self.model.state_dict().keys(), parameters)
        else:
            # Convert Parameters object to ndarray list
            params_dict = zip(self.model.state_dict().keys(), parameters_to_ndarrays(parameters))

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def train(self, trainloader: DataLoader, epochs: int, device: torch.device):
        self.logger.info(f"Training model on {device}...")
        self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        self.model.train()

        running_loss = 0.0
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in trainloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = self.model(images.to(device))

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                epoch_loss += loss.item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            epoch_loss /= len(trainloader)
            epoch_acc = correct / total
            self.logger.info(f"Epoch {epoch + 1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")

        self.logger.info("Training complete")

        avg_train_loss = running_loss / len(trainloader)
        return avg_train_loss

    def test(self, testloader: DataLoader, device: torch.device):
        self.logger.info("Evaluating the model on the test set")
        self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.model.eval()

        with torch.no_grad():
            for batch in testloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                outputs = self.model(images)
                loss += criterion(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss /= len(testloader)
        accuracy = correct / total
        self.logger.info(f"Test loss: {loss:.4f}, accuracy: {accuracy:.4f}")

        return loss, accuracy

    def load_initial_parameters(self, weight_path: str) -> List[np.ndarray]:
        """Load initial parameters from saved model weights."""

        if not os.path.exists(weight_path):
            self.logger.error(f"Weight file not found at {weight_path}. Initializing random weights.")
            return self.get_model_parameters()  # Return random weights

        try:
            self.model.load_state_dict(torch.load(weight_path, map_location="cpu"))
            self.logger.info("Pretrained weights loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading weights: {e}. Initializing random weights.")
            return self.get_model_parameters()  # Return random weights

        return self.get_model_parameters()  # Return loaded parameters
