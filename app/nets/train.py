import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
import numpy as np
from collections import OrderedDict
from flwr.common import ndarrays_to_parameters


class ModelTrainer:
    def __init__(self, model: nn.Module):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)

    def get_model(self) -> nn.Module:
        return self.model

    def get_weights(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def get_model_parameters(self) -> List[np.ndarray]:
        ndarrays = self.get_weights()
        return ndarrays_to_parameters(ndarrays)

    def set_weights(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def train(self, trainloader: DataLoader, epochs: int):
        print(f"\nTraining model on {self.device}...")

        # TODO: Check use of validation loader

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        self.model.train()

        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in trainloader:
                images, labels = batch["img"], batch["label"]
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Metrics
                epoch_loss += loss.item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            epoch_loss /= len(trainloader)
            epoch_acc = correct / total
            print(
                f"Epoch {epoch + 1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}"
            )

        print("Training complete")

    def test(self, testloader: DataLoader):
        print("\nEvaluating the model on the test set")

        criterion = nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.model.eval()

        with torch.no_grad():
            for batch in testloader:
                images, labels = batch["img"], batch["label"]
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss /= len(testloader)
        accuracy = correct / total
        print(f"Test loss: {loss:.4f}, accuracy: {accuracy:.4f}")

        return loss, accuracy
