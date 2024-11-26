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

    def update_weights(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def train(self, trainloader: DataLoader, epochs: int, device: torch.device):
        print(f"\nTraining model on {device}...")
        self.model.to(device)

        # TODO: Check use of validation loader

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        self.model.train()

        running_loss = 0.0
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in trainloader:
                images = batch["img"]
                labels = batch["label"]

                # images, labels = batch  # Unpack the tuple
                # images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images.to(device), labels.to(device))

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

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

        avg_train_loss = running_loss / len(trainloader)
        return avg_train_loss

    def test(self, testloader: DataLoader, device: torch.device):
        print("\nEvaluating the model on the test set")
        self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.model.eval()

        with torch.no_grad():
            for batch in testloader:
                images, labels = batch  # Unpack the tuple
                images, labels = images.to(self.device), labels.to(self.device)

                # images = batch["img"].to(device)
                # labels = batch["label"].to(device)

                outputs = self.model(images)
                loss += criterion(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss /= len(testloader)
        accuracy = correct / total
        print(f"Test loss: {loss:.4f}, accuracy: {accuracy:.4f}")

        return loss, accuracy
