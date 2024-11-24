import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Prepare MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# Split data into federated clients (simulation)
def federated_data_split(dataset, num_clients=5):
    data_size = len(dataset)
    client_data = []
    indices = np.random.permutation(data_size)
    split_size = data_size // num_clients
    
    for i in range(num_clients):
        start = i * split_size
        end = (i + 1) * split_size if i != num_clients - 1 else data_size
        client_data.append([dataset[i] for i in indices[start:end]])

    return client_data

# Simulate federated learning clients
federated_data = federated_data_split(trainset)

# Client training function
def train_client(model, data_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Federated aggregation function (averaging model weights)
def federated_aggregate(models):
    # Assume equal weighting for each model
    state_dicts = [model.state_dict() for model in models]
    avg_state_dict = state_dicts[0]

    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.mean(torch.stack([state_dict[key].float() for state_dict in state_dicts]), dim=0)

    # Update the global model with the averaged state dict
    return avg_state_dict
