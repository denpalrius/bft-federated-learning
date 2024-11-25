from torch.utils.data import DataLoader
from torchvision import transforms
from flwr_datasets import FederatedDataset
from config_base import BaseConfig


# - Load CIFAR-10 training and test sets.
# - Partition into ten smaller datasets (training and validation sets).
# - Wrap each partition in its own `DataLoader`.
# - Introduce `num_partitions` parameter for flexible partitioning.
# - Use `load_datasets` with different partition numbers.

class DatasetLoader:
    def __init__(self, config: BaseConfig):
        self.dataset = config.dataset
        self.num_partitions = config.num_partitions
        self.batch_size = config.batch_size
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def _apply_transforms(self, batch):
        batch["img"] = [self.transforms(img) for img in batch["img"]]
        return batch

    def load_partition(self, partition_id: int):
        fds = FederatedDataset(dataset=self.dataset, partitioners={"train": self.num_partitions})
        partition = fds.load_partition(partition_id)

        # Split partition into train and test
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        partition_train_test = partition_train_test.with_transform(self._apply_transforms)

        trainloader = DataLoader(
            partition_train_test["train"], batch_size=self.batch_size, shuffle=True
        )
        valloader = DataLoader(partition_train_test["test"], batch_size=self.batch_size)

        return trainloader, valloader

    def load_test_set(self):
        fds = FederatedDataset(dataset=self.dataset, partitioners={"train": self.num_partitions})
        testset = fds.load_split("test").with_transform(self._apply_transforms)
        testloader = DataLoader(testset, batch_size=self.batch_size)
        return testloader

    def load_datasets(self, partition_id: int):
        """Load train, validation, and test datasets."""
        trainloader, valloader = self.load_partition(partition_id)
        testloader = self.load_test_set()
        return trainloader, valloader, testloader


if __name__ == "__main__":
    config = BaseConfig()
    
    loader = DatasetLoader(config=config)
    trainloader, valloader, testloader = loader.load_datasets(0)
    
    print(f"Train: {len(trainloader.dataset)} samples")
    print(f"Validation: {len(valloader.dataset)} samples")
    print(f"Test: {len(testloader.dataset)} samples")
