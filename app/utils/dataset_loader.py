from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.utils.logger import setup_logger


class CIFAR10DatasetLoader:
    def __init__(self, num_partitions, batch_size):
        self.logger = setup_logger(self.__class__.__name__)

        self.root = "./data/cifar10"
        self.num_partitions = num_partitions
        self.batch_size = batch_size
        self.fds = None
        self.partitioners = None
        self.transforms = self._default_transforms()

    def initialize_fds(self):
        partitioner = IidPartitioner(num_partitions=self.num_partitions)
        self.fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    def _default_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def apply_transforms(self, batch):
        batch["img"] = [self.transforms(img) for img in batch["img"]]
        return batch

    def load_data_partition(self, partition_id: int):
        if not self.fds:
            raise ValueError(
                "FederatedDataset object not initialized. Call `initialize_fds()` first."
            )
        partition = self.fds.load_partition(partition_id, "train")
        self.logger.info(f"Loaded partition {partition_id} for training")

        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        partition_train_test = partition_train_test.with_transform(
            self.apply_transforms
        )

        trainloader = DataLoader(
            partition_train_test["train"], batch_size=32, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=32)
        return trainloader, testloader

    def load_test_data(self):
        centralized_test = self.fds.load_split("test")
        centralized_test = centralized_test.with_transform(self.apply_transforms)

        return DataLoader(centralized_test, batch_size=self.batch_size)
