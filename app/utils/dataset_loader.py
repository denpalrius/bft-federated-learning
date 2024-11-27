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
        self.partitioners = None
        self.transform_train, self.transform_test = self._default_transforms()

        # Initialize the FederatedDataset object
        partitioner = IidPartitioner(num_partitions=self.num_partitions)
        self.fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    def _default_transforms(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(
                    32, padding=4
                ),  # Data augmentation: Random cropping
                transforms.RandomHorizontalFlip(),  # Data augmentation: Random horizontal flip
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),  # Normalization
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),  # Normalization
            ]
        )

        return transform_train, transform_test

    def apply_transforms_train(self, batch):
        batch["img"] = [self.transform_train(img) for img in batch["img"]]
        return batch

    def apply_transforms_test(self, batch):
        batch["img"] = [self.transform_test(img) for img in batch["img"]]
        return batch

    def load_data_partition(self, partition_id: int):
        partition = self.fds.load_partition(partition_id, "train")
        self.logger.info(f"Loaded partition {partition_id} for training")

        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        partition_train_test = partition_train_test.with_transform(self.apply_transforms_train)

        # TODO: Use of validation loader

        trainloader = DataLoader(
            partition_train_test["train"], batch_size=self.batch_size, shuffle=True
        )
        testloader = DataLoader(
            partition_train_test["test"], batch_size=self.batch_size
        )
        return trainloader, testloader

    def load_test_data(self):
        centralized_test = self.fds.load_split("test")
        centralized_test = centralized_test.with_transform(self.apply_transforms_test)
        
        return DataLoader(centralized_test, batch_size=self.batch_size)
