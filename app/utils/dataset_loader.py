from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.utils.logger import setup_logger

# Global FederatedDataset object
fds = None

logger = setup_logger("CIFAR10DatasetLoader")

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),  # Data augmentation: Random cropping
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


def apply_transforms_train(batch, transform_train):
    batch["img"] = [transform_train(img) for img in batch["img"]]
    return batch


def apply_transforms_test(batch, transform_test):
    batch["img"] = [transform_test(img) for img in batch["img"]]
    return batch


def load_data_partition(num_partitions, partition_id, batch_size):
    global fds

    # logger.info(f"Number of Partitions: {num_partitions}")

    # Initialize the partitioner and assign it to the FederatedDataset object
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id, "train")
    logger.info(f"Loaded partition {partition_id} for training")

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(
        lambda batch: apply_transforms_train(batch, transform_train)
    )

    # logger.info(f"Total number of partition_train_test: {partition_train_test}")

    train_loader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return train_loader, val_loader


def load_test_data(batch_size):
    global fds
    if fds is None:
        fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={})

    centralized_test = fds.load_split("test")
    centralized_test = centralized_test.with_transform(
        lambda batch: apply_transforms_test(batch, transform_test)
    )

    test_loader = DataLoader(centralized_test, batch_size=batch_size)

    # Print total number of samples
    total_samples = len(centralized_test)
    # logger.info(f"Total number of test samples: {total_samples}")

    return test_loader
