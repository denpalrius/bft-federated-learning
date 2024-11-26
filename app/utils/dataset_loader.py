from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFAR10DatasetLoader:
    """Loads and preprocesses the CIFAR-10 dataset for evaluation."""

    def __init__(self, root: str = "./data"):
        self.root = root
        self.transforms = self._default_transforms()

    def _default_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def load_test_data(self, batch_size: int = 32) -> DataLoader:
        test_set = datasets.CIFAR10(
            root=self.root, train=False, download=True, transform=self.transforms
        )

        return DataLoader(test_set, batch_size=batch_size, shuffle=False)
