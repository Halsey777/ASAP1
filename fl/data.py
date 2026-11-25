"""Data loading and IID client partitioning for the ASAP protocol.

Reference-model selection uses only validation data held out from malicious
clients' own local partitions. No global/server validation split is created.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


class TinyValidation(Dataset):
    """Read official flat Tiny-ImageNet validation layout without rearranging it."""
    def __init__(self, root, class_to_idx, transform):
        self.root, self.transform = Path(root), transform
        self.samples = []
        lines = (self.root / "val_annotations.txt").read_text(encoding="utf-8").splitlines()
        seen = set()
        for line in lines:
            filename, label, *_ = line.split()
            if filename in seen or label not in class_to_idx:
                raise ValueError("Duplicate validation image or unknown Tiny-ImageNet class")
            seen.add(filename)
            path = self.root / "images" / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            self.samples.append((path, class_to_idx[label]))
        if not self.samples:
            raise ValueError("Empty Tiny-ImageNet validation set")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image
        path, label = self.samples[index]
        with Image.open(path) as image:
            x = self.transform(image.convert("RGB"))
        return x, label


def load_data(name, root, *, download, seed):
    if name == "synthetic":
        generator = torch.Generator().manual_seed(seed)

        def sample(size):
            x = torch.randn(size, 4, generator=generator)
            y = (x[:, 0] + .5 * x[:, 1] > 0).long()
            return TensorDataset(x, y)

        train, test = sample(200), sample(80)
        return train, test, train.tensors[1].numpy(), 2

    from torchvision import datasets, transforms

    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.1307,), (.3081,)),
        ])
        train = datasets.MNIST(root, train=True, transform=transform, download=download)
        test = datasets.MNIST(root, train=False, transform=transform, download=download)
        return train, test, train.targets.numpy(), 10

    if name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.4914, .4822, .4465], [.2023, .1994, .2010]),
        ])
        train = datasets.CIFAR10(root, train=True, transform=transform, download=download)
        test = datasets.CIFAR10(root, train=False, transform=transform, download=download)
        return train, test, np.asarray(train.targets), 10

    root = Path(root)
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    train = datasets.ImageFolder(root / "train", transform=transform)
    if len(train.classes) != 200:
        raise ValueError("Tiny-ImageNet training data must contain 200 classes")

    validation = root / "val"
    if (validation / "val_annotations.txt").is_file():
        test = TinyValidation(validation, train.class_to_idx, transform)
    else:
        test = datasets.ImageFolder(validation, transform=transform)
        if test.class_to_idx != train.class_to_idx:
            raise ValueError("Tiny-ImageNet train/val class mappings differ")

    return train, test, np.asarray(train.targets), 200


def split_and_partition(labels, *, num_clients, seed):
    """Assign every training example to exactly one IID client partition."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("Expected one label per training example")
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if len(labels) < num_clients:
        raise ValueError("Fewer training examples than clients")

    rng = np.random.default_rng(seed)
    partitions = [
        part.tolist()
        for part in np.array_split(rng.permutation(len(labels)), num_clients)
    ]
    return partitions


def split_partition_and_root(labels, *, num_clients, server_root_samples, seed):
    """Reserve a trusted server root set, then IID-partition all remaining data.

    This function is intended for FLTrust. The trusted root samples are disjoint
    from all client partitions. No central validation set is created.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("Expected one label per training example")
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if type(server_root_samples) is not int or server_root_samples < 0:
        raise ValueError("server_root_samples must be a nonnegative integer")

    if server_root_samples == 0:
        return split_and_partition(labels, num_clients=num_clients, seed=seed), []

    classes = list(np.unique(labels))
    if server_root_samples < len(classes):
        raise ValueError("server_root_samples must cover every class at least once")
    if server_root_samples > len(labels) - num_clients:
        raise ValueError("Trusted root reservation leaves fewer examples than clients")

    rng = np.random.default_rng(seed)
    by_class = {
        label: rng.permutation(np.flatnonzero(labels == label)).tolist()
        for label in classes
    }

    root = []
    offsets = {label: 0 for label in classes}
    while len(root) < server_root_samples:
        progressed = False
        for label in classes:
            if len(root) >= server_root_samples:
                break
            offset = offsets[label]
            values = by_class[label]
            if offset < len(values):
                root.append(values[offset])
                offsets[label] = offset + 1
                progressed = True
        if not progressed:
            raise RuntimeError("Unable to allocate requested trusted root samples")

    root_set = set(root)
    client_indices = [idx for idx in range(len(labels)) if idx not in root_set]
    partitions = [
        part.tolist()
        for part in np.array_split(rng.permutation(client_indices), num_clients)
    ]
    return partitions, sorted(root)


def split_attacker_validation(partitions, malicious_clients, *, validation_fraction, seed):
    """Hold out validation data only from attacker-controlled client partitions.

    Every malicious client contributes the requested fraction of its own local
    partition. The attacker evaluates the broadcast global model on the union of
    these held-out samples. Benign-client data is never used for reference-model
    selection.

    Returns:
        train_partitions: client partitions with malicious validation samples removed
        validation_ids: sorted union of attacker-owned validation samples
        validation_by_client: per-malicious-client validation indices
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")

    train_partitions = [list(part) for part in partitions]
    malicious_clients = sorted(set(int(client) for client in malicious_clients))
    if not malicious_clients:
        raise ValueError("At least one malicious client is required")

    validation_by_client = {}
    validation_ids = []

    for client in malicious_clients:
        if client < 0 or client >= len(train_partitions):
            raise ValueError(f"Invalid malicious client id: {client}")

        local = np.asarray(train_partitions[client], dtype=np.int64)
        if len(local) < 2:
            raise ValueError(
                f"Malicious client {client} needs at least two local samples "
                "to create train/validation splits"
            )

        rng = np.random.default_rng(seed + 104729 * (client + 1))
        shuffled = rng.permutation(local)
        validation_count = max(1, min(len(local) - 1, int(len(local) * validation_fraction)))

        validation = sorted(shuffled[:validation_count].tolist())
        train = shuffled[validation_count:].tolist()

        train_partitions[client] = train
        validation_by_client[client] = validation
        validation_ids.extend(validation)

    return train_partitions, sorted(validation_ids), validation_by_client
