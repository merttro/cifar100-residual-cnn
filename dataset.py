import torch
import torchvision
from torch.utils.data import DataLoader,Subset


#print(torch.cuda.is_available())
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

#Dataset Import

def get_dataloaders(batch_size=32):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    ])

    # Train dataset (augmentation açık)
    trainset = torchvision.datasets.CIFAR100(
        "./data",
        train=True,
        transform=train_transform,
        download=True
    )

    # Validation dataset (augmentation kapalı)
    valset = torchvision.datasets.CIFAR100(
        "./data",
        train=True,
        transform=test_transform,
        download=True
    )

    # Test dataset
    testset = torchvision.datasets.CIFAR100(
        "./data",
        train=False,
        transform=test_transform,
        download=True
    )

    # Train / Validation indekslerini oluştur
    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        len(trainset),
        generator=generator
    ).tolist()

    print(type(valset[0]))
    image , label = valset[0]
    print(image,label)

    train_indices = indices[:45000]
    val_indices = indices[45000:]

    # Ayrı dataset parçaları
    trainset = Subset(trainset, train_indices)
    valset = Subset(valset, val_indices)

    # DataLoaders
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )

    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader

