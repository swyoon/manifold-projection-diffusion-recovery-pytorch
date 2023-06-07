from loader.modified_dataset import MNIST_OOD, FashionMNIST_OOD, CIFAR10_OOD, CIFAR100_OOD, SVHN_OOD


root = 'datasets'
for split in ['training', 'validation', 'evaluation']:
    MNIST_OOD(root=root, download=True)
    FashionMNIST_OOD(root=root, download=True)
    CIFAR10_OOD(root=root, download=True)
    CIFAR100_OOD(root=root, download=True)
    SVHN_OOD(root=root, download=True)

