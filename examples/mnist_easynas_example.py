from easynas.genetic_algorithm import EasyNASGA
import torchvision
import torch

if __name__ == '__main__':
    train_data = torchvision.datasets.MNIST('/files/', train=True, download=True)
    test_data = torchvision.datasets.MNIST('/files/', train=False, download=True)
    X_train = train_data.data[:, None, :, :].float()
    X_test = test_data.data[:, None, :, :].float()
    y_train = train_data.targets.float()
    y_test = test_data.targets.float()
    easyga = EasyNASGA(X_train, y_train, X_test, y_test)
    easyga.ga.run()
