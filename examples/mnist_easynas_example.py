from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from easynas.genetic_algorithm import EasyNASGA
import torchvision
import torch

LIMIT_TRAIN_SIZE = 100

if __name__ == '__main__':
    train_data = torchvision.datasets.MNIST('/files/', train=True, download=True)
    test_data = torchvision.datasets.MNIST('/files/', train=False, download=True)
    X_train = train_data.data[:LIMIT_TRAIN_SIZE, None, :, :].float()
    y_train = train_data.targets[:LIMIT_TRAIN_SIZE].float()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    easyga = EasyNASGA(X_train, y_train, X_val, y_val, generations=5, population_size=10, max_epochs=1, weight_inheritance=True)
    easyga.ga.run()
    best_individual = easyga.get_best_individual()
    print(f'best individual: {best_individual}')

    X_test = test_data.data[:, None, :, :].float()
    y_test = test_data.targets.float()
    y_pred = torch.argmax(best_individual(X_test), dim=1).numpy()
    print(f'Test accuracy: {accuracy_score(y_test, y_pred)}')
