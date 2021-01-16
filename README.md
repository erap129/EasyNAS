# EasyNAS - a simple and effective CNN architecture generator

## Purpose
Given a dataset, this python package will utilize genetic algorithms and Pytorch to optimize the structure a simple CNN for the task of classification. With simple meaning that the generated architecture is built of a series of layers, where each layers input is the output the previous one. 

## Installation
```python
pip install easynas
```

## Input data format
The input data should be split into training and validation sets, with the following dimensions:  
```math
[#samples, #channels, height, width]
```
This means that 2D image-like data is the expected input. If dealing, for example, with 1D time series data that contains a 'channels' dimension, one should include an extra dimension as such (example with numpy):  
```python
X = X[:, :, :, None]
```

## Usage example
```python
from easynas.genetic_algorithm import EasyNASGA
import torchvision
from sklearn.model_selection import train_test_split

train_data = torchvision.datasets.MNIST('/files/', train=True, download=True)
test_data = torchvision.datasets.MNIST('/files/', train=False, download=True)
X_train = train_data.data[:, None, :, :].float()
y_train = train_data.targets.float()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
easyga = EasyNASGA(X_train, y_train, X_val, y_val, generations=5, population_size=10, max_epochs=1, weight_inheritance=True)
easyga.ga.run()
best_individual = easyga.get_best_individual()
print(f'best individual: {best_individual}')
```

## Credits
Anyone using this package for research/production purposes is requested to cite the following research article:
```markdown
Rapaport, Elad, Oren Shriki, and Rami Puzis.
"EEGNAS: Neural Architecture Search for Electroencephalography Data
Analysis and Decoding." International Workshop on Human Brain and
Artificial Intelligence. Springer, Singapore, 2019.
```
