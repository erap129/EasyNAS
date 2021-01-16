# EasyNAS - a simple and effective CNN architecture generator

## Purpose
Given a dataset, this python package will utilize genetic algorithms and Pytorch to optimize the structure a simple CNN for the task of classification. With simple meaning that the generated architecture is built of a series of layers, where each layers input is the output the previous one. 

## Installation
```python
pip install easynas
```

## Usage example
```python
from easynas.genetic_algorithm import EasyNASGA
easyga = EasyNASGA(X_train, y_train, X_val, y_val, generations=5, population_size=10, max_epochs=1, weight_inheritance=True)
easyga.ga.run()
best_individual = easyga.get_best_individual()
print(f'best individual: {best_individual}')
```

## Credits
Anyone using this package for research/production purposes is requested to cite the following research article:
Rapaport, Elad, Oren Shriki, and Rami Puzis. "EEGNAS: Neural Architecture Search for Electroencephalography Data Analysis and Decoding." International Workshop on Human Brain and Artificial Intelligence. Springer, Singapore, 2019.