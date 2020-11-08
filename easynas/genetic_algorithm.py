import random
from pyeasyga import pyeasyga
import logging
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm, trange
import os
import numpy as np

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)
from easynas.model_generation import generate_random_model, random_initialize_layer, calculate_activation_sizes, \
    generate_sequential, get_model_score


class NASGeneticAlgorithm(pyeasyga.GeneticAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        """Run (solve) the Genetic Algorithm."""
        for i in range(self.generations):
            log.info(f'Training population in generation {i + 1}...')
            if i == 0:
                self.create_first_generation()
            else:
                self.create_next_generation()
            best_sequential = generate_sequential(self.best_individual()[1], input_shape=self.X_train.shape[1:],
                                                  output_size=len(np.unique(self.y_train)))
            log.info(f'best individual: {best_sequential}')
            log.info(f'best individual score: {self.best_individual()[0]}')

    def calculate_population_fitness(self):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        """
        for individual in tqdm(self.current_generation):
            individual.fitness = self.fitness_function(
                individual.genes, self.seed_data)


class EasyNASGA:
    def __init__(self, X_train, y_train, X_val=None, y_val=None, validation_amount=0.2, population_size=100,
                 generations=100, model_n_layers=10, max_epochs=5, individual_progress_bars=False):
        if X_val is not None and y_val is not None:
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
        else:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train, stratify=y_train,
                                                                                  test_size=validation_amount)
        self.population_size = population_size
        self.model_n_layers = model_n_layers
        self.max_epochs = max_epochs
        self.individual_progress_bars = individual_progress_bars
        # general nn parameters
        self.batch_size = 32
        self.max_n_kernels = 2
        # convolution layers
        self.max_conv_kernel_size = (20, 20)
        self.max_conv_stride = (3, 3)
        self.max_conv_dilation = (1, 1)
        self.max_conv_out_channels = 50
        # max-pooling layers
        self.max_pooling_kernel_size = (3, 3)
        self.max_pooling_stride = (3, 3)
        # genetic algorithm parameters
        self.max_crossover_attempts = 50
        self.available_modules = [nn.Conv2d, nn.Dropout, nn.MaxPool2d, nn.Identity, nn.ReLU]
        # self.initialize_population()
        self.ga = NASGeneticAlgorithm((X_train, y_train), generations=generations, population_size=population_size)
        self.ga.create_individual = self.create_individual
        self.ga.fitness_function = self.fitness
        self.ga.mutate_function = self.mutate
        self.ga.crossover_function = self.crossover

    def create_individual(self, data):
        return generate_random_model(self.model_n_layers, self.X_train.shape[1:], self.available_modules,
                                     self.max_n_kernels,
                                     self.max_conv_kernel_size, self.max_conv_stride, self.max_conv_dilation,
                                     self.max_conv_out_channels, self.max_pooling_kernel_size, self.max_pooling_stride)

    def fitness(self, individual, data):
        sequential = generate_sequential(individual, input_shape=self.X_train.shape[1:],
                                         output_size=len(np.unique(self.y_train)))
        return get_model_score(sequential, self.X_train, self.y_train, self.X_val, self.y_val, self.batch_size,
                               self.max_epochs, progress_bar=self.individual_progress_bars)

    def crossover(self, parent_1, parent_2):
        crossover_index = random.randrange(1, len(parent_1))
        valid_1, valid_2 = False, False
        n_attempts = 0
        while not valid_1 and not valid_2:
            child_1 = parent_1[:crossover_index] + parent_2[crossover_index:]
            child_2 = parent_2[:crossover_index] + parent_1[crossover_index:]
            valid_1 = calculate_activation_sizes(child_1, self.X_train.shape[1:]) != -1
            valid_2 = calculate_activation_sizes(child_2, self.X_train.shape[1:]) != -1
            if n_attempts > self.max_crossover_attempts:
                break
        if valid_1 and valid_2:
            return child_1, child_2
        elif valid_1 and not valid_2:
            return child_1, parent_2
        elif valid_2 and not valid_1:
            return parent_1, child_2
        else:
            logging.debug(f'Failed crossover attempt after {self.max_crossover_attempts} tries')
            return parent_1, parent_2

    def mutate(self, individual):
        mutate_index = random.randrange(len(individual))
        valid_model = False
        while not valid_model:  # this will eventually work so infinite loop is OK
            individual[mutate_index] = random_initialize_layer(random.choice(self.available_modules),
                                                               self.X_train.shape[1:], self.max_n_kernels,
                                                               self.max_conv_kernel_size,
                                                               self.max_conv_stride, self.max_conv_dilation,
                                                               self.max_conv_out_channels,
                                                               self.max_pooling_kernel_size, self.max_pooling_stride)
            valid_model = calculate_activation_sizes(individual, self.X_train.shape[1:]) != -1

    def get_best_individual(self):
        if len(self.ga.current_generation) == 0:
            raise Exception('Cannot return best individual before running the GA')
        return generate_sequential(self.ga.best_individual()[1], self.X_train.shape[1:], len(np.unique(self.y_train)))
