import random
from pyeasyga import pyeasyga
import logging
from torch import nn
from easynas.model_generation import generate_random_model, random_initialize_layer, calculate_activation_sizes, \
    generate_sequential, get_model_score


class EasyNASGA:
    def __init__(self, X, y, population_size=100, model_n_layers=10):
        self.X = X
        self.y = y
        self.population_size = population_size
        self.model_n_layers = model_n_layers
        self.population = []
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
        self.ga = pyeasyga.GeneticAlgorithm((X, y), generations=5, population_size=5)
        self.ga.create_individual = self.create_individual
        self.ga.fitness_function = self.fitness
        self.ga.mutate_function = self.mutate
        self.ga.crossover_function = self.crossover

    def create_individual(self, data):
        return generate_random_model(self.model_n_layers, self.X.shape[1:], self.available_modules, self.max_n_kernels,
                                     self.max_conv_kernel_size, self.max_conv_stride, self.max_conv_dilation,
                                     self.max_conv_out_channels, self.max_pooling_kernel_size, self.max_pooling_stride)

    def fitness(self, individual, data):
        sequential = generate_sequential(individual, input_shape=self.X.shape[1:], output_size=10)
        return get_model_score(sequential, self.X, self.y, self.X, self.y, self.batch_size)

    def crossover(self, parent_1, parent_2):
        crossover_index = random.randrange(1, len(parent_1))
        valid_1, valid_2 = False, False
        n_attempts = 0
        while not valid_1 and not valid_2:
            child_1 = parent_1[:crossover_index] + parent_2[crossover_index:]
            child_2 = parent_2[:crossover_index] + parent_1[crossover_index:]
            valid_1 = calculate_activation_sizes(child_1, self.X.shape[1:]) != -1
            valid_2 = calculate_activation_sizes(child_2, self.X.shape[1:]) != -1
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
                                            self.X.shape[1:], self.max_n_kernels, self.max_conv_kernel_size,
                                            self.max_conv_stride, self.max_conv_dilation, self.max_conv_out_channels,
                                            self.max_pooling_kernel_size, self.max_pooling_stride)
            valid_model = calculate_activation_sizes(individual, self.X.shape[1:]) != -1
