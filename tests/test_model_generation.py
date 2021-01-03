import unittest
from copy import deepcopy
from torch import nn
from sklearn.datasets import make_classification
from easynas import model_generation
from easynas.genetic_algorithm import EasyNASGA
from easynas.model_generation import generate_sequential, get_model_score, sequential_to_layer_list, init_conv_layer

# test constants
MAX_N_KERNELS = 2
MAX_CONV_KERNEL_SIZE = (20, 20)
MAX_CONV_STRIDE = (3, 3)
MAX_CONV_DILATION = (1, 1)
MAX_CONV_OUT_CHANNELS = 50
MAX_POOLING_KERNEL_SIZE = (3, 3)
MAX_POOLING_STRIDE = (3, 3)
AVAILABLE_MODULES = [nn.Conv2d, nn.Dropout, nn.MaxPool2d, nn.Identity, nn.ReLU]


class ModelGenerationTests(unittest.TestCase):
    def test_simple_model_gen(self):
        model = model_generation.generate_random_model(10, (100, 100), AVAILABLE_MODULES, MAX_N_KERNELS,
                                                       MAX_CONV_KERNEL_SIZE, MAX_CONV_STRIDE, MAX_CONV_DILATION,
                                                       MAX_CONV_OUT_CHANNELS, MAX_POOLING_KERNEL_SIZE,
                                                       MAX_POOLING_STRIDE)
        self.assertEqual(len(model), 10)

    def test_gen_population(self):
        population = []
        for i in range(100):
            population.append(model_generation.generate_random_model(10, (100, 100), AVAILABLE_MODULES, MAX_N_KERNELS,
                                                        MAX_CONV_KERNEL_SIZE, MAX_CONV_STRIDE, MAX_CONV_DILATION,
                                                        MAX_CONV_OUT_CHANNELS, MAX_POOLING_KERNEL_SIZE,
                                                        MAX_POOLING_STRIDE))
            self.assertEqual(len(population[-1]), 10)
        self.assertEqual(len(population), 100)

    def test_weight_inheritance(self):
        X, y = make_classification(100, n_features=100)
        X = X[:, None, :, None]
        ea = EasyNASGA(X, y, population_size=2, generations=1, weight_inheritance=True)
        input_shape = [1, 100, 1]
        layer_list_1 = [init_conv_layer(input_shape, n_kernels=2, conv_kernel_size=(10, 1), conv_stride=(1, 1),
                                        conv_dilation=(1, 1), conv_out_channels=10, random_sizes=False),
                        init_conv_layer(input_shape, n_kernels=2, conv_kernel_size=(10, 1), conv_stride=(1, 1),
                                        conv_dilation=(1, 1), conv_out_channels=10, random_sizes=False),
                        init_conv_layer(input_shape, n_kernels=2, conv_kernel_size=(10, 1), conv_stride=(1, 1),
                                        conv_dilation=(1, 1), conv_out_channels=10, random_sizes=False)]
        layer_list_2 = [init_conv_layer(input_shape, n_kernels=2, conv_kernel_size=(20, 1), conv_stride=(1, 1),
                                        conv_dilation=(1, 1), conv_out_channels=10, random_sizes=False),
                        init_conv_layer(input_shape, n_kernels=2, conv_kernel_size=(20, 1), conv_stride=(1, 1),
                                        conv_dilation=(1, 1), conv_out_channels=10, random_sizes=False),
                        init_conv_layer(input_shape, n_kernels=2, conv_kernel_size=(20, 1), conv_stride=(1, 1),
                                        conv_dilation=(1, 1), conv_out_channels=10, random_sizes=False)]
        seq_1 = generate_sequential(layer_list_1, input_shape=input_shape, output_size=10)
        seq_2 = generate_sequential(layer_list_2, input_shape=input_shape, output_size=10)
        cut_point = 1
        child_1, child_2 = ea.crossover(seq_1, seq_2, cut_point)
        seq_1_child = generate_sequential(child_1, input_shape=input_shape, output_size=10, append_flatten=False)
        seq_2_child = generate_sequential(child_2, input_shape=input_shape, output_size=10, append_flatten=False)
        assert (seq_1_child._modules['0'].weight == seq_1._modules['0'].weight).all()
        assert (seq_1_child._modules['2'].weight == seq_2._modules['2'].weight).all()
        assert (seq_2_child._modules['0'].weight == seq_2._modules['0'].weight).all()
        assert (seq_2_child._modules['2'].weight == seq_1._modules['2'].weight).all()

    def test_crossover(self):
        X, y = make_classification(100)
        X = X[:, None, :, None]
        ea = EasyNASGA(X, y, population_size=2, generations=1, weight_inheritance=False)
        ea.ga.create_first_generation()
        child_1, child_2 = ea.crossover(ea.ga.current_generation[0].genes, ea.ga.current_generation[1].genes)
        self.assertEqual(len(child_1), 12)
        self.assertEqual(len(child_2), 12)
        # TODO - check validity of each of the possible crossover outcomes

    def test_crossover_weight_inheritance(self):
        X, y = make_classification(100, n_features=500)
        X = X[:, None, :, None]
        ea = EasyNASGA(X, y, population_size=2, generations=1, model_n_layers=20, weight_inheritance=True)
        ea.ga.create_first_generation()
        child_1, child_2 = ea.crossover(ea.ga.current_generation[0].genes, ea.ga.current_generation[1].genes)
        self.assertEqual(len(child_1), 22)
        self.assertEqual(len(child_2), 22)
        # TODO - check validity of each of the possible crossover outcomes

    def test_mutation(self):
        X, y = make_classification(100)
        X = X[:, None, :, None]
        ea = EasyNASGA(X, y, population_size=2, generations=1)
        ea.ga.create_first_generation()
        orig_individual = deepcopy(ea.ga.current_generation[0].genes)
        ea.mutate(ea.ga.current_generation[0].genes)
        self.assertNotEqual(orig_individual, ea.ga.current_generation[0].genes)

    def test_sequential_generation(self):
        input_shape = [3, 100, 100]
        layer_list = model_generation.generate_random_model(10, (100, 100), AVAILABLE_MODULES, MAX_N_KERNELS,
                                                        MAX_CONV_KERNEL_SIZE, MAX_CONV_STRIDE, MAX_CONV_DILATION,
                                                        MAX_CONV_OUT_CHANNELS, MAX_POOLING_KERNEL_SIZE,
                                                        MAX_POOLING_STRIDE)
        sequential = generate_sequential(layer_list, input_shape=input_shape, output_size=10)
        self.assertEqual(len(sequential), len(layer_list) + 2)

    def test_sequential_to_layer_list(self):
        input_shape = [3, 100, 100]
        layer_list = model_generation.generate_random_model(30, (100, 100), AVAILABLE_MODULES, MAX_N_KERNELS,
                                                        MAX_CONV_KERNEL_SIZE, MAX_CONV_STRIDE, MAX_CONV_DILATION,
                                                        MAX_CONV_OUT_CHANNELS, MAX_POOLING_KERNEL_SIZE,
                                                        MAX_POOLING_STRIDE)
        sequential = generate_sequential(layer_list, input_shape=input_shape, output_size=10)
        new_layer_list = sequential_to_layer_list(sequential)
        for new_layer, layer in zip(new_layer_list, layer_list):
            self.assertEqual(type(new_layer), type(layer))

    def test_score_model(self):
        input_shape = [1, 20]
        layer_list = model_generation.generate_random_model(10, input_shape, AVAILABLE_MODULES, MAX_N_KERNELS,
                                                        MAX_CONV_KERNEL_SIZE, MAX_CONV_STRIDE, MAX_CONV_DILATION,
                                                        MAX_CONV_OUT_CHANNELS, MAX_POOLING_KERNEL_SIZE,
                                                        MAX_POOLING_STRIDE)
        X, y = make_classification(100)
        X = X[:, None, :, None]
        sequential = generate_sequential(layer_list, input_shape=input_shape, output_size=10)
        print(get_model_score(sequential, X[:50], y[:50], X[50:], y[50:], 32, 1))

    def test_run_easynas(self):
        X, y = make_classification(100)
        X = X[:, None, :, None]
        ea = EasyNASGA(X, y, population_size=5, generations=3)
        ea.ga.run()
        print(ea.ga.best_individual())


if __name__ == '__main__':
    unittest.main()
