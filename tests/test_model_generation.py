import unittest
from copy import deepcopy

from sklearn.datasets import make_classification
from easynas import model_generation
from easynas.genetic_algorithm import EasyNASGA
from easynas.model_generation import generate_sequential, get_model_score


class ModelGenerationTests(unittest.TestCase):
    def test_simple_model_gen(self):
        model = model_generation.generate_random_model(n_layers=10, input_shape=(100, 100))
        self.assertEqual(len(model), 10)

    def test_gen_population(self):
        population = []
        for i in range(100):
            population.append(model_generation.generate_random_model(n_layers=10, input_shape=(100, 100)))
            self.assertEqual(len(population[-1]), 10)
        self.assertEqual(len(population), 100)

    def test_crossover(self):
        X, y = make_classification(100)
        en = EasyNASGA(X, y)
        child_1, child_2 = en.crossover(en.population[0], en.population[1])
        self.assertEqual(len(child_1), 10)
        self.assertEqual(len(child_2), 10)
        # TODO - check validity of each of the possible crossover outcomes

    def test_mutation(self):
        X, y = make_classification(100)
        en = EasyNASGA(X, y)
        orig_individual = deepcopy(en.population[0])
        en.mutate(en.population[0])
        self.assertNotEqual(orig_individual, en.population[0])

    def test_sequential_generation(self):
        input_shape = [3, 100, 100]
        layer_list = model_generation.generate_random_model(n_layers=10, input_shape=input_shape)
        sequential = generate_sequential(layer_list, input_shape=input_shape, output_size=10)
        self.assertEqual(len(sequential), len(layer_list) + 2)

    def test_score_model(self):
        input_shape = [1, 20]
        layer_list = model_generation.generate_random_model(n_layers=10, input_shape=input_shape)
        X, y = make_classification(100)
        X = X[:, None, :, None]
        sequential = generate_sequential(layer_list, input_shape=input_shape, output_size=10)
        print(get_model_score(sequential, X[:50], y[:50], X[50:], y[50:]))

    def test_run_easynas(self):
        X, y = make_classification(100)
        X = X[:, None, :, None]
        ea = EasyNASGA(X, y)
        ea.ga.run()
        print(ea.ga.best_individual())


if __name__ == '__main__':
    unittest.main()
