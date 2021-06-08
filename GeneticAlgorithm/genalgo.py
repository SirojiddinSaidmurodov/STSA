import random
from inspect import signature

import matplotlib.pyplot as plt
import numpy as np


class Phenotype:
    def __init__(self, genotype: list):
        self.genotype = genotype

    def __repr__(self):
        return str(self.genotype)


class GenAlgo:
    def __init__(self, objective_function: callable, bounds: tuple[float, float], population_size: int,
                 max_generations: int, mutation_probability: float, log: bool = False):
        self.fitness = objective_function
        self.bounds = bounds
        self.POPULATION_SIZE = population_size
        self.MAX_GENERATIONS = max_generations
        self.PARAM_SIZE = len(signature(self.fitness).parameters)
        self.population = [Phenotype([random.randint(int(bounds[0]), int(bounds[1])) for _ in range(self.PARAM_SIZE)])
                           for _ in range(self.POPULATION_SIZE)]
        self.MUTATION_PROBABILITY = mutation_probability
        self.log = log

    def run(self):
        maxFitness, minFitness, meanFitness, generations = [], [], [], []
        for generation in range(self.MAX_GENERATIONS):
            fitness = [self.fitness(*phenotype.genotype) for phenotype in self.population]

            if self.log:
                maxFitness.append(max(fitness))
                minFitness.append(min(fitness))
                meanFitness.append(sum(fitness) / len(fitness))
                generations.append(self.population)

            mating_pool = self.__selection__(fitness)
            elite_size = 1 if int(self.POPULATION_SIZE * 0.1) == 0 else int(self.POPULATION_SIZE * 0.1)
            elite = sorted(self.population,
                           key=lambda phenotype: self.fitness(*phenotype.genotype),
                           reverse=False)[:elite_size]
            next_gen = self.__crossover__(mating_pool=mating_pool, size=self.POPULATION_SIZE - elite_size)
            if self.MUTATION_PROBABILITY > 0:
                self.__mutate__(next_gen)
            self.population = elite + next_gen
        if self.log:
            self.logs = maxFitness, minFitness, meanFitness, generations
        self.result = min([self.fitness(*phenotype.genotype) for phenotype in self.population])

    def __selection__(self, fitness_values):
        """Proportional selection for minimization"""
        selecting_probabilities = fitness_values
        sum_probability = abs(sum(selecting_probabilities))
        max_fitness = max(selecting_probabilities)
        for i in range(len(selecting_probabilities)):
            selecting_probabilities[i] = abs(max_fitness - selecting_probabilities[i]) / sum_probability
        mating_pool = random.choices(population=self.population,
                                     weights=selecting_probabilities,
                                     k=self.POPULATION_SIZE)
        return mating_pool

    def __crossover__(self, mating_pool, size: int):
        result = []
        for i in range(int(size / 2)):
            result.extend(self.__single_point_crossover__(random.choice(mating_pool), random.choice(mating_pool)))
        if size % 2 != 0:
            result.append(self.__single_point_crossover__(random.choice(mating_pool), random.choice(mating_pool))[0])
        return result

    def __single_point_crossover__(self, parent1: Phenotype, parent2: Phenotype):
        point = random.randint(1, self.PARAM_SIZE - 1)
        child1 = Phenotype(parent1.genotype[:point] + parent2.genotype[point:])
        child2 = Phenotype(parent2.genotype[:point] + parent1.genotype[point:])
        return [child1, child2]

    def __mutate__(self, new_generation: list[Phenotype]):
        indexes = random.choices(range(len(new_generation)), k=int(self.MUTATION_PROBABILITY * len(new_generation)))
        for i in indexes:
            parameter = random.randint(0, self.PARAM_SIZE - 1)
            new_generation[i].genotype[parameter] += random.randint(-1, 1) * random.random()

    def get_logs(self) -> tuple[list, list, list, list]:
        return self.logs


def init_plot(bounds: tuple[float, float]):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    X = np.arange(bounds[0], bounds[1], 0.5)
    Y = np.arange(bounds[0], bounds[1], 0.5)
    X, Y = np.meshgrid(X, Y)
    Z = X ** 2 + 1.5 * Y ** 2 - 2 * X * Y + 4 * X - 8 * Y

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness function')
    ax1.set_title('max/mean/min fitness function')


if __name__ == '__main__':
    def func(x: float, y: float) -> float:
        return x ** 2 + 1.5 * y ** 2 - 2 * x * y + 4 * x - 8 * y  # optimum: (2.0;4.0) = -12.0;


    algo = GenAlgo(objective_function=func,
                   bounds=(-100, 100),
                   population_size=100,
                   max_generations=200,
                   mutation_probability=0.1,
                   log=True)
    algo.run()
    maxF, minF, meanFs, generationsList = algo.get_logs()
