import random
from inspect import signature


class Phenotype:
    def __init__(self, genotype: list, fitness_function: callable):
        self.genotype = genotype
        self.fitness_function = fitness_function

    def fitness(self):
        return self.fitness_function(*self.genotype)

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
        self.population = [
            Phenotype([random.randint(int(bounds[0]), int(bounds[1])) for _ in range(self.PARAM_SIZE)], self.fitness)
            for _ in range(self.POPULATION_SIZE)]
        self.MUTATION_PROBABILITY = mutation_probability
        self.log = log
        self.generations = []

    def run(self):
        generations = self.generations
        for generation in range(self.MAX_GENERATIONS):
            fitness = [phenotype.fitness() for phenotype in self.population]

            if self.log:
                generations.append(self.population)

            mating_pool = self.__selection__(fitness)
            elite_size = 1 if int(self.POPULATION_SIZE * 0.1) == 0 else int(self.POPULATION_SIZE * 0.1)
            elite = sorted(self.population,
                           key=lambda phenotype: phenotype.fitness(),
                           reverse=False)[:elite_size]
            next_gen = self.__crossover__(mating_pool=mating_pool, size=self.POPULATION_SIZE - elite_size)
            if self.MUTATION_PROBABILITY > 0:
                self.__mutate__(next_gen)
            self.population = elite + next_gen
        self.result = min([phenotype.fitness() for phenotype in self.population])

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
            result.extend(self.__single_point_crossover__(mating_pool[i * 2], mating_pool[i * 2 + 1]))
        if size % 2 != 0:
            result.append(self.__single_point_crossover__(mating_pool[-1], random.choice(mating_pool))[0])
        return result

    def __single_point_crossover__(self, parent1: Phenotype, parent2: Phenotype):
        point = random.randint(1, self.PARAM_SIZE - 1)
        child1 = Phenotype(parent1.genotype[:point] + parent2.genotype[point:], self.fitness)
        child2 = Phenotype(parent2.genotype[:point] + parent1.genotype[point:], self.fitness)
        return [child1, child2]

    def __mutate__(self, new_generation: list[Phenotype]):
        indexes = random.choices(range(len(new_generation)), k=int(self.MUTATION_PROBABILITY * len(new_generation)))
        for i in indexes:
            parameter = random.randint(0, self.PARAM_SIZE - 1)
            new_generation[i].genotype[parameter] += random.randint(-1, 1) * random.random()

    def get_logs(self) -> list[list[Phenotype]]:
        return self.generations
