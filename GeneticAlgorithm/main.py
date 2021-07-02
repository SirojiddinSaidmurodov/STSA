import math

from GeneticAlgorithm.genalgo import GenAlgo


def func(x: float, y: float) -> float:
    return 3 * (1 - x) ** 2 * math.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * math.exp(
        -x ** 2 - y ** 2) - 1 / 3 * math.exp(-(x + 1) ** 2 - y ** 2)


algo = GenAlgo(objective_function=func,
               bounds=(-3, 3),
               population_size=200,
               max_generations=100,
               mutation_probability=0.5,
               mutation_scale=0.03,
               log=True)
algo.run()
print(algo.result)
