import numpy as np
from matplotlib import pyplot as plt, animation

from GeneticAlgorithm.genalgo import GenAlgo

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

bounds = (-100, 100)
X = np.arange(bounds[0], bounds[1], 0.5)
Y = np.arange(bounds[0], bounds[1], 0.5)
X, Y = np.meshgrid(X, Y)
Z = X ** 2 + 1.5 * Y ** 2 - 2 * X * Y + 4 * X - 8 * Y


def init_plot():
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness function')
    ax1.set_title('max/mean/min fitness function')


maxHistory, meanHistory, minHistory = [], [], []


def update(frame, generations_list):
    ax2.clear()
    ax2.plot_wireframe(X, Y, Z, rstride=30, cstride=30, linewidth=1)

    ax2.scatter(2, 4, -12, marker='o', edgecolor='green', linewidths=2)
    population = generations_list[frame]
    fitness = []
    for phenotype in population:
        ax2.scatter(phenotype.genotype[0], phenotype.genotype[1], phenotype.fitness(), marker='.', edgecolor='red')
        fitness.append(phenotype.fitness())
    minHistory.append(min(fitness))
    meanHistory.append(sum(fitness) / len(fitness))
    maxHistory.append(max(fitness))

    ax1.clear()
    if frame > 5:
        start = frame - 5
        ax1.set_xlim(start, frame)
        ax1.set_ylim(min(minHistory[start: frame]), max(maxHistory[start: frame]))

    ax1.plot(minHistory, label='min')
    ax1.plot(meanHistory, label='mean')
    ax1.plot(maxHistory, label='max')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness function')
    ax1.set_title('max/mean/min fitness function')
    ax1.legend()

    ax2.set_title("Generation: " + str(frame) + ", best score: " + str(minHistory[frame]))


def func(x: float, y: float) -> float:
    r"""Function

    .. math::

        x^2 + 1.5y^2 - 2xy + 4x -8y

    Optimum point: (2.0;4.0)

    Optimum function:-12.0

    """
    return x ** 2 + 1.5 * y ** 2 - 2 * x * y + 4 * x - 8 * y


algo = GenAlgo(objective_function=func,
               bounds=(-100, 100),
               population_size=100,
               max_generations=200,
               mutation_probability=0.5,
               log=True)
algo.run()
generations = algo.get_logs()
ani = animation.FuncAnimation(fig, update, init_func=init_plot, fargs=(generations,))
plt.show()
print(min([i.fitness() for i in generations[-1]]))
