import math

import numpy as np
from matplotlib import pyplot as plt, animation, cm

from GeneticAlgorithm.genalgo import GenAlgo

fig = plt.figure()
fig.set_size_inches(12.8, 7.2, True)
fig.tight_layout()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax3 = fig.add_subplot(2, 2, 3)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])

bounds = (-3, 3)
X = np.arange(bounds[0], bounds[1], 0.5)
Y = np.arange(bounds[0], bounds[1], 0.5)
X, Y = np.meshgrid(X, Y)
Z = 3 * (1 - X) ** 2 * np.exp(-(X ** 2) - (Y + 1) ** 2) - 10 * (X / 5 - X ** 3 - Y ** 5) * np.exp(
    -X ** 2 - Y ** 2) - 1 / 3 * np.exp(-(X + 1) ** 2 - Y ** 2)


def init_plot():
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness function')
    ax1.set_title('max/mean/min fitness function')


maxHistory, meanHistory, minHistory = [], [], []


def update(frame, generations_list):
    ax2.clear()
    ax2.plot_wireframe(X, Y, Z, linewidth=0.5, alpha=0.5)
    ax2.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.coolwarm, antialiased=True)
    ax2.view_init(25, 0.6 * frame)
    ax2.set_axis_off()
    ax2.use_sticky_edges = True
    index = math.floor(frame / 30)
    population = generations_list[index]

    for phenotype in population:
        ax2.scatter(phenotype.genotype[0], phenotype.genotype[1], phenotype.fitness(), marker='.', edgecolor='red')

    if (frame % 30) == 0:
        fitness = []
        for phenotype in population:
            fitness.append(phenotype.fitness())

        minHistory.append(min(fitness))
        meanHistory.append(sum(fitness) / len(fitness))
        maxHistory.append(max(fitness))
        ax1.clear()
        if index > 5:
            start = index - 5
            ax1.set_xlim(start, index)
            ax1.set_ylim(min(minHistory[start: index]), max(maxHistory[start: index]))

        ax1.plot(minHistory, label='min')
        ax1.plot(meanHistory, label='mean')
        ax1.plot(maxHistory, label='max')
        ax1.set_ylabel('Fitness function')
        ax1.set_title('max/mean/min fitness function')
        ax1.legend()

        ax3.clear()
        ax3.set_title("Fitness of each phenotype")
        ax3.set_ylabel("Score")
        ax3.set_xlabel("Phenotype")
        ax3.bar([i + 1 for i in range(len(fitness))], fitness)

    ax2.set_title("Generation: " + str(index) + ", best score: " + str(minHistory[index]))

def func(x: float, y: float) -> float:
    r"""Function

    .. math::

        3(1 - x)^2 e^{(-(x^2) - (y + 1)^2)} - 10(x/5 - x^3 - y^5) e^(-x^2 - y^2)- 1/3 e^(-(x + 1)^2 - y^2)

    """
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
generations = algo.get_logs()
ani = animation.FuncAnimation(fig, update, init_func=init_plot, fargs=(generations,), interval=300, frames=3000)
# don't use with frames
# plt.show()
writer = animation.writers['ffmpeg'](fps=60, bitrate=5000)
ani.save('animation.mp4', writer=writer, dpi=150)
print(min([i.fitness() for i in generations[-1]]))
