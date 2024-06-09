import numpy as np
from mealpy import FloatVar, ACOR
import benchmark_functions as bf
from opfunu import cec_based
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

n_vars = 2
minimization = "min"
optimization = "hypersphere"


def bent_cigar(solution):
    func = cec_based.F32013(ndim=n_vars) #ndim - liczba genow
    return func.evaluate(solution)


def hypersphere(solution):
    func = bf.Hypersphere(n_dimensions=n_vars)
    return func(solution)


if __name__ == '__main__':

    paras_de_grid = {
        "epoch": [100, 200, 300],
        "pop_size": [50],
        "sample_count": [25],
        "intent_factor": [0.5],
        "zeta": [1],
    }

    if optimization == "bent_cigar":
        problem={
            "bounds": FloatVar(lb=(-5.,) * n_vars, ub=(5.,) * n_vars),
            "obj_func": bent_cigar,
            "minmax": minimization,
            "log_to": None,
        }
    else:
        problem = {
            "bounds": FloatVar(lb=(-10.,) * n_vars, ub=(10.,) * n_vars),
            "obj_func": hypersphere,
            "minmax": minimization,
            "log_to": None,
        }

    list_of_solutions = []
    all_histories = []
    labels = []

    for parameter_grid in list(ParameterGrid(paras_de_grid)):
        model = ACOR.OriginalACOR(**parameter_grid)
        g_best = model.solve(problem)
        list_of_solutions.append(g_best.target.fitness)

        fitness_history = [agent.target.fitness for agent in model.history.list_current_best]
        all_histories.append(fitness_history)

        label = f'epoch{parameter_grid["epoch"]}_pop_size{parameter_grid["pop_size"]}'
        labels.append(label)
        print(fitness_history)

    if minimization == "min":
        print(f'solution: {np.min(list_of_solutions)}')
    else:
        print(f'solution: {np.max(list_of_solutions)}')

    for i, history in enumerate(all_histories):
        plt.plot(history, label=labels[i])

    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('ACOR Optimization Progress')
    plt.legend()
    plt.show()