import numpy as np
from deap import base, tools, creator, cma
from collections import deque

# Ensure fitness and individual are created
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def run_bipop_cmaes(pymoo_problem, dimension, budget, verbose=False):
    SIGMA0 = 2.0  # Initial standard deviation

    toolbox = base.Toolbox()
    # Fitness function compatible with CocoProblemWrapper
    def fitness_function(individual):
        out = {}
        pymoo_problem._evaluate(np.array(individual).reshape(1, -1), out)
        return (out["F"][0],)

    toolbox.register("evaluate", fitness_function)

    halloffame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbooks = list()

    nsmallpopruns = 0
    smallbudget = list()
    largebudget = list()
    lambda0 = 4 + int(3 * np.log(dimension))
    regime = 1
    i = 0
    total_evaluations = 0  # Track the total number of evaluations

    while total_evaluations < budget:
        # Determine the regime
        if i > 0 and sum(smallbudget) < sum(largebudget):
            lambda_ = int(lambda0 * (0.5 * (2**(i - nsmallpopruns) * lambda0) / lambda0)**(np.random.rand()**2))
            sigma = 2 * 10**(-2 * np.random.rand())
            nsmallpopruns += 1
            regime = 2
            smallbudget += [0]
        else:
            lambda_ = 2**(i - nsmallpopruns) * lambda0
            sigma = SIGMA0
            regime = 1
            largebudget += [0]

        t = 0

        # Set the termination criterion constants
        if regime == 1:
            MAXITER = min(budget - total_evaluations, 100 + 50 * (dimension + 3)**2 / np.sqrt(lambda_))
        elif regime == 2:
            MAXITER = min(budget - total_evaluations, 0.5 * largebudget[-1] / lambda_)

        strategy = cma.Strategy(centroid=np.random.uniform(-4, 4, dimension), sigma=sigma, lambda_=lambda_)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        logbooks.append(tools.Logbook())
        logbooks[-1].header = "gen", "evals", "restart", "regime", "std", "min", "avg", "max"

        conditions = {"MaxIter": False}

        while not conditions["MaxIter"]:
            population = toolbox.generate()

            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            halloffame.update(population)
            record = stats.compile(population)
            logbooks[-1].record(gen=t, evals=lambda_, restart=i, regime=regime, **record)
            if verbose:
                print(logbooks[-1].stream)

            toolbox.update(population)

            total_evaluations += lambda_
            if total_evaluations >= budget:
                conditions["MaxIter"] = True

            t += 1

        i += 1

    best_individual = halloffame[0]
    best_fitness = best_individual.fitness.values[0]
    return best_individual, best_fitness
