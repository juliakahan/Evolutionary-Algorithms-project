import opfunu.cec_based.cec2014 as cec
import random
from deap import base, creator, tools, algorithms
import numpy as np

# Define the fitness function for CEC
cec_function = cec.F12014()  # Sphere function from CEC 2014

# DEAP setup for evolutionary algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def evalCEC(individual):
    return cec_function.evaluate(individual),

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -100, 100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=cec_function.ndim)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalCEC)

# Evolutionary Algorithm parameters
population = toolbox.population(n=50)
NGEN = 40
CXPB, MUTPB = 0.5, 0.2

# Run the evolutionary algorithm
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
    fits = map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

best_ind = tools.selBest(population, 1)[0]
print("CEC problem")
print(f"Best individual is {best_ind}, Fitness: {best_ind.fitness.values}")

# GECCO

# Define the Rastrigin function
def rastrigin(individual):
    A = 10
    return A * len(individual) + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in individual]),


# Create types
# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)

# Define toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5.12, 5.12)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the genetic operators
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", rastrigin)

# Create population
pop = toolbox.population(n=300)

# Run the algorithm
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, 
                                stats=stats, halloffame=hof, verbose=False)


print("GECCO problem")
print("Best individual:", hof[0])
print("Best fitness:", hof[0].fitness.values[0])

