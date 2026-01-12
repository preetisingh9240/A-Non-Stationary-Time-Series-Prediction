import numpy as np
import time


# Lyrebird Optimization Algorithm
def LOA(population, objective_function, ub, lb, T):
    N, variables = population.shape
    # Initialize the best candidate solution
    varmin = lb
    varmax = ub
    best_fitness = float('inf')
    best_solution = None
    fitness = np.zeros(N)
    for i in range(N):
        fitness[i] = objective_function(population[i])

    convergence = np.zeros(T)
    ct = time.time()

    for t in range(T):
        for i in range(N):
            rp = np.random.rand()

            if rp <= 0.5:
                # Phase 1
                safe_areas = np.where(objective_function(population) < objective_function(population[i]))[0]
                if len(safe_areas) > 0:
                    k = np.random.choice(safe_areas)
                    r = np.random.rand()
                    population[i] = population[i] + r * (population[k] - population[i])
                    if objective_function(population[i]) < objective_function(fitness):
                        best_solution = np.copy(population[i])
                    population[i] = np.clip(population[i], varmin[i], varmax[i])

            else:
                # Phase 2
                r = np.random.rand()
                population[i] = population[i] + (1 - 2 * r) * (population.max(axis=0) - population.min(axis=0))
                if objective_function(population[i]) < objective_function(fitness):
                    best_solution = np.copy(population[i])
                population[i] = np.clip(population[i], varmin[i], varmax[i])

        # Save the best candidate solution so far
        if objective_function(best_solution) < objective_function(fitness):
            best_fitness = objective_function(best_solution)
        convergence[t] = np.min(fitness)
    ct = time.time() - ct
    return best_fitness, convergence, best_solution, ct
