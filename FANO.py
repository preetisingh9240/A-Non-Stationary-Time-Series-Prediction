import numpy as np
import time


def constraints(x):
    return np.all(x >= -10) and np.all(x <= 10)  # Variable bounds


def FANO(population, objective_function, LB, UB, max_iterations):
    # Step 1: Initialize
    population_size, dimension = population.shape
    lower_bound, upper_bound = LB, UB
    fitness = np.apply_along_axis(objective_function, 1, population)

    # Store best solution so far
    if objective_function.__name__ == 'minimization_function':
        best_index = np.argmin(fitness)
    else:
        best_index = np.argmax(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]

    Convergence = np.zeros(max_iterations)
    ct = time.time()
    # Main optimization loop
    for t in range(max_iterations):
        for i in range(population_size):
            # Step 2: Calculate distances between members
            distances = np.linalg.norm(population - population[i], axis=1)

            # Determine Farthest Member (FM) and Nearest Member (NM)
            farthest_member_index = np.argmax(distances)
            nearest_member_index = np.argmin(distances[distances != 0])  # Ignore self-distance

            FM = population[farthest_member_index]
            NM = population[nearest_member_index]

            # Phase 1: Exploration (moving to the farthest member)
            r_i = np.random.uniform(0, 1, dimension)
            new_position_exploration = population[i] + r_i * (FM - population[i])

            # Validate constraints
            if constraints(new_position_exploration):
                new_fitness_exploration = objective_function(new_position_exploration)
                if (objective_function.__name__ == 'minimization_function' and
                    new_fitness_exploration < fitness[i]) or \
                        (objective_function.__name__ == 'maximization_function' and
                         new_fitness_exploration > fitness[i]):
                    population[i] = new_position_exploration
                    fitness[i] = new_fitness_exploration

            # Phase 2: Exploitation (moving to the nearest member)
            r_i = np.random.uniform(0, 1, dimension)
            new_position_exploitation = population[i] + r_i * (NM - population[i])

            # Validate constraints
            if constraints(new_position_exploitation):
                new_fitness_exploitation = objective_function(new_position_exploitation)
                if (objective_function.__name__ == 'minimization_function' and
                    new_fitness_exploitation < fitness[i]) or \
                        (objective_function.__name__ == 'maximization_function' and
                         new_fitness_exploitation > fitness[i]):
                    population[i] = new_position_exploitation
                    fitness[i] = new_fitness_exploitation

        # Update best solution
        if objective_function.__name__ == 'minimization_function':
            best_index = np.argmin(fitness)
        else:
            best_index = np.argmax(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        Convergence[t] = best_fitness
    ct = time.time() - ct
    return best_fitness, Convergence, best_solution, ct

