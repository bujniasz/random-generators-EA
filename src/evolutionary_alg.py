import numpy as np

def evolutionary_classic(q, p0, u, delta_small, delta_big, p_big_jump, pc, t_max, limit, rng):
    """
    Perform the evolutionary algorithm to optimize a given function.

    This function represents a classic evolutionary algorithm approach where a
    population of individuals evolves through reproduction and mutation processes
    to find an optimal or near-optimal solution to the optimization problem.

    Parameters:
        q (callable): Objective function to be minimized.
        p0 (list): Initial population, where each individual is a solution.
        u (int): Number of individuals in a population.
        p_big_jump (float): Probability of mutating far in the space.
        delta_big (int or float): Coeffcient of far mutation.
        delta_small (int or float): Coefficient of standard mutation.
        pc: Probability of crossover.
        t_max (int or float): Maximum number of generations (iterations).
        limit (int or float): Boundary limit for the solution.

    Returns:
        tuple:
            - float: Best score (objective function value) achieved during the evolution.
            - list of float: History of best scores per generation.
    """
    t = 0
    o = grade(q, p0)
    best_grade, best_x = find_best(o)

    history = [best_grade]

    while t <= t_max:
        r = reproduce(o, u, rng)
        c = crossover(r, pc, rng)
        m = mutate(c, delta_small, delta_big, p_big_jump, limit, rng)
        om = grade(q, m)
        curr_best_grade, curr_best_x = find_best(om)

        if curr_best_grade < best_grade:
            best_grade = curr_best_grade
            best_x = curr_best_x
        
        history.append(best_grade)
        o = om  
        t += 1

    return best_grade, history


def grade(q, population):
    """
    Grades each individual in the population according to the objective function.

    Parameters:
        q (callable): The objective function.
        population (list): The population to be graded.

    Returns:
        list of tuples: Each tuple contains the score and the individual.
    """
    graded_population = []

    for x in range(len(population)):
        graded_population.append((q(population[x]), population[x]))

    return graded_population


def find_best(graded_population):
    """
    Finds the best individual in the graded population.

    Parameters:
        graded_population (list of tuples): Each tuple contains a score and an individual.

    Returns:
        tuple: The best score and the corresponding individual.
    """
    graded_population.sort(key=lambda val: val[0])
    best_grade = graded_population[0][0]
    best_x = graded_population[0][1]

    return best_grade, best_x


def reproduce(graded_population, population_count, rng):
    """
    Tournament selection. Reproduces a new population from the graded population.

    Parameters:
        graded_population (list of tuples): The graded population.
        population_count (int): The number of individuals in the population.

    Returns:
        list: The new population.
    """
    new_population = []

    for _ in range(population_count):
        grade1, x1 = rng.choice(graded_population)
        grade2, x2 = rng.choice(graded_population)

        if grade1 > grade2:
            new_population.append(x2)
        else:
            new_population.append(x1)

    return new_population


def mutate(population, delta_small, delta_big, p_big_jump, limit, rng):
    """
    Applies Gaussian mutation to the population.

    Parameters:
        population (list): The population to mutate.
        p_big_jump (float): Probability of mutating far in the space.
        delta_big (int or float): Coeffcient of far mutation.
        delta_small (int or float): Coefficient of standard mutation.
        limit (int or float): The boundary limit for mutation.

    Returns:
        list: The mutated population.
    """
    mutated_population = []

    for x in population:
        if rng.rand() < p_big_jump:
            delta = delta_big
        else:
            delta = delta_small

        perturb = rng.uniform(-delta, delta, 1).reshape(-1)
        mutated_x = np.clip(x + perturb, -limit, limit)
        mutated_population.append(mutated_x)

    return mutated_population


def crossover(population, pc, rng):
    """
    One-point crossover operator. Combines individuals from the population 
    to produce new offspring based on crossover probability.

    Parameters:
        population (list): The current population of individuals (numpy arrays).
        pc (float): Probability of performing crossover on an individual.

    Returns:
        list: The new population after crossover.
    """
    crossed_population = []

    for x in population:
        if rng.rand() <= pc and len(x) > 2:
            candidates = [ind for ind in population if not np.array_equal(ind, x)]
            if not candidates:
                crossed_population.append(x)
                continue

            partner = rng.choice(candidates)
            split = rng.randrange(1, len(x) - 1)
            child = np.concatenate((x[:split], partner[split:]))
            crossed_population.append(child)
        else:
            crossed_population.append(x)

    return crossed_population