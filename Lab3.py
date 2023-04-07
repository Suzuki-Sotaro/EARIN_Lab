import numpy as np
import time
import matplotlib.pyplot as plt

def bukin_function(x, y):
    return 100 * np.sqrt(np.abs(y - 0.01 * x ** 2)) + 0.01 * np.abs(x + 10)

def create_initial_population(mu, x_range, y_range):
    population = np.column_stack([np.random.uniform(x_range[0], x_range[1], mu),
                                  np.random.uniform(y_range[0], y_range[1], mu)])
    return population

def tournament_selection(population, fitness_values, k=2):
    indices = np.random.choice(len(population), size=k, replace=False)
    best_index = np.argmin(fitness_values[indices])
    return population[indices[best_index]]

def mutate(parent, mutation_strength, mutation_probability):
    mutation_flag = np.random.rand(2) < mutation_probability
    mutation_noise = np.random.normal(0, mutation_strength, 2) * mutation_flag
    offspring = parent + mutation_noise
    return offspring

def plot_population(population, generation):
    plt.scatter(population[:, 0], population[:, 1], label=f'Generation {generation}')
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Evolutionary Strategy Optimization Process')
    plt.pause(0.1)
    plt.clf()

def evolutionary_strategy(mu, lambd, generations, x_range, y_range, mutation_strength, mutation_probability, visualize=False):
    population = create_initial_population(mu, x_range, y_range)
    
    for generation in range(1, generations+1):
        offspring_population = []
        
        for _ in range(lambd):
            parent = tournament_selection(population, np.array([bukin_function(x, y) for x, y in population]))
            offspring = mutate(parent, mutation_strength, mutation_probability)
            offspring_population.append(offspring)
        
        offspring_population = np.array(offspring_population)
        combined_population = np.vstack([population, offspring_population])
        
        fitness_values = np.array([bukin_function(x, y) for x, y in combined_population])
        best_individuals_indices = np.argsort(fitness_values)[:mu]
        population = combined_population[best_individuals_indices]

        if visualize:
            plot_population(population, generation)
    
    best_solution = population[np.argmin(fitness_values[:mu])]
    return best_solution

# Parameters
mu = 50
lambd = 100
generations = 200
x_range = (-15, -5)
y_range = (-3, 3)
mutation_strength = 0.5
mutation_probability = 0.9
visualize = True

# Measure optimization time and optimize the Bukin function
start_time = time.time()
optimal_solution = evolutionary_strategy(mu, lambd, generations, x_range, y_range, mutation_strength, mutation_probability, visualize)
optimization_time = time.time() - start_time

print("Optimal solution:", optimal_solution)
print(bukin_function(optimal_solution[0], optimal_solution[1]))
print("Optimization time (excluding visualization):", optimization_time, "seconds")


"""
Parameters' explanation

mu: This parameter represents the population size or the number of individuals in the population. 
A larger population size generally increases the diversity of the solutions being explored, which can help in avoiding local optima. However, larger populations also increase the computational cost of the algorithm.

lambd: This parameter represents the offspring size or the number of offspring generated in each generation. 
Larger offspring sizes provide more potential solutions to be explored, but they also increase the computational cost of the algorithm.

generations: This parameter represents the number of generations the algorithm will run for. 
The termination criterion for the algorithm is based on the number of generations. A larger number of generations allows the algorithm to explore more solutions, but it also increases the optimization time.

x_range and y_range: These parameters define the search space's bounds for the x and y coordinates. 
It is essential to set these ranges so that the global minimum of the Bukin function lies within the search space.

mutation_strength: This parameter determines the strength of the Gaussian mutation applied to the parent's coordinates. 
A higher mutation strength allows the algorithm to explore more distant solutions but might cause it to overshoot the optimal solution. 
A lower value will result in more focused exploration, but convergence might be slower.

mutation_probability: This parameter controls the probability of mutation happening for each coordinate of the offspring. 
A higher mutation probability results in more frequent mutations, increasing the exploration of the search space. 
However, it might also cause the search to be more random, potentially increasing the optimization time.

visualize: This is a boolean parameter that controls whether the optimization process is visualized or not. 
If set to True, the population of each generation will be plotted, showing the convergence of the algorithm. 
Visualization helps to understand the optimization process but might also slow down the program's execution. If set to False, the visualization will be disabled, and the program will run faster.
"""