import numpy as np
import time
import matplotlib.pyplot as plt
import argparse

# Define the Bukin function
def bukin_function(x, y):
    return 100 * np.sqrt(np.abs(y - 0.01 * x ** 2)) + 0.01 * np.abs(x + 10)

# Create the initial population within the given search space
def create_initial_population(mu, x_range, y_range):
    population = np.column_stack([np.random.uniform(x_range[0], x_range[1], mu),
                                  np.random.uniform(y_range[0], y_range[1], mu)])
    return population

# Tournament selection for parent selection
def tournament_selection(population, fitness_values, k=2):
    indices = np.random.choice(len(population), size=k, replace=False)
    best_index = np.argmin(fitness_values[indices])
    return population[indices[best_index]]

# Gaussian mutation with mutation probability
def mutate(parent, mutation_strength, mutation_probability):
    mutation_flag = np.random.rand(2) < mutation_probability
    mutation_noise = np.random.normal(0, mutation_strength, 2) * mutation_flag
    offspring = parent + mutation_noise
    return offspring

# Plot the population for visualization
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

# Evolutionary strategy optimization algorithm
def evolutionary_strategy(mu, lambd, generations, x_range, y_range, mutation_strength, mutation_probability, visualize=False):
    # Generate the initial population within the given search space
    population = create_initial_population(mu, x_range, y_range)
    
    # Iterate through the specified number of generations
    for generation in range(1, generations+1):
        offspring_population = []
        
        # Generate lambda offspring by applying mutation to the mu parents
        for _ in range(lambd):
            # Select a parent from the population using tournament selection
            parent = tournament_selection(population, np.array([bukin_function(x, y) for x, y in population]))
            # Generate a new offspring by applying Gaussian mutation to the parent with the given mutation probability
            offspring = mutate(parent, mutation_strength, mutation_probability)
            offspring_population.append(offspring)
        
        # Combine the offspring population with the parent population
        offspring_population = np.array(offspring_population)
        combined_population = np.vstack([population, offspring_population])
        
        # Calculate the fitness values for the combined population
        fitness_values = np.array([bukin_function(x, y) for x, y in combined_population])
        # Select the top mu individuals from the combined population based on their fitness values
        best_individuals_indices = np.argsort(fitness_values)[:mu]
        population = combined_population[best_individuals_indices]

        # Visualize the optimization process if the option is enabled
        if visualize:
            plot_population(population, generation)
    
    # Return the best solution from the final population as the optimal solution
    best_solution = population[np.argmin(fitness_values[:mu])]
    return best_solution


# # Parameters
# mu = 50
# lambd = 100
# generations = 200
# x_range = (-15, -5)
# y_range = (-3, 3)
# mutation_strength = 0.5
# mutation_probability = 0.9
# visualize = True

import csv

# Option-1 Parameters to test
mu_values = [10, 50, 100]
lambd_values = [20, 100, 200]
mutation_strength_values = [0.1, 0.5, 1.0]
mutation_probability_values = [0.5, 0.9, 0.99]
generations = 200
x_range = (-15, -5)
y_range = (-3, 3)
# Disable visualization for parameter testing
visualize = False

# Option-2 Parse terminal arguments
parser = argparse.ArgumentParser(description='Evolutionary Strategy Optimization for the Bukin Function')
parser.add_argument('--mu', type=int, default=50, help='Number of parents (default: 50)')
parser.add_argument('--lambd', type=int, default=100, help='Number of offspring (default: 100)')
parser.add_argument('--generations', type=int, default=200, help='Number of generations (default: 200)')
parser.add_argument('--x_min', type=float, default=-15, help='Minimum x value for search space (default: -15)')
parser.add_argument('--x_max', type=float, default=-5, help='Maximum x value for search space (default: -5)')
parser.add_argument('--y_min', type=float, default=-3, help='Minimum y value for search space (default: -3)')
parser.add_argument('--y_max', type=float, default=3, help='Maximum y value for search space (default: 3)')
parser.add_argument('--mutation_strength', type=float, default=0.5, help='Mutation strength (default: 0.5)')
parser.add_argument('--mutation_probability', type=float, default=0.9, help='Mutation probability (default: 0.9)')
parser.add_argument('--visualize', action='store_true', help='Enable visualization (default: False)')
args = parser.parse_args()
mu = args.mu
lambd = args.lambd
generations = args.generations
x_range = (args.x_min, args.x_max)
y_range = (args.y_min, args.y_max)
mutation_strength = args.mutation_strength
mutation_probability = args.mutation_probability
visualize = args.visualize
# Example
# python main.py --mu 50 --lambd 100 --generations 200 --x_min -15 --x_max -5 --y_min -3 --y_max 3 --mutation_strength 0.5 --mutation_probability 0.9


# CSV file to save results
csv_file = "results.csv"

results = []

with open(csv_file, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    # Write header row
    csv_writer.writerow(['mu', 'lambd', 'mutation_strength', 'mutation_probability', 'optimal_solution_x', 'optimal_solution_y', 'bukin_function(optimal_solution[0], optimal_solution[1])', 'optimization_time'])

    for mu in mu_values:
        for lambd in lambd_values:
            for mutation_strength in mutation_strength_values:
                for mutation_probability in mutation_probability_values:
                    # Measure optimization time and optimize the Bukin function with the current parameter combination
                    start_time = time.time()
                    optimal_solution = evolutionary_strategy(mu, lambd, generations, x_range, y_range, mutation_strength, mutation_probability, visualize)
                    optimization_time = time.time() - start_time

                    # Write the result row
                    csv_writer.writerow([mu, lambd, mutation_strength, mutation_probability, optimal_solution[0], optimal_solution[1], bukin_function(optimal_solution[0], optimal_solution[1]), optimization_time])
                    print(f"Parameters: mu={mu}, lambd={lambd}, mutation_strength={mutation_strength}, mutation_probability={mutation_probability}")
                    print("Optimal solution:", optimal_solution)
                    print("Optimization time (excluding visualization):", optimization_time, "seconds")
                    print("----------------------------------------------------")
                    
                    # Calculate the fitness value of the optimal solution
                    fitness_value = bukin_function(optimal_solution[0], optimal_solution[1])

                    # Store the result in the results list
                    result = {'mu': mu, 'lambd': lambd, 'mutation_strength': mutation_strength, 'mutation_probability': mutation_probability,
                              'optimal_solution': optimal_solution, 'fitness_value': fitness_value, 'optimization_time': optimization_time}
                    results.append(result)


# # Measure optimization time and optimize the Bukin function
# start_time = time.time()
# optimal_solution = evolutionary_strategy(mu, lambd, generations, x_range, y_range, mutation_strength, mutation_probability, visualize)
# optimization_time = time.time() - start_time

# # Output the results
# print("Optimal solution:", optimal_solution)
# print("Optimization time (excluding visualization):", optimization_time, "seconds")

# Sort the results list based on fitness_value
sorted_results = sorted(results, key=lambda x: x['fitness_value'])

# Print the best result (lowest fitness value)
best_result = sorted_results[0]
print("Best Result:")
print(f"Parameters: mu={best_result['mu']}, lambd={best_result['lambd']}, mutation_strength={best_result['mutation_strength']}, mutation_probability={best_result['mutation_probability']}")
print("Optimal solution:", best_result['optimal_solution'])
print("Fitness value:", best_result['fitness_value'])
print("Optimization time (excluding visualization):", best_result['optimization_time'], "seconds")



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