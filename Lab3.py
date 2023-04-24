import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
import csv

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

# Parse terminal arguments
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

# Assign the parsed arguments to variables
mu = args.mu
lambd = args.lambd
generations = args.generations
x_range = (args.x_min, args.x_max)
y_range = (args.y_min, args.y_max)
mutation_strength = args.mutation_strength
mutation_probability = args.mutation_probability
visualize = args.visualize

# CSV file to save results
csv_file = "results.csv"

results = []

with open(csv_file, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    # Write header row
    csv_writer.writerow(['mu', 'lambd', 'mutation_strength', 'mutation_probability', 'optimal_solution_x', 'optimal_solution_y', 'bukin_function(optimal_solution[0], optimal_solution[1])', 'optimization_time'])

    # Perform the optimization with the parsed arguments
    start_time = time.time()
    optimal_solution = evolutionary_strategy(mu, lambd, generations, x_range, y_range, mutation_strength, mutation_probability, visualize)
    optimization_time = time.time() - start_time

    # Write the result row
    csv_writer.writerow([mu, lambd, mutation_strength, mutation_probability, optimal_solution[0], optimal_solution[1], bukin_function(optimal_solution[0], optimal_solution[1]), optimization_time])

# Output the results
print(f"Parameters: mu={mu}, lambd={lambd}, mutation_strength={mutation_strength}, mutation_probability={mutation_probability}")
print("Optimal solution:", optimal_solution)
print("Optimization time (excluding visualization):", optimization_time, "seconds")
