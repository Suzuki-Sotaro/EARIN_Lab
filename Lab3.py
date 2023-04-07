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
print("Optimization time (excluding visualization):", optimization_time, "seconds")
