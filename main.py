import time

from deap import base
from deap import creator
from deap import tools
import random
from matplotlib import pyplot as plt


def individual(icls):
    genome = list()
    genome.append(random.uniform(-10, 10))
    genome.append(random.uniform(-10, 10))
    return icls(genome)


def fitnessFunction(individual):
    result = pow(individual[0] + 2 * individual[1] - 7, 2) + pow(2 * individual[0] + individual[1] - 5, 2)
    return result,


def main():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register('individual', individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitnessFunction)
    plot_x, plot_value, plot_std, plot_avg = [], [], [], []
    print("It's a Genetic algorithm with couple of deap library examples")
    print("Please choose details:")

    print("Choose selection: \n 0: Tournament \n 1: Random \n 2: Best \n 3: Worst \n 4: Roulette \n 5: Blend")
    user_input_selection = input()
    if user_input_selection == '0':
        print("Choose tournament size: ")
        tournament_size = input()
        toolbox.register("select", tools.selTournament, tournsize=int(tournament_size))
    elif user_input_selection == '1':
        toolbox.register("select", tools.selRandom)
    elif user_input_selection == '2':
        toolbox.register("select", tools.selBest)
    elif user_input_selection == '3':
        toolbox.register("select", tools.selWorst)
    elif user_input_selection == '4':
        toolbox.register("select", tools.selRoulette)
    elif user_input_selection == '5':
        toolbox.register("select", tools.cxUniformPartialyMatched, indpb=0.9)
    else:
        toolbox.register("select", tools.selBest)

    print("Choose crossover method: \n 0: One point \n 1: Two point \n 2: uniform \n 3: Messy One point")
    user_input_mate = input()
    if user_input_mate == '0':
        toolbox.register("mate", tools.cxOnePoint)
    elif user_input_mate == '1':
        toolbox.register("mate", tools.cxTwoPoint)
    elif user_input_mate == '2':
        toolbox.register("mate", tools.cxUniform, indpb=0.1)
    elif user_input_mate == '3':
        toolbox.register("mate", tools.cxMessyOnePoint)
    else:
        toolbox.register("mate", tools.cxOnePoint)

    print("Choose mutation method: \n 0: Gaussian \n 1: Uniform Int \n 2: Shuffle Indexes")
    user_input_mutation = input()
    if user_input_mutation == '0':
        toolbox.register("mutate", tools.mutGaussian, mu=5, sigma=10, indpb=0.9)
    if user_input_mutation == '1':
        toolbox.register("mutate", tools.mutUniformInt, low=-10, up=10)
    if user_input_mutation == '2':
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.9)
    else:
        toolbox.register("mutate", tools.mutGaussian, mu=5, sigma=10, indpb=0.9)

    print("Set size of population: ")
    # user_input_population_size = input()
    user_input_population_size = '50'
    sizePopulation = int(user_input_population_size)

    print("Set mutation probablility: ")
    # user_input_mutation_probability = input()
    user_input_mutation_probability = '0.1'
    probabilityMutation = float(user_input_mutation_probability)

    print("Set crossover probability: ")
    # user_input_crossover_probability = input()
    user_input_crossover_probability = '0.9'
    probabilityCrossover = float(user_input_crossover_probability)

    print("Set number of generations: ")
    # user_input_number_of_generations = input()
    user_input_number_of_generations = '50'
    numberIteration = int(user_input_number_of_generations)

    pop = toolbox.population(n=sizePopulation)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    g = 0
    numberElitism = 2
    start_time = time.clock()
    while g < numberIteration:
        g = g + 1
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        listElitism = []
        for x in range(0, numberElitism):
            listElitism.append(tools.selBest(pop, 1)[0])
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < probabilityCrossover:
                toolbox.mate(child1, child2)
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < probabilityMutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values
            # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(" Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring + listElitism
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        print(" Min %s" % min(fits))
        print(" Max %s" % max(fits))
        print(" Avg %s" % mean)
        print(" Std %s" % std)
        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind,
                                             best_ind.fitness.values))
        plot_x.append(g)
        plot_avg.append(mean)
        plot_std.append(std)
        plot_value.append(best_ind.fitness.values)
    #
    timeOfAll = time.clock() - start_time
    print(f'Time: {timeOfAll}')
    print("-- End of (successful) evolution --")

    plt.figure()
    plt.plot(plot_value)
    plt.ylabel('Best values')
    plt.show()
    plt.figure()
    plt.plot(plot_avg)
    plt.ylabel('Averages')
    plt.show()
    plt.figure()
    plt.plot(plot_std)
    plt.ylabel('Standard deviation')
    plt.show()

if __name__ == '__main__':
    main()
