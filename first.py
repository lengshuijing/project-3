import numpy as np
import random
from itertools import count
import imageio
import gym
from tqdm import tqdm
from graphviz import Digraph

# Gene class, representing the structure of a neural network
class Gene:
    def __init__(self, input_size, output_size):
        self.nodes = {i: {'type': 'input'} for i in range(input_size)}
        self.nodes.update({i + input_size: {'type': 'output'} for i in range(output_size)})
        self.connections = {}
        self.next_node_id = count(start=input_size + output_size)
        self.initialize_connections()

    def initialize_connections(self):
        # Initialize connections from inputs to outputs
        for input_node in self.nodes:
            if self.nodes[input_node]['type'] == 'input':
                for output_node in self.nodes:
                    if self.nodes[output_node]['type'] == 'output':
                        weight = np.random.randn()
                        self.connections[(input_node, output_node)] = weight

    def add_node(self, connection):
        # Add a new node to split a connection
        new_node_id = next(self.next_node_id)
        self.nodes[new_node_id] = {'type': 'hidden'}
        weight = self.connections.pop(connection)
        self.connections[(connection[0], new_node_id)] = 1.0
        self.connections[(new_node_id, connection[1])] = weight

    def add_connection(self, from_node, to_node):
        # Add a new connection
        self.connections[(from_node, to_node)] = np.random.randn()

# NEAT population class, managing the evolution of genes
class NeatPopulation:
    def __init__(self, input_size, output_size, population_size):
        self.population = [Gene(input_size, output_size) for _ in range(population_size)]
        self.fitness = [0.0] * population_size
        self.generation = 0

    def evaluate_fitness(self, env, num_episodes=5):
        # Evaluate the fitness of each gene in the population
        for i, gene in enumerate(self.population):
            fitness = 0.0
            for _ in range(num_episodes):
                state = env.reset()
                done = False
                while not done:
                    action = self.activate(gene, state)
                    state, reward, done, _ = env.step(action)
                    fitness += reward
            self.fitness[i] = fitness / num_episodes

    def activate(self, gene, state):
        # Forward propagation to compute network output
        output = np.zeros(len(gene.nodes))
        for node in gene.nodes:
            if gene.nodes[node]['type'] == 'input':
                output[node] = state[node]
        for (from_node, to_node), weight in gene.connections.items():
            output[to_node] += output[from_node] * weight
        return output.argmax()  # Assuming discrete actions

    def evolve(self):
        # Evolve the population through selection, crossover, and mutation
        parents = sorted(range(len(self.population)), key=lambda k: self.fitness[k], reverse=True)[:2]
        new_population = [self.population[i] for i in parents]
        for _ in range(len(self.population) - len(parents)):
            parent1 = random.choice(new_population)
            parent2 = random.choice(new_population)
            child = self.mutate(parent1, parent2)
            new_population.append(child)
        self.population = new_population
        self.generation += 1

    def mutate(self, parent1, parent2):
        # Create a new gene and apply mutations
        child = Gene(len(parent1.nodes), len(parent1.nodes))
        # Inherit connections
        for (from_node, to_node) in parent1.connections:
            weight = parent1.connections.get((from_node, to_node), 0.0)
            child.connections[(from_node, to_node)] = weight
        # Apply mutations
        if random.random() < 0.2:
            child.add_node(random.choice(list(child.connections.keys())))
        elif random.random() < 0.4:
            child.add_connection(*random.sample(child.nodes.keys(), 2))
        return child

# Visualize the neural network structure
def visualize_network(gene, filename='network.png'):
    dot = Digraph()
    for node_id in gene.nodes:
        dot.node(str(node_id))
    for (from_node, to_node), weight in gene.connections.items():
        dot.edge(str(from_node), str(to_node), label=f"{weight:.2f}")
    dot.render(filename, view=True)

# Train the NEAT agent
def train_neat(population, env, generations=100):
    for generation in tqdm(range(generations), desc="Training NEAT"):
        population.evaluate_fitness(env)
        population.evolve()

# Save a GIF animation
def save_gif(gene, env, filename='slimevolley.gif'):
    frames = []
    state = env.reset()
    done = False
    while not done:
        frames.append(env.render(mode='rgb_array'))
        action = population.activate(gene, state)
        state, _, done, _ = env.step(action)
    imageio.mimsave(filename, frames, fps=30)

if __name__ == "__main__":
    # Initialize the environment
    env = gym.make("SlimeVolley-v0")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    # Initialize the population
    population = NeatPopulation(input_size, output_size, population_size=50)

    # Train the population
    train_neat(population, env, generations=1000)

    # Save the best gene
    best_gene = population.population[np.argmax(population.fitness)]
    visualize_network(best_gene)
    save_gif(best_gene, env)