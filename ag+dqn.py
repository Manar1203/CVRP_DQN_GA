# -*- coding: utf-8 -*-
"""AG+DQN.ipynb

!pip install torch matplotlib numpy


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import pandas as pd

# --- Chargement de l'instance CVRP ---
def load_cvrp_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    coords, demands = [], []
    capacity = 0
    reading_coords, reading_demands = False, False

    for line in lines:
        if "CAPACITY" in line:
            capacity = int(line.split()[-1])
        elif "NODE_COORD_SECTION" in line:
            reading_coords = True
        elif "DEMAND_SECTION" in line:
            reading_coords = False
            reading_demands = True
        elif "DEPOT_SECTION" in line:
            break
        elif reading_coords:
            parts = line.strip().split()
            coords.append((float(parts[1]), float(parts[2])))
        elif reading_demands:
            parts = line.strip().split()
            demands.append(int(parts[1]))

    return np.array(coords), np.array(demands), capacity

# --- Réseau de neurones DQN ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DQN:
    def __init__(self, input_dim, output_dim, gamma=0.95, lr=1e-3):
        self.q_net = QNetwork(input_dim, output_dim)
        self.target_q_net = QNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.memory = []
        self.batch_size = 64

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        q_values = self.q_net(states)
        next_q_values = self.target_q_net(next_states)

        target = q_values.clone().detach()
        for i in range(self.batch_size):
            target[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i]).item()

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, state):
        with torch.no_grad():
            q_values = self.q_net(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()

# --- Fonctions CVRP ---
def evaluate(solution, coords, demands, capacity):
    cost, load, prev = 0, 0, 0
    for node in solution + [0]:
        if load + demands[node] > capacity:
            cost += np.linalg.norm(coords[prev] - coords[0])
            prev = 0
            load = 0
        cost += np.linalg.norm(coords[prev] - coords[node])
        load += demands[node]
        prev = node
    cost += np.linalg.norm(coords[prev] - coords[0])
    return cost

def mutate(solution, dqn):
    new_solution = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    state = np.array([i / len(solution), j / len(solution)])
    action = dqn.act(state)
    if action == 0:
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1]*size
    child[a:b] = p1[a:b]
    ptr = 0
    for x in p2:
        if x not in child:
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = x
    return child

def extract_routes(solution, coords, demands, capacity):
    routes, route, load = [], [], 0
    for node in solution:
        if load + demands[node] > capacity:
            routes.append(route)
            route = []
            load = 0
        route.append(node)
        load += demands[node]
    if route:
        routes.append(route)
    return routes

def plot_routes(routes, coords):
    depot = coords[0]
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab20', len(routes))

    for i, route in enumerate(routes):
        route_coords = [depot] + [coords[node] for node in route] + [depot]
        xs, ys = zip(*route_coords)
        plt.plot(xs, ys, marker='o', label=f'Tournée {i+1}', color=colors(i))

    plt.scatter(depot[0], depot[1], c='red', s=100, label='Dépôt')
    plt.title("Visualisation des tournées du véhicule (B-n38-k6)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def genetic_algorithm(dqn, coords, demands, capacity, n_gen=200, pop_size=50):
    clients = list(range(1, len(coords)))
    population = [random.sample(clients, len(clients)) for _ in range(pop_size)]
    best_solution = None
    best_cost = float('inf')
    for gen in range(n_gen):
        population.sort(key=lambda sol: evaluate(sol, coords, demands, capacity))
        new_pop = population[:10]  # élitisme
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:25], 2)
            child = crossover(p1, p2)
            child = mutate(child, dqn)
            new_pop.append(child)
        population = new_pop
        cost = evaluate(population[0], coords, demands, capacity)
        if cost < best_cost:
            best_cost = cost
            best_solution = population[0]
        if gen % 20 == 0 or gen == n_gen-1:
            print(f"Generation {gen+1}, coût: {cost:.2f}")
    return best_solution, best_cost

# --- Chargement de l'instance B-n38-k6 ---
coords, demands, capacity = load_cvrp_instance("B-n38-k6.vrp")

# --- Pré-entraînement du DQN ---
dqn = DQN(input_dim=2, output_dim=2)
for _ in range(1000):
    s = np.random.rand(2)
    a = np.random.randint(0, 2)
    r = np.random.rand()
    s2 = np.random.rand(2)
    dqn.memory.append((s, a, r, s2))
    dqn.update()

# --- Exécution multiple (30 runs) ---
results = []
best_overall_cost = float('inf')
best_overall_solution = None

for run in range(30):
    sol, cost = genetic_algorithm(dqn, coords, demands, capacity)
    results.append(cost)
    if cost < best_overall_cost:
        best_overall_cost = cost
        best_overall_solution = sol
    print(f"Run {run+1}: Coût = {cost:.2f}")

# --- Affichage de la meilleure tournée ---
routes = extract_routes(best_overall_solution, coords, demands, capacity)
plot_routes(routes, coords)

# --- Graphe des coûts sur les runs ---
plt.figure(figsize=(10,5))
plt.plot(results, marker='o')
plt.title("Évolution des coûts sur 30 runs (AG + DQN) pour B-n38-k6")
plt.xlabel("Run")
plt.ylabel("Coût total")
plt.grid(True)
plt.show()

# --- Export CSV ---
df = pd.DataFrame({"Best_Solution": best_overall_solution})
df.to_csv("best_solution_B-n38-k6.csv", index=False)
print(f"✅ Meilleure solution enregistrée avec un coût de {best_overall_cost:.2f}")

from google.colab import files
uploaded = files.upload()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import pandas as pd

# --- Chargement de l'instance CVRP ---
def load_cvrp_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    coords, demands = [], []
    capacity = 0
    reading_coords, reading_demands = False, False

    for line in lines:
        if "CAPACITY" in line:
            capacity = int(line.split()[-1])
        elif "NODE_COORD_SECTION" in line:
            reading_coords = True
        elif "DEMAND_SECTION" in line:
            reading_coords = False
            reading_demands = True
        elif "DEPOT_SECTION" in line:
            break
        elif reading_coords:
            parts = line.strip().split()
            coords.append((float(parts[1]), float(parts[2])))
        elif reading_demands:
            parts = line.strip().split()
            demands.append(int(parts[1]))

    return np.array(coords), np.array(demands), capacity

# --- Réseau de neurones DQN ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DQN:
    def __init__(self, input_dim, output_dim, gamma=0.95, lr=1e-3):
        self.q_net = QNetwork(input_dim, output_dim)
        self.target_q_net = QNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.memory = []
        self.batch_size = 64

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        q_values = self.q_net(states)
        next_q_values = self.target_q_net(next_states)

        target = q_values.clone().detach()
        for i in range(self.batch_size):
            target[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i]).item()

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, state):
        with torch.no_grad():
            q_values = self.q_net(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()

# --- Fonctions CVRP ---
def evaluate(solution, coords, demands, capacity):
    cost, load, prev = 0, 0, 0
    for node in solution + [0]:
        if load + demands[node] > capacity:
            cost += np.linalg.norm(coords[prev] - coords[0])
            prev = 0
            load = 0
        cost += np.linalg.norm(coords[prev] - coords[node])
        load += demands[node]
        prev = node
    cost += np.linalg.norm(coords[prev] - coords[0])
    return cost

def mutate(solution, dqn):
    new_solution = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    state = np.array([i / len(solution), j / len(solution)])
    action = dqn.act(state)
    if action == 0:
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1]*size
    child[a:b] = p1[a:b]
    ptr = 0
    for x in p2:
        if x not in child:
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = x
    return child

def extract_routes(solution, coords, demands, capacity):
    routes, route, load = [], [], 0
    for node in solution:
        if load + demands[node] > capacity:
            routes.append(route)
            route = []
            load = 0
        route.append(node)
        load += demands[node]
    if route:
        routes.append(route)
    return routes

def plot_routes(routes, coords):
    depot = coords[0]
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab20', len(routes))

    for i, route in enumerate(routes):
        route_coords = [depot] + [coords[node] for node in route] + [depot]
        xs, ys = zip(*route_coords)
        plt.plot(xs, ys, marker='o', label=f'Tournée {i+1}', color=colors(i))

    plt.scatter(depot[0], depot[1], c='red', s=100, label='Dépôt')
    plt.title("Visualisation des tournées du véhicule (Li_32)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def genetic_algorithm(dqn, coords, demands, capacity, n_gen=200, pop_size=50):
    clients = list(range(1, len(coords)))
    population = [random.sample(clients, len(clients)) for _ in range(pop_size)]
    best_solution = None
    best_cost = float('inf')
    for gen in range(n_gen):
        population.sort(key=lambda sol: evaluate(sol, coords, demands, capacity))
        new_pop = population[:10]  # élitisme
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:25], 2)
            child = crossover(p1, p2)
            child = mutate(child, dqn)
            new_pop.append(child)
        population = new_pop
        cost = evaluate(population[0], coords, demands, capacity)
        if cost < best_cost:
            best_cost = cost
            best_solution = population[0]
        if gen % 20 == 0 or gen == n_gen-1:
            print(f"Generation {gen+1}, coût: {cost:.2f}")
    return best_solution, best_cost

# --- Chargement de l'instance Li_32 ---
coords, demands, capacity = load_cvrp_instance("Li_32.vrp")

# --- Pré-entraînement du DQN ---
dqn = DQN(input_dim=2, output_dim=2)
for _ in range(1000):
    s = np.random.rand(2)
    a = np.random.randint(0, 2)
    r = np.random.rand()
    s2 = np.random.rand(2)
    dqn.memory.append((s, a, r, s2))
    dqn.update()

# --- Exécution multiple (30 runs) ---
results = []
best_overall_cost = float('inf')
best_overall_solution = None

for run in range(30):
    sol, cost = genetic_algorithm(dqn, coords, demands, capacity)
    results.append(cost)
    if cost < best_overall_cost:
        best_overall_cost = cost
        best_overall_solution = sol


# --- Affichage de la meilleure tournée ---
routes = extract_routes(best_overall_solution, coords, demands, capacity)
plot_routes(routes, coords)

# --- Graphe des coûts sur les runs ---
plt.figure(figsize=(10,5))
plt.plot(results, marker='o')
plt.title("Évolution des coûts sur 30 runs (AG + DQN) pour Li_32")
plt.xlabel("Run")
plt.ylabel("Coût total")
plt.grid(True)
plt.show()

# --- Export CSV ---
df = pd.DataFrame({"Best_Solution": best_overall_solution})
df.to_csv("best_solution_Li_32.csv", index=False)
print(f"✅ Meilleure solution enregistrée avec un coût de {best_overall_cost:.2f}")

from google.colab import files
uploaded = files.upload()

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# --- Chargement de l'instance CVRP ---
def load_cvrp_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    coords, demands = [], []
    capacity = 0
    reading_coords, reading_demands = False, False

    for line in lines:
        if "CAPACITY" in line:
            capacity = int(line.split()[-1])
        elif "NODE_COORD_SECTION" in line:
            reading_coords = True
        elif "DEMAND_SECTION" in line:
            reading_coords = False
            reading_demands = True
        elif "DEPOT_SECTION" in line:
            break
        elif reading_coords:
            parts = line.strip().split()
            coords.append((float(parts[1]), float(parts[2])))
        elif reading_demands:
            parts = line.strip().split()
            demands.append(int(parts[1]))

    return np.array(coords), np.array(demands), capacity

# --- Réseau de neurones DQN ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DQN:
    def __init__(self, input_dim, output_dim, gamma=0.99, lr=1e-3):
        self.q_net = QNetwork(input_dim, output_dim)
        self.target_q_net = QNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.memory = []
        self.batch_size = 64

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        q_values = self.q_net(states)
        next_q_values = self.target_q_net(next_states)

        target = q_values.clone().detach()
        for i in range(self.batch_size):
            target[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i]).item()

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, state):
        with torch.no_grad():
            q_values = self.q_net(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()

# --- Fonctions CVRP ---
def evaluate(solution, coords, demands, capacity):
    cost, load, prev = 0, 0, 0
    for node in solution + [0]:
        if load + demands[node] > capacity:
            cost += np.linalg.norm(coords[prev] - coords[0])
            prev = 0
            load = 0
        cost += np.linalg.norm(coords[prev] - coords[node])
        load += demands[node]
        prev = node
    cost += np.linalg.norm(coords[prev] - coords[0])
    return cost

def mutate(solution, dqn):
    new_solution = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    state = np.array([i / len(solution), j / len(solution)])
    action = dqn.act(state)
    if action == 0:
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1]*size
    child[a:b] = p1[a:b]
    ptr = 0
    for x in p2:
        if x not in child:
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = x
    return child

def genetic_algorithm(dqn, coords, demands, capacity, n_gen=200, pop_size=50):
    clients = list(range(1, len(coords)))
    population = [random.sample(clients, len(clients)) for _ in range(pop_size)]
    best_solution = None
    best_cost = float('inf')
    convergence = []

    for gen in range(n_gen):
        population.sort(key=lambda sol: evaluate(sol, coords, demands, capacity))
        new_pop = population[:10]  # élite
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:25], 2)
            child = crossover(p1, p2)
            child = mutate(child, dqn)
            new_pop.append(child)
        population = new_pop
        cost = evaluate(population[0], coords, demands, capacity)
        convergence.append(cost)
        if cost < best_cost:
            best_cost = cost
            best_solution = population[0]

    return best_solution, best_cost, convergence

def split_routes(solution, demands, capacity):
    routes = []
    current_route = []
    current_load = 0
    for client in solution:
        demand = demands[client]
        if current_load + demand > capacity:
            routes.append(current_route)
            current_route = [client]
            current_load = demand
        else:
            current_route.append(client)
            current_load += demand
    if current_route:
        routes.append(current_route)
    return routes

def plot_routes(routes, coords):
    depot = coords[0]
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab20', len(routes))
    for i, route in enumerate(routes):
        route_coords = [depot] + [coords[node] for node in route] + [depot]
        xs, ys = zip(*route_coords)
        plt.plot(xs, ys, marker='o', label=f'Tournée {i+1}', color=colors(i))
    plt.scatter(depot[0], depot[1], c='red', s=100, label='Dépôt')
    plt.title("Visualisation des tournées du véhicule - Meilleure solution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Chargement instance ---
coords, demands, capacity = load_cvrp_instance("X-n101-k25.vrp")

# --- Pré-entraînement simple du DQN ---
dqn = DQN(input_dim=2, output_dim=2)
for _ in range(1000):
    s = np.random.rand(2)
    a = np.random.randint(0, 2)
    r = np.random.rand()
    s2 = np.random.rand(2)
    dqn.memory.append((s, a, r, s2))
    dqn.update()

# --- Boucle sur 30 runs ---
results = []
best_overall_cost = float('inf')
best_overall_solution = None
best_convergence = None

for run in range(30):
    sol, cost, convergence = genetic_algorithm(dqn, coords, demands, capacity)
    results.append(cost)
    if cost < best_overall_cost:
        best_overall_cost = cost
        best_overall_solution = sol
        best_convergence = convergence
    print(f"Run {run+1}: Cost = {cost:.2f}")

# --- Graphique évolution des coûts sur 30 runs ---
plt.figure(figsize=(10,5))
plt.plot(results, marker='o')
plt.title("Évolution des 30 coûts (AG + DQN)")
plt.xlabel("Run")
plt.ylabel("Coûts totaux")
plt.grid(True)
plt.show()

# --- Courbe de convergence du meilleur run ---
plt.figure(figsize=(10,5))
plt.plot(best_convergence, marker='x')
plt.title("Courbe de convergence - Meilleur run (AG + DQN)")
plt.xlabel("Génération")
plt.ylabel("Coût")
plt.grid(True)
plt.show()

# --- Export CSV de la meilleure solution ---
df = pd.DataFrame({"Client": best_overall_solution})
df.to_csv("best_solution.csv", index=False)
print(f"✅ Meilleure solution enregistrée avec un coût de {best_overall_cost:.2f}")

# --- Affichage des tournées de la meilleure solution ---
routes = split_routes(best_overall_solution, demands, capacity)
plot_routes(routes, coords)

from google.colab import files
uploaded = files.upload()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import pandas as pd

# --- Chargement de l'instance CVRP ---
def load_cvrp_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    coords, demands = [], []
    capacity = 0
    reading_coords, reading_demands = False, False

    for line in lines:
        if "CAPACITY" in line:
            capacity = int(line.split()[-1])
        elif "NODE_COORD_SECTION" in line:
            reading_coords = True
        elif "DEMAND_SECTION" in line:
            reading_coords = False
            reading_demands = True
        elif "DEPOT_SECTION" in line:
            break
        elif reading_coords:
            parts = line.strip().split()
            coords.append((float(parts[1]), float(parts[2])))
        elif reading_demands:
            parts = line.strip().split()
            demands.append(int(parts[1]))

    return np.array(coords), np.array(demands), capacity

# --- Réseau de neurones DQN ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DQN:
    def __init__(self, input_dim, output_dim, gamma=0.99, lr=1e-3):
        self.q_net = QNetwork(input_dim, output_dim)
        self.target_q_net = QNetwork(input_dim, output_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # synchronisation initiale
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.memory = []
        self.batch_size = 64
        self.update_counter = 0
        self.target_update_freq = 100  # nombre d'updates avant copie vers target

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        q_values = self.q_net(states)
        next_q_values = self.target_q_net(next_states)

        target = q_values.clone().detach()
        for i in range(self.batch_size):
            target[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i]).item()

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def act(self, state):
        with torch.no_grad():
            q_values = self.q_net(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()

# --- Fonctions CVRP ---
def evaluate(solution, coords, demands, capacity):
    cost, load, prev = 0, 0, 0
    for node in solution + [0]:
        if load + demands[node] > capacity:
            cost += np.linalg.norm(coords[prev] - coords[0])
            prev = 0
            load = 0
        cost += np.linalg.norm(coords[prev] - coords[node])
        load += demands[node]
        prev = node
    cost += np.linalg.norm(coords[prev] - coords[0])
    return cost

def mutate(solution, dqn):
    new_solution = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    state = np.array([i / len(solution), j / len(solution)])
    action = dqn.act(state)
    if action == 0:
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    # Sinon ne rien faire (mutation "neutre")
    return new_solution

# --- PMX (Partially Mapped Crossover) ---
def pmx_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size

    # Copier la section du parent 1
    child[a:b] = p1[a:b]

    # Pour les positions dans la tranche du parent 2
    for i in range(a, b):
        if p2[i] not in child:
            pos = i
            val = p2[i]
            while True:
                val_in_p1 = p1[pos]
                if val_in_p1 not in child:
                    pos = p2.index(val_in_p1)
                else:
                    break
            child[pos] = val

    # Remplir les positions restantes avec le parent 2
    for i in range(size):
        if child[i] == -1:
            child[i] = p2[i]

    return child

def extract_routes(solution, coords, demands, capacity):
    routes, route, load = [], [], 0
    for node in solution:
        if load + demands[node] > capacity:
            routes.append(route)
            route = []
            load = 0
        route.append(node)
        load += demands[node]
    if route:
        routes.append(route)
    return routes

def plot_routes(routes, coords):
    depot = coords[0]
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab20', len(routes))

    for i, route in enumerate(routes):
        route_coords = [depot] + [coords[node] for node in route] + [depot]
        xs, ys = zip(*route_coords)
        plt.plot(xs, ys, marker='o', label=f'Tournée {i+1}', color=colors(i))

    plt.scatter(depot[0], depot[1], c='red', s=100, label='Dépôt')
    plt.title("Visualisation des tournées du véhicule (A-n32-k5) + DQN")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def genetic_algorithm(dqn, coords, demands, capacity, n_gen=200, pop_size=50):
    clients = list(range(1, len(coords)))
    population = [random.sample(clients, len(clients)) for _ in range(pop_size)]
    best_solution = None
    best_cost = float('inf')

    convergence = []

    for gen in range(n_gen):
        population.sort(key=lambda sol: evaluate(sol, coords, demands, capacity))
        current_best_cost = evaluate(population[0], coords, demands, capacity)
        convergence.append(current_best_cost)

        new_pop = population[:10]  # élitisme : garder les 10 meilleurs

        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:25], 2)  # sélection restreinte
            child = pmx_crossover(p1, p2)  # PMX au lieu du crossover simple
            child = mutate(child, dqn)
            new_pop.append(child)

        population = new_pop

        # Update best solution
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_solution = population[0]

    return best_solution, best_cost, convergence

# --- Chargement de l'instance ---
coords, demands, capacity = load_cvrp_instance("A-n32-k5.vrp")

# --- Pré-entraînement du DQN ---
dqn = DQN(input_dim=2, output_dim=2)
for _ in range(1000):
    s = np.random.rand(2)
    a = np.random.randint(0, 2)
    r = np.random.rand()
    s2 = np.random.rand(2)
    dqn.memory.append((s, a, r, s2))
    dqn.update()

# --- Exécution unique avec suivi de convergence ---
best_solution, best_cost, convergence = genetic_algorithm(dqn, coords, demands, capacity)

print(f"✅ Meilleure solution obtenue avec un coût de {best_cost:.2f}")

# --- Affichage de la meilleure tournée ---
routes = extract_routes(best_solution, coords, demands, capacity)
plot_routes(routes, coords)

# --- Courbe de convergence ---
plt.figure(figsize=(10,5))
plt.plot(convergence, label='Coût de la meilleure solution par génération')
plt.title("Courbe de convergence (AG + DQN)")
plt.xlabel("Génération")
plt.ylabel("Coût total")
plt.grid(True)
plt.legend()
plt.show()

# --- Export CSV ---
df = pd.DataFrame({"Best_Solution": best_solution})
df.to_csv("best_solution_A-n32-k5_pmx.csv", index=False)

from google.colab import files
uploaded = files.upload()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import pandas as pd

# --- Chargement de l'instance CVRP ---
def load_cvrp_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    coords, demands = [], []
    capacity = 0
    reading_coords, reading_demands = False, False

    for line in lines:
        if "CAPACITY" in line:
            capacity = int(line.split()[-1])
        elif "NODE_COORD_SECTION" in line:
            reading_coords = True
        elif "DEMAND_SECTION" in line:
            reading_coords = False
            reading_demands = True
        elif "DEPOT_SECTION" in line:
            break
        elif reading_coords:
            parts = line.strip().split()
            coords.append((float(parts[1]), float(parts[2])))
        elif reading_demands:
            parts = line.strip().split()
            demands.append(int(parts[1]))

    return np.array(coords), np.array(demands), capacity

# --- Réseau de neurones DQN ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DQN:
    def __init__(self, input_dim, output_dim, gamma=0.95, lr=1e-3):
        self.q_net = QNetwork(input_dim, output_dim)
        self.target_q_net = QNetwork(input_dim, output_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # Synchronisation initiale
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.memory = []
        self.batch_size = 64

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        q_values = self.q_net(states)
        next_q_values = self.target_q_net(next_states)

        target = q_values.clone().detach()
        for i in range(self.batch_size):
            target[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i]).item()

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, state):
        with torch.no_grad():
            q_values = self.q_net(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()

# --- PMX crossover ---
def pmx_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size

    a, b = sorted(random.sample(range(size), 2))
    child[a:b+1] = parent1[a:b+1]

    for i in range(a, b+1):
        gene = parent2[i]
        if gene not in child:
            pos = i
            while True:
                gene_in_parent1 = parent1[pos]
                if gene_in_parent1 in parent2[a:b+1]:
                    pos = parent2.index(gene_in_parent1)
                else:
                    break
            child[pos] = gene

    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]

    return child

# --- Fonctions CVRP ---
def evaluate(solution, coords, demands, capacity):
    cost, load, prev = 0, 0, 0
    for node in solution + [0]:
        if load + demands[node] > capacity:
            cost += np.linalg.norm(coords[prev] - coords[0])
            prev = 0
            load = 0
        cost += np.linalg.norm(coords[prev] - coords[node])
        load += demands[node]
        prev = node
    cost += np.linalg.norm(coords[prev] - coords[0])
    return cost

def mutate(solution, dqn):
    new_solution = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    state = np.array([i / len(solution), j / len(solution)])
    action = dqn.act(state)
    if action == 0:
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    # tu peux rajouter d'autres actions si besoin
    return new_solution

def extract_routes(solution, coords, demands, capacity):
    routes, route, load = [], [], 0
    for node in solution:
        if load + demands[node] > capacity:
            routes.append(route)
            route = []
            load = 0
        route.append(node)
        load += demands[node]
    if route:
        routes.append(route)
    return routes

def plot_routes(routes, coords):
    depot = coords[0]
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab20', len(routes))

    for i, route in enumerate(routes):
        route_coords = [depot] + [coords[node] for node in route] + [depot]
        xs, ys = zip(*route_coords)
        plt.plot(xs, ys, marker='o', label=f'Tournée {i+1}', color=colors(i))

    plt.scatter(depot[0], depot[1], c='red', s=100, label='Dépôt')
    plt.title("Visualisation des tournées du véhicule (P-n16-k8)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Algorithme génétique ---
def genetic_algorithm(dqn, coords, demands, capacity, n_gen=200, pop_size=50):
    clients = list(range(1, len(coords)))
    population = [random.sample(clients, len(clients)) for _ in range(pop_size)]
    best_solution = None
    best_cost = float('inf')
    history_costs = []

    for gen in range(n_gen):
        population.sort(key=lambda sol: evaluate(sol, coords, demands, capacity))
        current_best_cost = evaluate(population[0], coords, demands, capacity)
        history_costs.append(current_best_cost)

        new_pop = population[:10]
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:25], 2)
            child = pmx_crossover(p1, p2)
            child = mutate(child, dqn)
            new_pop.append(child)
        population = new_pop

        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_solution = population[0]

    return best_solution, best_cost, history_costs

# --- Chargement instance P-n16-k8 ---
coords, demands, capacity = load_cvrp_instance("P-n16-k8.vrp")

# --- Pré-entraînement du DQN ---
dqn = DQN(input_dim=2, output_dim=2)
for _ in range(1000):
    s = np.random.rand(2)
    a = np.random.randint(0, 2)
    r = np.random.rand()
    s2 = np.random.rand(2)
    dqn.memory.append((s, a, r, s2))
    dqn.update()

# --- Exécution multiple (30 runs) ---
results = []
best_overall_cost = float('inf')
best_overall_solution = None
all_histories = []

for run in range(30):
    sol, cost, history = genetic_algorithm(dqn, coords, demands, capacity)
    results.append(cost)
    all_histories.append(history)
    if cost < best_overall_cost:
        best_overall_cost = cost
        best_overall_solution = sol
    print(f"Run {run+1}: Cost = {cost:.2f}")

# --- Affichage de la meilleure tournée ---
routes = extract_routes(best_overall_solution, coords, demands, capacity)
plot_routes(routes, coords)

# --- Courbe de convergence (exemple run le meilleur) ---
plt.figure(figsize=(10,5))
plt.plot(all_histories[results.index(best_overall_cost)], label="Coût")
plt.title("Courbe de convergence du meilleur run (AG + DQN avec PMX)")
plt.xlabel("Génération")
plt.ylabel("Coût total")
plt.grid(True)
plt.legend()
plt.show()

# --- Graphe des coûts sur les 30 runs ---
plt.figure(figsize=(10,5))
plt.plot(results, marker='o')
plt.title("Coûts obtenus sur les 30 runs (AG + DQN avec PMX)")
plt.xlabel("Run")
plt.ylabel("Coût total")
plt.grid(True)
plt.show()

# --- Export CSV ---
df = pd.DataFrame({"Best_Solution": best_overall_solution})
df.to_csv("best_solution_P-n16-k8.csv", index=False)
print(f"✅ Meilleure solution enregistrée avec un coût de {best_overall_cost:.2f}")

from google.colab import files
uploaded = files.upload()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import pandas as pd

# --- Chargement de l'instance CVRP ---
def load_cvrp_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    coords, demands = [], []
    capacity = 0
    reading_coords, reading_demands = False, False

    for line in lines:
        if "CAPACITY" in line:
            capacity = int(line.split()[-1])
        elif "NODE_COORD_SECTION" in line:
            reading_coords = True
        elif "DEMAND_SECTION" in line:
            reading_coords = False
            reading_demands = True
        elif "DEPOT_SECTION" in line:
            break
        elif reading_coords:
            parts = line.strip().split()
            coords.append((float(parts[1]), float(parts[2])))
        elif reading_demands:
            parts = line.strip().split()
            demands.append(int(parts[1]))

    return np.array(coords), np.array(demands), capacity

# --- Réseau de neurones DQN ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DQN:
    def __init__(self, input_dim, output_dim, gamma=0.99, lr=1e-3):
        self.q_net = QNetwork(input_dim, output_dim)
        self.target_q_net = QNetwork(input_dim, output_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # Synchronisation initiale
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.memory = []
        self.batch_size = 64

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        q_values = self.q_net(states)
        next_q_values = self.target_q_net(next_states)

        target = q_values.clone().detach()
        for i in range(self.batch_size):
            target[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i]).item()

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, state):
        with torch.no_grad():
            q_values = self.q_net(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()

# --- PMX crossover ---
def pmx_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size

    a, b = sorted(random.sample(range(size), 2))
    child[a:b+1] = parent1[a:b+1]

    for i in range(a, b+1):
        gene = parent2[i]
        if gene not in child:
            pos = i
            while True:
                gene_in_parent1 = parent1[pos]
                if gene_in_parent1 in parent2[a:b+1]:
                    pos = parent2.index(gene_in_parent1)
                else:
                    break
            child[pos] = gene

    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]

    return child

# --- Fonctions CVRP ---
def evaluate(solution, coords, demands, capacity):
    cost, load, prev = 0, 0, 0
    for node in solution + [0]:
        if load + demands[node] > capacity:
            cost += np.linalg.norm(coords[prev] - coords[0])
            prev = 0
            load = 0
        cost += np.linalg.norm(coords[prev] - coords[node])
        load += demands[node]
        prev = node
    cost += np.linalg.norm(coords[prev] - coords[0])
    return cost

def mutate(solution, dqn):
    new_solution = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    state = np.array([i / len(solution), j / len(solution)])
    action = dqn.act(state)
    if action == 0:
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    # tu peux étendre les actions ici
    return new_solution

def extract_routes(solution, coords, demands, capacity):
    routes, route, load = [], [], 0
    for node in solution:
        if load + demands[node] > capacity:
            routes.append(route)
            route = []
            load = 0
        route.append(node)
        load += demands[node]
    if route:
        routes.append(route)
    return routes

def plot_routes(routes, coords):
    depot = coords[0]
    plt.figure(figsize=(12, 9))
    colors = plt.cm.get_cmap('tab20', len(routes))

    for i, route in enumerate(routes):
        route_coords = [depot] + [coords[node] for node in route] + [depot]
        xs, ys = zip(*route_coords)
        plt.plot(xs, ys, marker='o', label=f'Tournée {i+1}', color=colors(i))

    plt.scatter(depot[0], depot[1], c='red', s=120, label='Dépôt')
    plt.title("Visualisation des tournées du véhicule (B-n38-k6)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Algorithme génétique ---
def genetic_algorithm(dqn, coords, demands, capacity, n_gen=300, pop_size=80):
    clients = list(range(1, len(coords)))
    population = [random.sample(clients, len(clients)) for _ in range(pop_size)]
    best_solution = None
    best_cost = float('inf')
    history_costs = []

    for gen in range(n_gen):
        population.sort(key=lambda sol: evaluate(sol, coords, demands, capacity))
        current_best_cost = evaluate(population[0], coords, demands, capacity)
        history_costs.append(current_best_cost)

        new_pop = population[:15]
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:30], 2)
            child = pmx_crossover(p1, p2)
            child = mutate(child, dqn)
            new_pop.append(child)
        population = new_pop

        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_solution = population[0]

    return best_solution, best_cost, history_costs

# --- Chargement instance B-n38-k6 ---
coords, demands, capacity = load_cvrp_instance("B-n38-k6.vrp")

# --- Pré-entraînement du DQN ---
dqn = DQN(input_dim=2, output_dim=2)
for _ in range(1500):
    s = np.random.rand(2)
    a = np.random.randint(0, 2)
    r = np.random.rand()
    s2 = np.random.rand(2)
    dqn.memory.append((s, a, r, s2))
    dqn.update()

# --- Exécution multiple (30 runs) ---
results = []
best_overall_cost = float('inf')
best_overall_solution = None
all_histories = []

for run in range(30):
    sol, cost, history = genetic_algorithm(dqn, coords, demands, capacity)
    results.append(cost)
    all_histories.append(history)
    if cost < best_overall_cost:
        best_overall_cost = cost
        best_overall_solution = sol
    print(f"Run {run+1}: Coût = {cost:.2f}")

# --- Affichage de la meilleure solution ---
routes = extract_routes(best_overall_solution, coords, demands, capacity)
plot_routes(routes, coords)

# --- Courbe de convergence du meilleur run ---
plt.figure(figsize=(10,5))
plt.plot(all_histories[results.index(best_overall_cost)], label="Coût")
plt.title("Courbe de convergence du meilleur run (AG + DQN avec PMX)")
plt.xlabel("Génération")
plt.ylabel("Coût total")
plt.grid(True)
plt.legend()
plt.show()

# --- Graphe des coûts sur les 30 runs ---
plt.figure(figsize=(10,5))
plt.plot(results, marker='o')
plt.title("Coûts obtenus sur les 30 runs (AG + DQN avec PMX)")
plt.xlabel("Run")
plt.ylabel("Coût total")
plt.grid(True)
plt.show()

# --- Export CSV ---
df = pd.DataFrame({"Meilleure_Solution": best_overall_solution})
df.to_csv("best_solution_B-n38-k6.csv", index=False)
print(f"✅ Meilleure solution enregistrée avec un coût de {best_overall_cost:.2f}")
