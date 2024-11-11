from imports import *

def simulated_annealing(G, start, end, initial_temp, cooling_rate, max_iter):
    current_temp = initial_temp
    current_path = nx.shortest_path(G, start, end, weight='weight')
    current_cost = sum(G[u][v]['weight'] for u, v in zip(current_path[:-1], current_path[1:]))
    best_path = current_path
    best_cost = current_cost
    for i in range(max_iter):
        middle = random.choice(current_path[1:-1])
        neighbors = list(G.neighbors(middle))
        if neighbors:
            next_hop = random.choice(neighbors)
            new_path = nx.shortest_path(G, start, next_hop, weight='weight') + nx.shortest_path(G, next_hop, end, weight='weight')[1:]
            new_cost = sum(G[u][v]['weight'] for u, v in zip(new_path[:-1], new_path[1:]))
            if new_cost < current_cost or random.uniform(0, 1) < np.exp(-(new_cost - current_cost) / current_temp):
                current_path = new_path
                current_cost = new_cost
                if current_cost < best_cost:
                    best_path = current_path
                    best_cost = current_cost
            current_temp *= cooling_rate
    return best_path, best_cost
