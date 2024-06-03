import time
import numpy as np
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

def heuristic(node, end):
    return np.linalg.norm(node.position - end.position)

def is_collision(point, obstacle):
    min_corner, max_corner = obstacle
    return np.all(point >= min_corner) and np.all(point <= max_corner)

def astar(start, end, obstacles, grid_size):
    start_node = Node(np.array(start))
    end_node = Node(np.array(end))

    open_list = []
    closed_list = set()

    heapq.heappush(open_list, start_node)

    directions = [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]),
                  np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([0, 0, -1])]

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(tuple(current_node.position))

        if np.array_equal(current_node.position, end_node.position):
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        neighbors = [current_node.position + direction * grid_size for direction in directions]

        for next_position in neighbors:
            if tuple(next_position) in closed_list:
                continue

            if any(is_collision(next_position, obstacle) for obstacle in obstacles):
                continue

            neighbor_node = Node(next_position, current_node)
            neighbor_node.g = current_node.g + np.linalg.norm(current_node.position - neighbor_node.position)
            neighbor_node.h = heuristic(neighbor_node, end_node)
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            if not any(np.array_equal(neighbor_node.position, node.position) and neighbor_node.f >= node.f for node in open_list):
                heapq.heappush(open_list, neighbor_node)

    return None

def objective_function(path):
    cost = 0
    collision_penalty = 1000
    collision_count = 0
    for i in range(len(path) - 1):
        cost += np.linalg.norm(path[i + 1] - path[i])
        if np.any([check_collision(path[i], path[i + 1], obstacle) for obstacle in obstacles]):
            cost += collision_penalty
            collision_count += 1
    return cost, collision_count

def check_collision(point1, point2, obstacle, num_samples=10):
    min_corner, max_corner = obstacle
    points = np.linspace(point1, point2, num_samples)
    for point in points:
        if np.all(point >= min_corner) and np.all(point <= max_corner):
            return True
    return False

def adaptive_differential_evolution(objective, bounds, pop_size=20, max_gen=500, tol=1e-6):
    num_points = 10
    dimension = 3
    pop = np.random.rand(pop_size, num_points, dimension)
    pop[:, 0, :] = start
    pop[:, -1, :] = end

    for i in range(1, num_points - 1):
        pop[:, i, :] = bounds[:, 0] + pop[:, i, :] * (bounds[:, 1] - bounds[:, 0])

    fitness = np.zeros(pop_size)
    collision_counts = np.zeros(pop_size)
    for j in range(pop_size):
        fitness[j], collision_counts[j] = objective(pop[j])

    best_idx = np.argmin(fitness)
    best_cost = fitness[best_idx]

    F = 0.5  # Mutation factor
    CR = 0.9  # Crossover probability

    for generation in range(max_gen):
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
            cross_points = np.random.rand(num_points, dimension) < CR
            trial = np.where(cross_points, mutant, pop[i])
            f, collision_count = objective(trial)
            if f < fitness[i]:
                fitness[i] = f
                collision_counts[i] = collision_count
                pop[i] = trial
                if f < best_cost:
                    best_cost = f
                    best_idx = i

        F = 0.5 + 0.5 * np.random.rand()
        CR = 0.9 * np.random.rand()

        if best_cost < tol:
            break

    return pop[best_idx], collision_counts[best_idx]

def genetic_algorithm(objective, bounds, pop_size=20, max_gen=500, mutation_prob=0.1, crossover_prob=0.9, tol=1e-6):
    num_points = 10
    dimension = 3
    pop = np.random.rand(pop_size, num_points, dimension)
    pop[:, 0, :] = start
    pop[:, -1, :] = end

    for i in range(1, num_points - 1):
        pop[:, i, :] = bounds[:, 0] + pop[:, i, :] * (bounds[:, 1] - bounds[:, 0])

    fitness = np.zeros(pop_size)
    collision_counts = np.zeros(pop_size)
    for j in range(pop_size):
        fitness[j], collision_counts[j] = objective(pop[j])

    best_idx = np.argmin(fitness)
    best_cost = fitness[best_idx]

    for generation in range(max_gen):
        new_pop = []
        for i in range(pop_size):
            parents = np.random.choice(pop_size, 2, replace=False)
            parent1, parent2 = pop[parents[0]], pop[parents[1]]
            cross_points = np.random.rand(num_points, dimension) < crossover_prob
            offspring = np.where(cross_points, parent1, parent2)
            if np.random.rand() < mutation_prob:
                mutation = np.random.randint(1, num_points-1)
                offspring[mutation] = bounds[:, 0] + np.random.rand(dimension) * (bounds[:, 1] - bounds[:, 0])
            new_pop.append(offspring)
        pop = np.asarray(new_pop)
        fitness = np.zeros(pop_size)
        collision_counts = np.zeros(pop_size)
        for j in range(pop_size):
            fitness[j], collision_counts[j] = objective(pop[j])

        best_idx = np.argmin(fitness)
        best_cost = fitness[best_idx]

        if best_cost < tol:
            break

    return pop[best_idx], collision_counts[best_idx]

def particle_swarm_optimization(objective, bounds, pop_size=20, max_gen=500, tol=1e-6):
    num_points = 10
    dimension = 3
    w_max = 0.9  # Initial inertia weight
    w_min = 0.4  # Final inertia weight
    c1 = 2.0  # Cognitive constant
    c2 = 2.0  # Social constant
    pop = np.random.rand(pop_size, num_points, dimension)
    pop[:, 0, :] = start
    pop[:, -1, :] = end
    velocity = np.zeros_like(pop)

    for i in range(1, num_points - 1):
        pop[:, i, :] = bounds[:, 0] + pop[:, i, :] * (bounds[:, 1] - bounds[:, 0])

    fitness = np.zeros(pop_size)
    collision_counts = np.zeros(pop_size)
    for j in range(pop_size):
        fitness[j], collision_counts[j] = objective(pop[j])

    p_best = pop.copy()
    p_best_fitness = fitness.copy()
    p_best_collisions = collision_counts.copy()

    g_best_idx = np.argmin(fitness)
    g_best = pop[g_best_idx]
    g_best_fitness = fitness[g_best_idx]
    g_best_collisions = collision_counts[g_best_idx]

    for generation in range(max_gen):
        w = w_max - (w_max - w_min) * (generation / max_gen)  # Linearly decrease inertia weight
        r1, r2 = np.random.rand(2)
        velocity = w * velocity + c1 * r1 * (p_best - pop) + c2 * r2 * (g_best - pop)
        pop = np.clip(pop + velocity, bounds[:, 0], bounds[:, 1])
        fitness = np.zeros(pop_size)
        collision_counts = np.zeros(pop_size)
        for j in range(pop_size):
            fitness[j], collision_counts[j] = objective(pop[j])

        better_idx = fitness < p_best_fitness
        p_best[better_idx] = pop[better_idx]
        p_best_fitness[better_idx] = fitness[better_idx]
        p_best_collisions[better_idx] = collision_counts[better_idx]

        if np.min(fitness) < g_best_fitness:
            g_best_idx = np.argmin(fitness)
            g_best = pop[g_best_idx]
            g_best_fitness = fitness[g_best_idx]
            g_best_collisions = collision_counts[g_best_idx]

        if g_best_fitness < tol:
            break

    return g_best, g_best_collisions

start = np.array([0, 0, 0])
end = np.array([10, 10, 10])

# Define multiple obstacles as cuboids with z=0 for the bottom
obstacles = [
    (np.array([2, 2, 0]), np.array([3, 4, 9])),  # Positioned on the direct line
    (np.array([4, 4, 0]), np.array([5, 5, 9])),  # Positioned on the direct line
    (np.array([6, 6, 0]), np.array([7, 7, 9])),  # Positioned on the direct line
    (np.array([8, 8, 0]), np.array([9, 9, 9])),  # Positioned on the direct line
    (np.array([1, 7, 0]), np.array([2, 8, 9])),
    (np.array([7, 1, 0]), np.array([8, 2, 9])),
    (np.array([3, 6, 0]), np.array([5, 7, 9])),
    (np.array([6, 3, 0]), np.array([7, 4, 9]))
]

bounds = np.array([[0, 10], [0, 10], [0, 10]])

# Run A* algorithm
start_time = time.time()
path_astar = astar(start, end, obstacles, grid_size=1)
astar_time = time.time() - start_time
astar_collisions = 0
if path_astar:
    _, astar_collisions = objective_function(np.array(path_astar))

# Run ADE algorithm
start_time = time.time()
best_path_ade, ade_collisions = adaptive_differential_evolution(objective_function, bounds, max_gen=200)
ade_time = time.time() - start_time

# Run GA algorithm
start_time = time.time()
best_path_ga, ga_collisions = genetic_algorithm(objective_function, bounds, max_gen=200)
ga_time = time.time() - start_time

# Run PSO algorithm
start_time = time.time()
best_path_pso, pso_collisions = particle_swarm_optimization(objective_function, bounds, max_gen=200)
pso_time = time.time() - start_time

# Calculate path lengths
astar_length = np.sum(np.linalg.norm(np.diff(path_astar, axis=0), axis=1)) if path_astar is not None else float('inf')
ade_length = np.sum(np.linalg.norm(np.diff(best_path_ade, axis=0), axis=1))
ga_length = np.sum(np.linalg.norm(np.diff(best_path_ga, axis=0), axis=1))
pso_length = np.sum(np.linalg.norm(np.diff(best_path_pso, axis=0), axis=1))

# Output comparison data
print(f"A* Algorithm: Path Length = {astar_length}, Time = {astar_time} seconds, Collisions = {astar_collisions}")
print(f"DE Algorithm: Path Length = {ade_length}, Time = {ade_time} seconds, Collisions = {ade_collisions}")
print(f"GA Algorithm: Path Length = {ga_length}, Time = {ga_time} seconds, Collisions = {ga_collisions}")
print(f"PSO Algorithm: Path Length = {pso_length}, Time = {pso_time} seconds, Collisions = {pso_collisions}")

# Plot the paths
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(start[0], start[1], start[2], color='green', s=100, label='Start')
ax.scatter(end[0], end[1], end[2], color='blue', s=100, label='End')
for obstacle in obstacles:
    min_corner, max_corner = obstacle
    x = [min_corner[0], max_corner[0], max_corner[0], min_corner[0], min_corner[0]]
    y = [min_corner[1], min_corner[1], max_corner[1], max_corner[1], min_corner[1]]
    z = [min_corner[2], min_corner[2], min_corner[2], min_corner[2], min_corner[2]]
    ax.plot3D(x, y, z, color="r")
    z = [max_corner[2], max_corner[2], max_corner[2], max_corner[2], max_corner[2]]
    ax.plot3D(x, y, z, color="r")
    for i in range(2):
        x = [min_corner[0], max_corner[0], max_corner[0], min_corner[0], min_corner[0]]
        y = [min_corner[1], min_corner[1], max_corner[1], max_corner[1], min_corner[1]]
        z = [min_corner[2], min_corner[2], min_corner[2], min_corner[2], min_corner[2]]
        if i == 1:
            z = [max_corner[2], max_corner[2], max_corner[2], max_corner[2], max_corner[2]]
        ax.plot3D(x, y, z, color="r")

    for i in range(2):
        y = [min_corner[1], min_corner[1], min_corner[1], min_corner[1], min_corner[1]]
        z = [min_corner[2], max_corner[2], max_corner[2], min_corner[2], min_corner[2]]
        if i == 1:
            y = [max_corner[1], max_corner[1], max_corner[1], max_corner[1], max_corner[1]]
        ax.plot3D(x, y, z, color="r")

    for i in range(2):
        x = [min_corner[0], min_corner[0], min_corner[0], min_corner[0], min_corner[0]]
        z = [min_corner[2], max_corner[2], max_corner[2], min_corner[2], min_corner[2]]
        if i == 1:
            x = [max_corner[0], max_corner[0], max_corner[0], max_corner[0], max_corner[0]]
        ax.plot3D(x, y, z, color="r")

if path_astar:
    path_astar = np.array(path_astar)
    ax.plot(path_astar[:, 0], path_astar[:, 1], path_astar[:, 2], 'bo-', label='A* Path')

ax.plot(best_path_ade[:, 0], best_path_ade[:, 1], best_path_ade[:, 2], 'go-', label='SHADE Path')
ax.plot(best_path_ga[:, 0], best_path_ga[:, 1], best_path_ga[:, 2], 'co-', label='GA Path')
ax.plot(best_path_pso[:, 0], best_path_pso[:, 1], best_path_pso[:, 2], 'yo-', label='PSO Path')
ax.legend()
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.title('3D Path Planning: A* vs DE vs GA vs PSO')
plt.show()
