import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

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
    points = np.linspace(point1, point2, num_samples)

    if obstacle["type"] == "cuboid":
        min_corner, max_corner = obstacle["min_corner"], obstacle["max_corner"]
        for point in points:
            if np.all(point >= min_corner) and np.all(point <= max_corner):
                return True
    elif obstacle["type"] == "cylinder":
        center, radius, height = obstacle["center"], obstacle["radius"], obstacle["height"]
        for point in points:
            # Check horizontal distance from cylinder axis
            dist = np.linalg.norm(point[:2] - center[:2])
            if dist <= radius and 0 <= point[2] <= height:
                return True

    return False


def differential_evolution(objective, bounds, pop_size=20, max_gen=200, F=0.8, CR=0.7):
    num_points = 10
    dimension = 3
    pop = np.random.rand(pop_size, num_points, dimension)
    pop[:, 0, :] = start
    pop[:, -1, :] = end

    for i in range(1, num_points - 1):
        pop[:, i, :] = bounds[:, 0] + pop[:, i, :] * (bounds[:, 1] - bounds[:, 0])

    fitness = np.asarray([objective(ind)[0] for ind in pop])
    best_idx = np.argmin(fitness)
    best_cost = fitness[best_idx]

    for generation in range(max_gen):
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
            cross_points = np.random.rand(num_points, dimension) < CR
            trial = np.where(cross_points, mutant, pop[i])
            f = objective(trial)[0]
            if f < fitness[i]:
                fitness[i] = f
                pop[i] = trial
                if f < best_cost:
                    best_cost = f
                    best_idx = i

    return pop[best_idx]

def adaptive_differential_evolution(objective, bounds, pop_size=20, max_gen=200, tol=1e-6):
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

    def adjust_parameters():
        return 0.5 + 0.5 * np.random.rand(), 0.9 * np.random.rand()

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

        F, CR = adjust_parameters()

        if best_cost < tol:
            break

    return pop[best_idx], collision_counts[best_idx]

def jade(objective, bounds, pop_size=20, max_gen=200, tol=1e-6):
    num_points = 10
    dimension = 3
    c = 0.1  # Learning rate
    p = 0.2  # Increase the top p% individuals used for mutation
    mu_F = 0.5
    mu_CR = 0.5

    pop = np.random.rand(pop_size, num_points, dimension)
    pop[:, 0, :] = start
    pop[:, -1, :] = end

    for i in range(1, num_points - 1):
        pop[:, i, :] = bounds[:, 0] + pop[:, i, :] * (bounds[:, 1] - bounds[:, 0])

    fitness = np.asarray([objective(ind)[0] for ind in pop])
    best_idx = np.argmin(fitness)
    best_cost = fitness[best_idx]

    for generation in range(max_gen):
        F = np.clip(np.random.normal(mu_F, 0.1), 0, 1)
        CR = np.clip(np.random.normal(mu_CR, 0.1), 0, 1)

        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            p_best_idx = np.random.randint(0, int(pop_size * p))
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(pop[i] + F * (pop[p_best_idx] - pop[i]) + F * (b - c), bounds[:, 0], bounds[:, 1])
            cross_points = np.random.rand(num_points, dimension) < CR
            trial = np.where(cross_points, mutant, pop[i])
            f = objective(trial)[0]
            if f < fitness[i]:
                fitness[i] = f
                pop[i] = trial
                if f < best_cost:
                    best_cost = f
                    best_idx = i

        # Update mu_F and mu_CR using c value
        mu_F = (1 - c) * mu_F + c * F
        mu_CR = (1 - c) * mu_CR + c * CR

        if best_cost < tol:
            break

    return pop[best_idx]


def shade(objective, bounds, pop_size=20, max_gen=200, H=10, tol=1e-6):
    num_points = 10
    dimension = 3
    archive = []
    mem_F = [0.5] * H
    mem_CR = [0.5] * H
    k = 0

    pop = np.random.rand(pop_size, num_points, dimension)
    pop[:, 0, :] = start
    pop[:, -1, :] = end

    for i in range(1, num_points - 1):
        pop[:, i, :] = bounds[:, 0] + pop[:, i, :] * (bounds[:, 1] - bounds[:, 0])

    fitness = np.asarray([objective(ind)[0] for ind in pop])
    best_idx = np.argmin(fitness)
    best_cost = fitness[best_idx]

    for generation in range(max_gen):
        F = np.clip(np.random.normal(mem_F[k % H], 0.1), 0, 1)
        CR = np.clip(np.random.normal(mem_CR[k % H], 0.1), 0, 1)

        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
            cross_points = np.random.rand(num_points, dimension) < CR
            trial = np.where(cross_points, mutant, pop[i])
            f = objective(trial)[0]
            if f < fitness[i]:
                fitness[i] = f
                pop[i] = trial
                if f < best_cost:
                    best_cost = f
                    best_idx = i

                archive.append(pop[i])
                if len(archive) > pop_size:
                    archive.pop(0)

                mem_F[k % H] = (mem_F[k % H] * len(archive) + F) / (len(archive) + 1)
                mem_CR[k % H] = (mem_CR[k % H] * len(archive) + CR) / (len(archive) + 1)

        k += 1
        if best_cost < tol:
            break

    return pop[best_idx]

start = np.array([0, 0, 0])
end = np.array([10, 10, 10])

# Define multiple obstacles as cuboids with z=0 for the bottom
obstacles = [
    {"type": "cuboid", "min_corner": np.array([2, 2, 0]), "max_corner": np.array([3, 4, 9])},
    {"type": "cuboid", "min_corner": np.array([5, 5, 0]), "max_corner": np.array([6, 6, 9])},
    {"type": "cylinder", "center": np.array([7, 7, 0]), "radius": 1, "height": 9},
    {"type": "cuboid", "min_corner": np.array([8, 8, 0]), "max_corner": np.array([9, 9, 9])},
    {"type": "cuboid", "min_corner": np.array([1, 7, 0]), "max_corner": np.array([2, 8, 9])},
    {"type": "cylinder", "center": np.array([7, 1, 0]), "radius": 1, "height": 9},
    {"type": "cuboid", "min_corner": np.array([3, 6, 0]), "max_corner": np.array([5, 7, 9])},
    {"type": "cylinder", "center": np.array([6, 3, 0]), "radius": 1, "height": 9}
]

bounds = np.array([[0, 10], [0, 10], [0, 10]])

# Run DE algorithm
start_time = time.time()
best_path_de = differential_evolution(objective_function, bounds, max_gen=200)
de_time = time.time() - start_time
_, de_collisions = objective_function(best_path_de)

# Run ADE algorithm
start_time = time.time()
best_path_ade, ade_collisions = adaptive_differential_evolution(objective_function, bounds, max_gen=200)
ade_time = time.time() - start_time

# Run JADE algorithm
start_time = time.time()
best_path_jade = jade(objective_function, bounds, max_gen=200)
jade_time = time.time() - start_time
_, jade_collisions = objective_function(best_path_jade)

# Run SHADE algorithm
start_time = time.time()
best_path_shade = shade(objective_function, bounds, max_gen=200)
shade_time = time.time() - start_time
_, shade_collisions = objective_function(best_path_shade)

# Calculate path lengths
de_length = np.sum(np.linalg.norm(np.diff(best_path_de, axis=0), axis=1))
ade_length = np.sum(np.linalg.norm(np.diff(best_path_ade, axis=0), axis=1))
jade_length = np.sum(np.linalg.norm(np.diff(best_path_jade, axis=0), axis=1))
shade_length = np.sum(np.linalg.norm(np.diff(best_path_shade, axis=0), axis=1))

# Output results
print(f"DE Algorithm: Path Length = {de_length}, Time = {de_time} seconds, Collisions = {de_collisions}")
print(f"ADE Algorithm: Path Length = {ade_length}, Time = {ade_time} seconds, Collisions = {ade_collisions}")
print(f"JADE Algorithm: Path Length = {jade_length}, Time = {jade_time} seconds, Collisions = {jade_collisions}")
print(f"SHADE Algorithm: Path Length = {shade_length}, Time = {shade_time} seconds, Collisions = {shade_collisions}")

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(start[0], start[1], start[2], color='green', s=100, label='Start')
ax.scatter(end[0], end[1], end[2], color='blue', s=100, label='End')
for obstacle in obstacles:
    if obstacle["type"] == "cuboid":
        min_corner, max_corner = obstacle["min_corner"], obstacle["max_corner"]
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
    elif obstacle["type"] == "cylinder":
        center, radius, height = obstacle["center"], obstacle["radius"], obstacle["height"]
        z = np.linspace(0, height, 50)
        theta = np.linspace(0, 2 * np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = center[0] + radius * np.cos(theta_grid)
        y_grid = center[1] + radius * np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color='r', alpha=0.5)

ax.plot(best_path_de[:, 0], best_path_de[:, 1], best_path_de[:, 2], 'bo-', label='DE Path')
ax.plot(best_path_ade[:, 0], best_path_ade[:, 1], best_path_ade[:, 2], 'go-', label='ADE Path')
ax.plot(best_path_jade[:, 0], best_path_jade[:, 1], best_path_jade[:, 2], 'co-', label='JADE Path')
ax.plot(best_path_shade[:, 0], best_path_shade[:, 1], best_path_shade[:, 2], 'mo-', label='SHADE Path')
ax.legend()
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.title('3D Path Planning with DE, ADE, JADE, and SHADE')
plt.show()
