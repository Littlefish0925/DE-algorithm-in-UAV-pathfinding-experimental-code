import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import heapq

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

def check_interpath_collision(path1, path2, num_samples=10):
    for i in range(len(path1) - 1):
        for j in range(len(path2) - 1):
            # Create a dictionary to represent the path segment as an obstacle
            segment_obstacle = {"type": "cuboid", "min_corner": np.minimum(path2[j], path2[j + 1]), "max_corner": np.maximum(path2[j], path2[j + 1])}
            if check_collision(path1[i], path1[i + 1], segment_obstacle, num_samples=num_samples):
                return True
    return False

def shade(objective, bounds, start, end, other_paths, pop_size=20, max_gen=200, H=10, tol=1e-6):
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

    mem_F = np.full(H, 0.5)
    mem_CR = np.full(H, 0.5)
    archive = []
    k = 0

    for generation in range(max_gen):
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            p_best = pop[np.random.choice(np.argsort(fitness)[:max(1, int(pop_size * 0.1))])]
            x_best = pop[best_idx]
            r1, r2 = pop[np.random.choice(idxs, 2, replace=False)]
            F = np.clip(np.random.normal(mem_F[k % H], 0.1), 0, 1)
            CR = np.clip(np.random.normal(mem_CR[k % H], 0.1), 0, 1)
            mutant = np.clip(pop[i] + F * (p_best - pop[i]) + F * (r1 - r2), bounds[:, 0], bounds[:, 1])
            cross_points = np.random.rand(num_points, dimension) < CR
            trial = np.where(cross_points, mutant, pop[i])
            f, collision_count = objective(trial)
            if f < fitness[i] and not any(check_interpath_collision(trial, other_path) for other_path in other_paths):
                fitness[i] = f
                collision_counts[i] = collision_count
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

# Define start and end points
start1 = np.array([0, 0, 0])
start2 = np.array([0, 10, 0])
start3 = np.array([10, 0, 0])
end1 = np.array([10, 10, 10])
end2 = np.array([10, 5, 10])
end3 = np.array([0, 10, 10])

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

# Run SHADE algorithm for each path
start_time = time.time()
path1 = shade(objective_function, bounds, start1, end1, [])
path2 = shade(objective_function, bounds, start2, end2, [path1])
path3 = shade(objective_function, bounds, start3, end3, [path1, path2])
shade_time = time.time() - start_time

# Calculate path lengths
path1_length = np.sum(np.linalg.norm(np.diff(path1, axis=0), axis=1))
path2_length = np.sum(np.linalg.norm(np.diff(path2, axis=0), axis=1))
path3_length = np.sum(np.linalg.norm(np.diff(path3, axis=0), axis=1))

# Calculate collisions
_, path1_collisions = objective_function(path1)
_, path2_collisions = objective_function(path2)
_, path3_collisions = objective_function(path3)

# Output results
print(f"Path 1: Length = {path1_length}, Collisions = {path1_collisions}")
print(f"Path 2: Length = {path2_length}, Collisions = {path2_collisions}")
print(f"Path 3: Length = {path3_length}, Collisions = {path3_collisions}")
print(f"Total Time: {shade_time} seconds")

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(start1[0], start1[1], start1[2], color='green', s=100, label='Start 1')
ax.scatter(start2[0], start2[1], start2[2], color='blue', s=100, label='Start 2')
ax.scatter(start3[0], start3[1], start3[2], color='purple', s=100, label='Start 3')
ax.scatter(end1[0], end1[1], end1[2], color='red', s=100, label='End1')
ax.scatter(end2[0], end2[1], end2[2], color='gray', s=100, label='End2')
ax.scatter(end3[0], end3[1], end3[2], color='black', s=100, label='End3')
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

ax.plot(path1[:, 0], path1[:, 1], path1[:, 2], 'bo-', label='Path 1')
ax.plot(path2[:, 0], path2[:, 1], path2[:, 2], 'go-', label='Path 2')
ax.plot(path3[:, 0], path3[:, 1], path3[:, 2], 'mo-', label='Path 3')
ax.legend()
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.title('3D Path Planning with SHADE for Multiple Paths')
plt.show()