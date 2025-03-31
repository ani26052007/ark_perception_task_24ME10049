import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from skimage.draw import line

# Load the maze image
maze = cv2.imread('maze.png', cv2.IMREAD_GRAYSCALE)

# Convert the image to binary (black=0, white=255)
_, binary_maze = cv2.threshold(maze, 127, 255, cv2.THRESH_BINARY)

# Define start and goal positions
start = (162, 19)
goal = (446, 302)

# RRT Parameters
max_iterations = 6000  # Maximum number of iterations
step_size = 10         # Step size for tree expansion
goal_sample_rate = 0.1 # 10% chance to sample the goal

# Check if a point is inside free space
def is_free(x, y):
    if 0 <= x < binary_maze.shape[1] and 0 <= y < binary_maze.shape[0]:
        return binary_maze[y, x] == 255  # White (free space)
    return False

# Find the nearest node in the tree
def nearest_node(tree, point):
    return min(tree, key=lambda n: math.dist(n, point))

# Efficient collision check using Bresenham’s Line Algorithm
def is_collision_free(p1, p2):
    rr, cc = line(p1[1], p1[0], p2[1], p2[0])  # Get line pixels
    return all(is_free(x, y) for y, x in zip(rr, cc))

# RRT Algorithm
tree = {start: None}  # Tree represented as {node: parent}
found = False

for _ in range(max_iterations):
    # Random sampling with goal biasing
    if random.random() < goal_sample_rate:
        rand_point = goal
    else:
        rand_point = (random.randint(0, binary_maze.shape[1] - 1), 
                      random.randint(0, binary_maze.shape[0] - 1))
    
    # Find the nearest node
    nearest = nearest_node(tree, rand_point)

    # Compute step direction and adaptive step size
    theta = math.atan2(rand_point[1] - nearest[1], rand_point[0] - nearest[0])
    step_size_adaptive = min(step_size, math.dist(nearest, rand_point))
    new_point = (int(nearest[0] + step_size_adaptive * math.cos(theta)), 
                 int(nearest[1] + step_size_adaptive * math.sin(theta)))

    # Check if new point is valid and collision-free
    if is_free(new_point[0], new_point[1]) and is_collision_free(nearest, new_point):
        tree[new_point] = nearest  # Add to tree

        # Check if goal is reached
        if math.dist(new_point, goal) < step_size:
            tree[goal] = new_point
            found = True
            break

# Traceback the path if found
path = []
if found:
    node = goal
    while node is not None:
        path.append(node)
        node = tree[node]
    path.reverse()

    # Path Smoothing
    smoothed_path = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i:
            if is_collision_free(path[i], path[j]):
                smoothed_path.append(path[j])
                i = j
                break
            j -= 1
    path = smoothed_path

# Plot the results
plt.figure(figsize=(10, 6))
plt.imshow(binary_maze, cmap="gray")
plt.scatter(start[0], start[1], color="blue", label="Start")
plt.scatter(goal[0], goal[1], color="red", label="Goal")

# Draw the RRT tree
for node, parent in tree.items():
    if parent:
        plt.plot([node[0], parent[0]], [node[1], parent[1]], color="yellow", linewidth=0.5)

# Draw the final path if found
if path:
    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, color="orange", linewidth=2, label="Smoothed Path")

plt.legend()
plt.title("RRT Pathfinding in Maze with Enhancements")
plt.show()
