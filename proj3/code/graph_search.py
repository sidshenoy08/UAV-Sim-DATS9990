from heapq import heappush, heappop  # Recommended.
import numpy as np

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])

def chebyshev_dist(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]), abs(p1[2] - p2[2]))

def optimize_path(path):

    path = np.delete(path, [1, -2], axis=0)

    optimized_path = [path[0]] # the first point

    counter = 0

    for i in range(1, len(path) - 1):
        prev_vec = np.subtract(path[i], path[i - 1])
        next_vec = np.subtract(path[i + 1], path[i])

        counter += 1

        if not np.array_equal(prev_vec, next_vec) or counter == 2:
            optimized_path.append(path[i])
            counter = 0

    optimized_path.append(path[-1])  # the last point

    return optimized_path

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    if occ_map.is_occupied_index(goal_index):
        print("The goal is occupied and unreachable")
        return None, 0

    directions = [(nx, ny, nz)
                    for nx in [-1, 0, 1]
                    for ny in [-1, 0, 1]
                    for nz in [-1, 0, 1]
                    if (nx, ny, nz) != (0, 0, 0)]

    # Priority queue
    queue = []
    heappush(queue, (0, start_index))  # (cost, index)

    cost_so_far = {start_index: 0}
    came_from = {}
    nodes_expanded = 0

    while queue: # while queue not empty
        current_cost, current_index = heappop(queue)
        nodes_expanded += 1

        # If reached goal
        if current_index == goal_index:

            path = []
            current_index = came_from[current_index]

            while current_index != start_index:
                path.append(current_index)
                current_index = came_from[current_index]

            path.reverse()

            # path = optimize_path(path) # optimize path in grid coord

            path = [occ_map.index_to_metric_center(point) for point in path] # convert to world coord

            path.insert(0, start)
            path.append(goal)

            return np.array(path), nodes_expanded

        # Explore neighbors
        for direction in directions:
            neighbor_index = tuple(np.add(current_index, direction))

            if not occ_map.is_occupied_index(neighbor_index):
                cost = manhattan_dist(current_index, neighbor_index)
                new_cost = current_cost + cost

                if neighbor_index not in cost_so_far or new_cost < cost_so_far[neighbor_index]:
                    cost_so_far[neighbor_index] = new_cost
                    came_from[neighbor_index] = current_index

                    if astar:
                        heuristic = chebyshev_dist(goal_index, neighbor_index)
                        priority = new_cost + heuristic
                    else:
                        priority = new_cost

                    heappush(queue, (priority, neighbor_index))

    # If no path is found, return None
    print("No path is found")
    return None, nodes_expanded