import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors

class MapGen:
    @classmethod
    def generate_main_with_subclusters_map(cls, rows, cols, main_cluster_size_range, subcluster_config):
        grid_map = np.zeros((rows, cols), dtype=int)

        def is_valid_position(x, y):
            return 1 <= x < rows - 1 and 1 <= y < cols - 1

        def grow_cluster(center_x, center_y, cluster_size):
            cluster_cells = [(center_x, center_y)]
            grid_map[center_x, center_y] = 1
            steps = 0
            max_steps = 300

            while len(cluster_cells) < cluster_size and steps < max_steps:
                current_x, current_y = random.choice(cluster_cells)
                directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,-1), (-1,1), (1,-1)]
                random.shuffle(directions)
                expanded = False
                for dx, dy in directions:
                    new_x, new_y = current_x + dx, current_y + dy
                    if is_valid_position(new_x, new_y) and grid_map[new_x, new_y] == 0:
                        grid_map[new_x, new_y] = 1
                        cluster_cells.append((new_x, new_y))
                        expanded = True
                        break
                if not expanded:
                    steps += 1  # only increment if unable to grow
            pass  # cluster grown silently

        main_x, main_y = np.random.randint(5, rows - 5), np.random.randint(5, cols - 5)
        main_cluster_size = np.random.randint(*main_cluster_size_range)
        grow_cluster(main_x, main_y, main_cluster_size)
        main_center = (main_x, main_y)


        num_subclusters = np.random.randint(*subcluster_config["num_range"])
        subcluster_centers = []
        sub_attempts = 0
        max_subcluster_attempts = 300

        while len(subcluster_centers) < num_subclusters and sub_attempts < max_subcluster_attempts:
            angle = random.uniform(0, 2 * np.pi)
            distance = random.randint(*subcluster_config["distance_from_main"])
            sub_x = int(main_center[0] + distance * np.cos(angle))
            sub_y = int(main_center[1] + distance * np.sin(angle))

            if not is_valid_position(sub_x, sub_y):
                sub_attempts += 1
                continue
            if any(np.linalg.norm(np.array((sub_x, sub_y)) - np.array(c)) < subcluster_config["distance_between_subclusters"]
                   for c in [main_center] + subcluster_centers):
                sub_attempts += 1
                continue

            subcluster_size = np.random.randint(*subcluster_config["size_range"])
            grow_cluster(sub_x, sub_y, subcluster_size)
            subcluster_centers.append((sub_x, sub_y))
            sub_attempts += 1


        def grow_obstacle_cluster(start_x, start_y, max_size):
            frontier = [(start_x, start_y)]
            grid_map[start_x, start_y] = 2
            count = 1
            steps = 0
            max_steps = 100

            while count < max_size and steps < max_steps and frontier:
                x, y = random.choice(frontier)
                directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
                random.shuffle(directions)
                expanded = False
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if is_valid_position(nx, ny) and grid_map[nx, ny] == 0:
                        grid_map[nx, ny] = 2
                        frontier.append((nx, ny))
                        count += 1
                        expanded = True
                        break
                if not expanded:
                    frontier.remove((x, y))
                steps += 1
            return count >= 3

        placed_obstacles = 0
        for i in range(100):
            if placed_obstacles >= 3:
                break
            ox = np.random.randint(1, rows - 1)
            oy = np.random.randint(1, cols - 1)
            if grid_map[ox, oy] == 0:
                if grow_obstacle_cluster(ox, oy, 7):
                    placed_obstacles += 1
        return grid_map


# Test + visualize
if __name__ == "__main__":
    rows, cols = 35, 35
    main_cluster_size_range = (30, 60)
    subcluster_config = {
        "num_range": (4, 8),
        "size_range": (5, 15),
        "distance_from_main": (9, 10),
        "distance_between_subclusters": 8
    }

    map_generator = MapGen()
    generated_map = map_generator.generate_main_with_subclusters_map(
        rows, cols, main_cluster_size_range, subcluster_config
    )

    cmap = mcolors.ListedColormap(['white', 'red', 'black'])  # 0 = white, 1 = red, 2 = black
    bounds = [0, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    print(generated_map)
    plt.figure(figsize=(6, 6))
    plt.imshow(generated_map, cmap=cmap, norm=norm, origin="upper")
    plt.title("Map with Chemical Spills (Red) and Obstacles (Black)")
    plt.grid(True, color='lightgray', linewidth=0.5)
    plt.xticks([]); plt.yticks([])
    plt.show()
