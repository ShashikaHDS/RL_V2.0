import numpy as np
import matplotlib.pyplot as plt
import random

class MapGen:
    @classmethod
    def generate_main_with_subclusters_map(cls, rows, cols, main_cluster_size_range, subcluster_config):
        """
        Generate a map with one large obstacle cluster and multiple surrounding smaller clusters.

        Args:
            rows (int): Number of rows in the map.
            cols (int): Number of columns in the map.
            main_cluster_size_range (tuple): (min_size, max_size) for the main cluster.
            subcluster_config (dict): Contains:
                - num_range (tuple): Range of number of subclusters.
                - size_range (tuple): Size range of each subcluster.
                - distance_from_main (tuple): Range of distance from main cluster.
                - distance_between_subclusters (int): Minimum distance between subclusters.

        Returns:
            np.ndarray: A 2D map grid with obstacles marked as 1.
        """
        grid_map = np.zeros((rows, cols), dtype=int)
        occupied_cells = set()

        def is_valid_position(x, y):
            return 1 <= x < rows - 1 and 1 <= y < cols - 1

        def is_far_enough(new_center, centers, min_distance):
            for center in centers:
                if np.linalg.norm(np.array(new_center) - np.array(center)) < min_distance:
                    return False
            return True

        def grow_cluster(center_x, center_y, cluster_size):
            cluster_cells = [(center_x, center_y)]
            grid_map[center_x, center_y] = 1
            occupied_cells.add((center_x, center_y))

            while len(cluster_cells) < cluster_size:
                current_x, current_y = random.choice(cluster_cells)
                directions = [
                    (0, 1), (0, -1), (1, 0), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)
                ]
                random.shuffle(directions)

                for dx, dy in directions:
                    new_x, new_y = current_x + dx, current_y + dy
                    if is_valid_position(new_x, new_y) and grid_map[new_x, new_y] == 0:
                        grid_map[new_x, new_y] = 1
                        cluster_cells.append((new_x, new_y))
                        occupied_cells.add((new_x, new_y))
                        break

        # Step 1: Generate main cluster
        while True:
            main_x = np.random.randint(5, rows - 5)
            main_y = np.random.randint(5, cols - 5)
            if is_valid_position(main_x, main_y):
                break

        main_cluster_size = np.random.randint(*main_cluster_size_range)
        grow_cluster(main_x, main_y, main_cluster_size)
        main_center = (main_x, main_y)

        # Step 2: Generate subclusters
        num_subclusters = np.random.randint(*subcluster_config["num_range"])
        subcluster_centers = []

        for _ in range(num_subclusters):
            attempts = 0
            while attempts < 100:
                angle = random.uniform(0, 2 * np.pi)
                distance = random.randint(*subcluster_config["distance_from_main"])
                sub_x = int(main_center[0] + distance * np.cos(angle))
                sub_y = int(main_center[1] + distance * np.sin(angle))

                if not is_valid_position(sub_x, sub_y):
                    attempts += 1
                    continue
                if not is_far_enough((sub_x, sub_y), [main_center] + subcluster_centers, subcluster_config["distance_between_subclusters"]):
                    attempts += 1
                    continue

                subcluster_size = np.random.randint(*subcluster_config["size_range"])
                grow_cluster(sub_x, sub_y, subcluster_size)
                subcluster_centers.append((sub_x, sub_y))
                break

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

    plt.imshow(generated_map, cmap="Greys", origin="upper")
    plt.title("Map with Main and Subclusters")
    plt.grid(True, color='lightgray', linewidth=0.5)
    plt.show()
