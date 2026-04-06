import gym
import numpy as np
import random
import pygame
import time
import os
from map_gen_v5 import MapGen
from stable_baselines3 import PPO

pygame.init()

# Colors
bg_color = (255, 255, 255)
grid_color = pygame.Color("grey")
obs_color = (0, 255, 0)

normal_area_color=(255, 255, 0)  #exploring
robot_color = (255, 0, 0)

spill_color = (165, 42, 42)  # Brown for uncleaned spill
spill_cleaned_color = (0, 128, 0)  # Green for cleaned spill


class ChemicalClean(gym.Env):
    def __init__(self, grid_map, num_robots, robot_positions, field_of_view, chem_fov):
        super(ChemicalClean, self).__init__()

        # Initialize environment variables
        self.grid_map = np.array(grid_map)
        self.num_robots = num_robots
        self.robot_positions = np.array(robot_positions)
        self.prev_positions = self.robot_positions.copy()
        self.field_of_view = field_of_view
        self.chem_fov = chem_fov
        self.tot_chemical=0
        #self.num_spills = num_spills

        self.step_count = 0
        self.step_count_ = 1
        self.cleaned_cell_count=0
        self.non_spill_exp_count=0

        self.distances_traveled = np.zeros(num_robots)
        self.steps_taken = np.zeros(num_robots, dtype=int)

        self.paths=[[]]
        self.robot_distance=[0]

        # Generate chemical spills in the environment
        #self.grid_map = self.generate_spills(self.grid_map, num_spills)

        # Map dimensions
        self.grid_size = self.grid_map.shape
        #self.max_rectangle_area = self.calculate_rectangle_area()

        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
        self.action_space = gym.spaces.MultiDiscrete([4] * self.num_robots)
        #ele=[[0,1],[0,0]]
        # Observation space
        self.observation_space = gym.spaces.Dict({
            "known_map": gym.spaces.Box(0, 1, shape=self.grid_map.shape, dtype=np.uint8),
            "spill_map": gym.spaces.Box(0, 1, shape=self.grid_map.shape, dtype=np.uint8),
            "robot_positions": gym.spaces.Box(0, max(self.grid_size), shape=(self.num_robots, 2), dtype=np.int32),
        })

        self.known_map = np.zeros_like(self.grid_map)
        self.spill_map = np.zeros_like(self.grid_map)  # Initialize spill map to track detected spills
        self.cumulative_reward = 0

    # def generate_spills(self, grid_map, num_spills):
    #     """Randomly generate chemical spills in the grid map"""
    #     for _ in range(num_spills):
    #         x, y = random.choice(np.argwhere(grid_map == 0))  # Randomly select free space
    #         grid_map[x, y] = 1  # Mark as chemical spill
    #     return grid_map

    def reset(self):
        """Reset the environment"""
        self.step_count = 0
        self.cleaned_cell_count=0
        self.non_spill_exp_count=0
        cell_size = 20
        width = self.grid_size[1] * cell_size
        height = self.grid_size[0] * cell_size

        self.grid_map=grid_map
        screen = pygame.display.set_mode((width, height))
        screen.fill(bg_color)
        self.tot_chemical=0
        
        #self.robot_positions = np.array([[1, 1]])  # Starting position for the robot(s)
        #random_indices = random.sample(range(len(free_cells)), self.num_robots)
        self.robot_positions = np.array([(9,9)])
        self.prev_positions = self.robot_positions.copy()

        # map reset
        self.known_map = np.zeros_like(self.grid_map)
        self.spill_map = np.zeros_like(self.grid_map)  

        # Reset spill map
        self.cumulative_reward = 0
        self.distances_traveled = np.zeros(self.num_robots)
        self.steps_taken = np.zeros(self.num_robots, dtype=int)
        self.paths=[[]]

        return self._get_observation()

    def step(self, actions):
        
        self.step_count += 1
        reward = 0
        new_positions = []

        robot_colors = [(0, 0, 0)]

        cell_size = 20
        width = self.grid_size[1] * cell_size
        height = self.grid_size[0] * cell_size
        screen = pygame.display.set_mode((width, height))
        screen.fill(bg_color)

        valid_move=True

        for y in range(height // cell_size):
            for x in range(width // cell_size):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, grid_color, rect, 1)


        for i, action in enumerate(actions):

            #print(f"robot positions: {self.robot_positions}")

            new_position, pre_pos = self._move_robot(self.robot_positions[i], action)
            self.paths[i].append((pre_pos.copy(),new_position.copy()))
            abs_x=new_position[0]-pre_pos[0]
            abs_y=new_position[1]-pre_pos[1]

            self.robot_positions[i] = new_position
            self._update_known_map(self.robot_positions[i])

            #Two step counts for the turning
            if (((abs_x!=0) and (abs_y!=0))):
                self.step_count+=1

            # Reward shaping for cleaning chemical spills
            if self.grid_map[new_position[0], new_position[1]] == 1:  # If it's a spill
                #print([new_position[0], new_position[1]])
                #print("cleaning start")

                if self.spill_map[new_position[0], new_position[1]] == 0:  
                    reward = 50  # Reward for cleaning
                    self.cleaned_cell_count+=1
                    #print("Cleaning")
                    self.spill_map[new_position[0], new_position[1]] = 1  # If not cleaned yet# Mark as cleaned
                else:  #if its already cleaned cell, give a negative reward
                    reward=-1
                    self.non_spill_exp_count+=1

            else:
                reward = -1  # Penalty for moving to non-spill cells
                self.non_spill_exp_count+=1

            # Step constraint check
            total_allowed_steps = int(1.2 * (self.grid_size[0] * self.grid_size[1]))
            if self.step_count >= total_allowed_steps:
                print(f"Cleaned cell count: {self.cleaned_cell_count}")
                print(f"total contaminated cells: {self.tot_chemical}")
                #print(f"Exploring cells : {self.non_spill_exp_count}")
                time.sleep(1)
                done = True
                #print(self.step_count)
                print("Cleaning complete")
                time.sleep(1)
                #reward -= 50  # Penalty for exceeding steps
            else:
                done = False

            rect = pygame.Rect(self.robot_positions[i][1] * cell_size, self.robot_positions[i][0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, robot_color, rect)

            new_positions.append(tuple(new_position))
            pre_action=action

        #Visualize explored map and spills
        screen.fill(bg_color)
        #print(" working...........111111111111111111111.........................")
        #Draw grid           
        #Draw spills and cleaned spills
        self.tot_chemical=0
        for y in range(self.grid_size[0]):
                for x in range(self.grid_size[1]):
                    rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                    if self.grid_map[y, x] == 1:
                        if self.spill_map[y, x] == 1:
                            #pygame.draw.rect(screen, (0,0,0), rect)  # green
                            pass
                            #print("cover")
                        else:
                            self.tot_chemical+=1
                            #print(f"total chemical cells: {self.tot_chemical}")
                            pygame.draw.rect(screen, (165, 42, 42), rect)  # brown

        if self.step_count<3:
            for y in range(self.grid_size[0]):
                for x in range(self.grid_size[1]):
                    rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                    if self.grid_map[y, x] == 1:
                        if self.spill_map[y, x] == 1:
                            #pygame.draw.rect(screen, (0,0,0), rect)  # green
                            pass
                            #print("cover")
                        else:
                            self.tot_chemical+=1
                            #print(f"total chemical cells: {self.tot_chemical}")
                            #pygame.draw.rect(screen, (165, 42, 42), rect)  # brown
                            pass
            print(f"chemical cells iteration : {self.tot_chemical}")           
      
        obs = np.argwhere(self.known_map > 0)
        for data in obs:
            rect = pygame.Rect(data[1] * cell_size, data[0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, obs_color, rect)
            
        self.paths[i].append((pre_pos.copy(),new_position.copy()))
        for ind,paths in enumerate(self.paths):
            #print(paths)
            #print("<<<<<<<<<<<<loooppppp>>>>>>>>>>>>>>>>>")
            # Check code does not in paths because path array is emty;

            for subpaths in paths:                
                pygame.draw.line(screen, robot_colors[ind],  # Assign unique color for each robot
                (subpaths[0][1] * cell_size + cell_size // 2, subpaths[0][0] * cell_size + cell_size // 2),
                (subpaths[1][1] * cell_size + cell_size // 2, subpaths[1][0] * cell_size + cell_size // 2),
                3 )
                #print(subpaths)

        pygame.display.update()
        #print(self.spill_map)
        time.sleep(0.01)
        return self._get_observation(), reward, done, {}
        
    def _move_robot(self, position, action):
        """Move the robot according to the action"""
        #print(f"Position: {position}")
        #position=np.array(position)
        x, y = position.copy()
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.grid_size[0] - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.grid_size[1] - 1:
            y += 1

        else:
            x=x
            y=y
        return np.array([x, y]), position
    
    def _update_known_map(self, position):
        """Update known map based on robot's field of view"""
        
        x, y = position
        """for dx in range(-self.field_of_view, self.field_of_view + 1):
            for dy in range(-self.field_of_view, self.field_of_view + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    self.known_map[nx, ny] = self.grid_map[nx, ny]"""

        # Update spill map using the smaller chemical sensor field of view
        for dx in range(-self.chem_fov, self.chem_fov + 1):
            for dy in range(-self.chem_fov, self.chem_fov + 1):
                #nx, ny = x + dx, y + dy
                nx, ny=x, y
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    if self.grid_map[nx, ny] == 1:
                        self.known_map[nx, ny] = 1  # Mark detected spills
        

    def _get_observation(self):
        """Return the current observation"""
        return {
            "known_map": self.known_map.copy(),
            "spill_map": self.spill_map.copy(),
            "robot_positions": self.robot_positions.copy(),
        }

    def calculate_rectangle_area(self):
        """Calculate the area of the rectangle based on the robot's positions"""
        x_coords = self.robot_positions[:, 0]
        y_coords = self.robot_positions[:, 1]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        side_length = max(max_x - min_x, max_y - min_y) + 1
        return side_length ** 2
#test case 1 map 2
grid_map = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
grid_map=np.array(grid_map)

'''

# Usage example:
rows, cols = 20, 20
main_cluster_size_range = (30, 60)
subcluster_config = {
        "num_range": (4, 8),
        "size_range": (5, 15),
        "distance_from_main": (9, 10),
        "distance_between_subclusters": 8
    }
min_distance=5
grid_maps = MapGen()
grid_map_call=grid_maps.generate_main_with_subclusters_map(
        rows, cols, main_cluster_size_range, subcluster_config
    )#num_spills = 5  # Number of chemical spills to generate
grid_map=grid_map_call

grid_map=np.array(grid_map)
print(grid_map)
# Generate random positions for robots
free_cells = np.argwhere(grid_map == 0)
print(len(free_cells))
random_indices = random.sample(range(len(free_cells)), 1)
robot_positions = np.array([tuple(free_cells[i]) for i in random_indices])

env = ChemicalClean(grid_map, num_robots=1, robot_positions=robot_positions, field_of_view=0, chem_fov=1)
models_dir = f"models/Chemical_spils_02_06_2025/{int(time.time())}/"
logdir = f"logs/Chemical_spils_02_06_2025/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

print("Training started...")
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir,ent_coef=0.05, device="cuda", learning_rate=3e-4)
TIMESTEPS = 130000
iters=0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, log_interval=1)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
    #log_data = pd.read_csv(env.get_monitor_file())
    #model.save("loading/rectangle_reduction_model_PPO_2025_01_15_v5.0")
    print("Model saved successfully!")
'''