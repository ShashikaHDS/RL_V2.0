import gym
import numpy as np
from stable_baselines3 import PPO
import random
import pygame
import os
import time 
from map_gen_v4 import MapGen

pygame.init()

# Colors
bg_color = (255, 255, 255)
grid_color = pygame.Color("grey")
obs_color = (0, 0, 0)
robot_color = (255, 0, 0)

class RectangleReductionEnv(gym.Env):
    def __init__(self, grid_map, num_robots, robot_positions, field_of_view, chem_fov):
        super(RectangleReductionEnv, self).__init__()
        
        # Initialize environment variables
        self.grid_map = np.array(grid_map)
        self.num_robots = 1
        self.robot_positions = np.array(robot_positions)
        self.prev_positions = self.robot_positions.copy()
        self.field_of_view = field_of_view
        self.step_count = 0
        self.map_use_count = 0
        self.max_map_uses = 3                               #no of maps per epi
        self.reward_contribution = []
        self.robot_distance=[0]
        #self.paths = [[],[],[],[]] #for 4 robots
        self.paths = [[]]
        # Distance tracking
        self.distances_traveled = np.zeros(num_robots)  # Initialize distances for all robots

        self.steps_taken = np.zeros(num_robots, dtype=int)  # Tracks valid steps for each robot

        # Map dimensions
        self.grid_size = self.grid_map.shape

        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
        self.action_space = gym.spaces.MultiDiscrete([5] * self.num_robots)

        # Visited cells
        self.visit_map = np.zeros_like(self.grid_map)

        # Observation space
        self.observation_space = gym.spaces.Dict({
            "known_map": gym.spaces.Box(0, 1, shape=self.grid_map.shape, dtype=np.uint8),
            "spill_map": gym.spaces.Box(0, 1, shape=self.grid_map.shape, dtype=np.uint8),
            #"elevation_map": gym.spaces.Box(0, 2, shape=self.grid_map.shape, dtype=np.uint8),
            "robot_positions": gym.spaces.Box(0, max(self.grid_size), shape=(self.num_robots, 2), dtype=np.int32),
        })

        # Maps
        self.known_map = np.zeros_like(self.grid_map)
        self.spill_map = np.zeros_like(self.grid_map)
        #self.elevation_map = np.zeros_like(self.grid_map)
        
        self.cumulative_reward = 0

    def reset(self):
        self.step_count = 0

        #self.grid_map=grid_map
        self.grid_map = grid_maps.generate_connected_clusters_map(
                rows=20, cols=20, num_clusters=5, cluster_size_range=(2, 10), min_distance=3)

        free_cells = np.argwhere(self.grid_map == 0)
        random_indices = random.sample(range(len(free_cells)), self.num_robots)
        self.robot_positions = np.array([tuple(free_cells[i]) for i in random_indices])
        #self.robot_positions =np.array( [[1,1]])
        #self.robot_1=[1,1]


        #rest each map
        self.prev_positions = self.robot_positions.copy()
        self.known_map = np.zeros_like(self.grid_map)
        self.spill_map = np.zeros_like(self.grid_map)
        #self.elevation_map = np.zeros_like(self.grid_map)

        #self.max_rectangle_area,free_status,bounds = self.calculate_rectangle_area()
        self.cumulative_reward = 0
        self.step_count_=1


        # Reset distances traveled
        self.distances_traveled = np.zeros(self.num_robots)
        self.steps_taken = np.zeros(self.num_robots, dtype=int)  # Reset steps counter
        self.paths = [[]]     #change with no of robots


        return self._get_observation()

    def step(self, actions):
        
        if self.step_count==0:
            pass        
        
        self.step_count += 1

        #print(f"step count : {self.step_count}")
        reward = 0
        new_positions = []
        cell_size = 20
        width = self.grid_size[1] * cell_size
        height = self.grid_size[0] * cell_size
        screen = pygame.display.set_mode((width, height))
        screen.fill(bg_color)
        # Define different colors for each robot path
        robot_colors = [
        (255, 0, 0)
        ]

        previous_positions = self.robot_positions.copy()

        valid_move=True
        self_collide = False
        obs_collide = False
        smt_wrong = False
  
        for y in range(height // cell_size):
            for x in range(width // cell_size):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, grid_color, rect, 1)
        
        #print(self.robot)
        for i, action in enumerate(actions):  
                                                                          
            new_position, pre_pos = self._move_robot(self.robot_positions[i], action)
            
            if not isinstance(pre_pos, np.ndarray):
                pre_pos = np.array(pre_pos)            

            #print(f"Robot positions: {self._move_robot(self.robot_positions[i], action)}")
            #print(self.step_count_)
            #distance measure
            if pre_pos.shape != (2,):
                print(f"Error: prev_position[{i}] has shape {pre_pos.shape}, expected (2,)")
                print(" what happend ? ")
                continue  # Skip this iteration if prev_position is invalid



            #new reward function starts
            for i, action in enumerate(actions):
                new_position, pre_pos = self._move_robot(self.robot_positions[i], action)
                self.robot_positions[i] = new_position
                self._update_known_map(self.robot_positions[i])

                # Reward shaping for cleaning chemical spills
                if self.grid_map[new_position[0], new_position[1]] == 1:  # If it's a spill
                    if self.spill_map[new_position[0], new_position[1]] == 0:  # If not cleaned yet
                        reward += 50  # Reward for cleaning
                        self.spill_map[new_position[0], new_position[1]] = 1  # Mark as cleaned
                else:
                    reward -= 1  # Penalty for moving to non-spill areas

                # Step constraint check
                total_allowed_steps = int(0.6 * (self.grid_size[0] * self.grid_size[1]))
                if self.step_count >= total_allowed_steps:
                    done = True
                    reward -= 50  # Penalty for exceeding steps
                else:
                    done = False

                new_positions.append(tuple(new_position))

            
            if valid_move:                                         #Increment only if the move is valid
                new_positions.append(tuple(new_position))

            if not np.array_equal(new_position, pre_pos):     # Ensure movement happened
                self.distances_traveled[i] += 1               # Count only actual steps
                self.paths[i].append((pre_pos.copy(),new_position.copy()))
            #return self._get_observation(), reward, done, {}           
     
            
            self.robot_positions[i] = new_position
            self._update_known_map(self.robot_positions[i])
            #print(self.paths)
            rect = pygame.Rect(self.robot_positions[i][1] * cell_size, self.robot_positions[i][0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, robot_color, rect)

        
        #pygame display
        obs = np.argwhere(self.known_map > 0)

        
        for data in obs:
            rect = pygame.Rect(data[1] * cell_size, data[0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, obs_color, rect)


        for ind,paths in enumerate(self.paths):
            
            for subpaths in paths:
                
                pygame.draw.line(screen, robot_colors[ind],  # Assign unique color for each robot
                (subpaths[0][1] * cell_size + cell_size // 2, subpaths[0][0] * cell_size + cell_size // 2),
                (subpaths[1][1] * cell_size + cell_size // 2, subpaths[1][0] * cell_size + cell_size // 2),
                3 )

        #new_area,free_status,square_bounds = self.calculate_rectangle_area()

        #self.draw_square(screen, cell_size)
        pygame.display.update()

        new_min_x = min(self.robot_positions[:, 0])
        new_max_x = max(self.robot_positions[:, 0])
        new_min_y = min(self.robot_positions[:, 1])
        new_max_y = max(self.robot_positions[:, 1])

        prev_min_x = min(self.prev_positions[:, 0])
        prev_max_x = max(self.prev_positions[:, 0])
        prev_min_y = min(self.prev_positions[:, 1])
        prev_max_y = max(self.prev_positions[:, 1])

        center_previous=self.calculate_rectangle_centre_point(prev_max_x,prev_min_x,prev_max_y,prev_min_y)
        center_new=self.calculate_rectangle_centre_point(prev_max_x,prev_min_x,prev_max_y,prev_min_y)
        if self.step_count==1:
            pass

        
        self.prev_positions = self.robot_positions.copy()

        #time.sleep(0.1)
        
        new_positions.append(tuple(new_position))
        self.robot_positions[i] = new_position
        self._update_known_map(new_position)
        self.step_count_+=1

        return self._get_observation(), reward, done, {}
    
    def is_square_free(self, min_x, max_x, min_y, max_y):
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if self.grid_map[x, y] != 0:  # 0 represents free cells
                    return False
        return True

    def _move_robot(self, position, action):
        x, y = position.copy()
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.grid_size[0] - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.grid_size[1] - 1:
            y += 1
        return np.array([x, y]), position
    
    def draw_square(self, screen, bounds, cell_size):
        min_x, max_x, min_y, max_y = bounds
        rect_x = min_y * cell_size  # Convert grid coordinates to pixel coordinates
        rect_y = min_x * cell_size
        rect_width = (max_y - min_y + 1) * cell_size
        rect_height = (max_x - min_x + 1) * cell_size
        square_color = (0, 255, 0)  # Green for valid square

        pygame.draw.rect(screen, square_color, (rect_x, rect_y, rect_width, rect_height), 2)  # Thickness = 2


    def calculate_rectangle_area(self):
        x_coords = self.robot_positions[:, 0]
        y_coords = self.robot_positions[:, 1]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Ensure it's a square by enforcing equal side lengths
        side_length = max(max_x - min_x, max_y - min_y) + 1
        square_min_x = min_x
        square_max_x = square_min_x + side_length - 1
        square_min_y = min_y
        square_max_y = square_min_y + side_length - 1

        # Check if all cells inside the square are free
        free = True
        if square_max_x >= self.grid_size[0] or square_max_y >= self.grid_size[1]:
            free = False
        else:
            for x in range(square_min_x, square_max_x + 1):
                for y in range(square_min_y, square_max_y + 1):
    
                    if self.grid_map[x, y] != 0:  # Non-free cell
                        free  = False
                                                                                                                                        
        # Valid square area
        return side_length ** 2, free , (square_min_x, square_max_x, square_min_y, square_max_y)

    def calculate_rectangle_centre_point(self,prev_max_x,prev_min_x,prev_max_y,prev_min_y):
        x_center=(prev_max_x-prev_min_x)/2
        y_center=(prev_max_y-prev_min_y)/2
        return [x_center,y_center]


    def _get_observation(self):
        return {
            "known_map": self.known_map.copy(),
            "robot_positions": self.robot_positions.copy(),
            "spill_map": self.spill_map.copy(),
        }

    def _update_known_map(self, position):
        x, y = position
        for dx in range(-self.field_of_view, self.field_of_view + 1):
            for dy in range(-self.field_of_view, self.field_of_view + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    self.known_map[nx, ny] = self.grid_map[nx, ny]
        
grid_maps=MapGen()

num_robots = 1
field_of_view = 1
chem_fov=1
grid_map_call = grid_maps.generate_connected_clusters_map(rows=20, cols=20, num_clusters=5, cluster_size_range=(2, 10), min_distance=3)


#elevation map call


grid_map = grid_map_call.copy()
free_cells = np.argwhere(grid_map == 0)
random_indices = random.sample(range(len(free_cells)), num_robots)
robot_positions = np.array([tuple(free_cells[i]) for i in random_indices])

env = RectangleReductionEnv(grid_map, num_robots, robot_positions=robot_positions, field_of_view=2,chem_fov=chem_fov)
#env = DummyVecEnv([lambda: env])  # Convert to vectorized env
#env = VecMonitor(env, filename="./logs/monitor_log")  # Add VecMonitor for tracking rewards
models_dir = f"models/graph_models/5_graphrobots/{int(time.time())}/"
logdir = f"logs/new_logs/graph/{int(time.time())}/"

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
