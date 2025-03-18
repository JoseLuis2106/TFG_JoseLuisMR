import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.max_steps = 300  # Maximun number of steps
        self.num_obstacles = max(1, int(0.01*self.size**2)) # Number of obstacles

        # Observations are dictionaries with the agent's and the target's relative location.
        # This relative location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "relative_target": spaces.Box(-size + 1, size - 1, shape=(2,), dtype=int),
                "relative_obstacles": spaces.Box(-size + 1, size - 1, shape=(self.num_obstacles, 2), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        return {"relative_target": self._target_location - self._agent_location,
                "relative_obstacles": self._obstacles - self._agent_location}
    
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def reset(self, seed=None, options=None):
        # Reset the number of steps and best distance
        self.steps=0
        self.best_dist=self.size*np.sqrt(2)

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        distance=0.0
        while np.array_equal(self._target_location, self._agent_location) or distance<self.size*0.8:
            if self._agent_location[0]<self.size/2:
                x=self.np_random.integers(self.size//2, self.size, dtype=int)
            else:
                x=self.np_random.integers(0, self.size//2, dtype=int)
            if self._agent_location[1]<self.size/2:
                y=self.np_random.integers(self.size//2, self.size, dtype=int)
            else:
                y=self.np_random.integers(0, self.size//2, dtype=int)

            self._target_location = np.array([x, y])
            distance=self._get_info()["distance"]

        # Generation of obstacles
        self._obstacles = []
        while len(self._obstacles) < self.num_obstacles:
            obstacle = self.np_random.integers(0, self.size, size=2, dtype=int)
            if (
                not np.array_equal(obstacle, self._agent_location)
                and not np.array_equal(obstacle, self._target_location)
                and not any(np.array_equal(obstacle, obs) for obs in self._obstacles)
            ):
                self._obstacles.append(obstacle)
        self._obstacles = np.array(self._obstacles)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # Update number of steps
        self.steps+=1

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # An episode is done if the agent has reached the target or the number of steps gets to the limit (300)
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = self.steps>self.max_steps
        reward=200 if terminated else -1
        observation = self._get_obs()
        info = self._get_info()

        distance=info["distance"]
        if distance<self.best_dist:
            self.best_dist=distance
            reward+=(self.size*np.sqrt(2)-distance)/5

        if truncated or any(np.array_equal(self._agent_location, obs) for obs in self._obstacles):
            reward = -200
            terminated=True

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Now, the obstacles
        for obstacle in self._obstacles:
            pygame.draw.rect(
                canvas, (0, 0, 0),
                pygame.Rect(pix_square_size * obstacle, (pix_square_size, pix_square_size)),
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()