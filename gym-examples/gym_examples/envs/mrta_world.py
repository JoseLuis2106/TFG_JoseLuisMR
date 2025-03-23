import gym
from gym import spaces
import pygame
import numpy as np


class MRTAWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, rows=8, cols=8, num_robots=2, num_tasks=3):
        self.rows = rows                # Numero de filas del grid
        self.cols = cols                # Numero de columnas del grid
        self.size = rows*cols           # Tamaño del grid
        self.window_size = 512          # Tamaño de la ventana de PyGame
        self.max_steps = 300            # Maximo numero de steps
        self.num_robots = num_robots    # Numero de robots
        self.num_tasks = num_tasks      # Numero de tareas

        # Las observaciones son diccionarios con: 
        # Posicion de los robots y de las tareas
        # Estado de cada tarea
        # Asignacion de cada tarea
        # Tiempo transcurrido desde la ultima accion

        self.observation_space = spaces.Dict(
            {
                "robots_positions": spaces.Box(0, self.size - 1, shape=(self.num_robots,), dtype=int),      #Pos in interval (0,size-1)
                "tasks_positions": spaces.Box(0, self.size - 1, shape=(self.num_tasks,), dtype=int),        #Pos in interval (0,size-1)
                "tasks_states": spaces.MultiDiscrete([3] * self.num_tasks),                                 #0: Pending, 1: Completed, 2: Failed
                "tasks_allocations": spaces.Box(0, self.num_robots, shape=(self.num_tasks,), dtype=int),    #0: Not allocated, else (n): Allocated to robot n (Probar a cambiar por multidiscrete)
                "time_lapse": spaces.Box(0, np.inf, dtype=float),                                           #Time lapsed since last step
            }
        )

        # Cada accion es un conjunto de asignaciones de tareas
        self.action_space = spaces.MultiDiscrete([self.num_robots + 1] * self.num_tasks)

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
        return {
                "robots_positions": self._robots_positions,
                "tasks_positions": self._tasks_positions,
                "tasks_states": self._tasks_states,
                "tasks_allocations": self._tasks_allocations,
                "time_lapse": self._time_lapse,
                }
    
    # def _get_info(self):
    #     return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def reset(self, seed=None, options=None):
        # Reinicie el numero de steps
        self.steps=0

        # Semilla de self.np_random
        super().reset(seed=seed)

        # Posicion aleatoria inicial de robots y tareas
        self._robots_positions = self.np_random.choice(self.size, size=self.num_robots, replace=False)
        self._tasks_positions = self.np_random.choice(
                                np.setdiff1d(np.arange(self.size), self._robots_positions),  # Sin superposición entre tareas y robots
                                size=self.num_tasks,
                                replace=False
                                )
        self._tasks_states=np.zeros(self.num_tasks, dtype=int)
        self._tasks_allocations=np.zeros(self.num_tasks, dtype=int)
        self._time_lapse = 0.0

        observation = self._get_obs()
        # info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, {}#info

    def _pos2rowcol(self,pos):
        """
        Transforma una posición como valor entero a posición como (fila,columna).
        """
        row, col = pos//self.cols, pos%self.cols
        return row, col

    def _rowcol2pos(self,row,col):
        """
        Transforma una posición como (fila,columna) a posición como valor entero.
        """
        return row*self.cols+col  
    
    def step(self, action):
        # Actualiza numero de steps y reinicia time lapse y variable end_step
        self.steps += 1
        self._time_lapse = 0
        end_step = 0
        action = np.asarray(action)

        #Ejecucion del step
        terminated = 0
        truncated = 0
        observation = self._get_obs()

        num_robots_assigned = np.count_nonzero(action)
        num_tasks_pending = np.count_nonzero(self._tasks_states == 0)

        if num_robots_assigned == 0:
            reward = -100
            end_step = 1
        elif num_robots_assigned < min(self.num_robots,num_tasks_pending):
            reward = -10*min(self.num_robots-num_robots_assigned,num_tasks_pending)
        else:
            reward = 0


        while not end_step:
            # Actualizacion del time lapse
            self._time_lapse += 1

            # Movimiento de los robots a sus tareas
            for i, robot_id in enumerate(np.asarray(action)):  # Para cada asignacion
                if robot_id > 0:  # Si la tarea está asignada
                    task_idx = i  # Indice de la tarea
                    task_pos = self._tasks_positions[task_idx]          # Posicion de la tarea
                    robot_pos = self._robots_positions[robot_id-1]      # Posicion del robot

                    # Movimiento de cada robot
                    if robot_pos != task_pos:
                        task_row, task_col = self._pos2rowcol(task_pos)
                        robot_row, robot_col = self._pos2rowcol(robot_pos)

                        dir_row = np.sign(task_row - robot_row)
                        dir_col = np.sign(task_col - robot_col)

                        robot_row += dir_row
                        robot_col += dir_col
                        robot_pos = self._rowcol2pos(robot_row,robot_col)

                        self._robots_positions[robot_id-1] = robot_pos

                    # Comprueba si el robot ha llegado a su tarea (por modificar: duracion variable y probabilidad fallo)
                    if self._robots_positions[robot_id-1] == task_pos and self._tasks_states[task_idx] != 1:
                        reward += 20
                        self._tasks_states[task_idx] = 1  # Task completed
                        end_step = 1
                        break  # Final del step

            # Un episodio acaba si todas las tareas se completan o si el numero de steps alcanza el limite (300)
            terminated = np.all(self._tasks_states == 1)
            truncated = self.steps>self.max_steps
            reward += 100 if terminated else -1
            observation = self._get_obs()
            # info = self._get_info()
            
            end_step=end_step or terminated or truncated


            if self.render_mode == "human":
                self._render_frame()

        return observation, reward, terminated, truncated, {}#info
    
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
            self.window_size / self.rows
        )  # Tamaño en pixeles de una celda

        # Se dibujan primero las tareas
        for task, task_pos in enumerate(self._tasks_positions):
            task_row, task_col = self._pos2rowcol(task_pos)
            if self._tasks_states[task] == 0:
                color = (0, 0, 0)
            elif self._tasks_states[task] == 1:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            pygame.draw.rect(
                canvas, color,
                pygame.Rect(pix_square_size * np.array([task_col, task_row]), (pix_square_size, pix_square_size)),
            )

        # Luego, los robots
        for rob_pos in self._robots_positions:
            rob_row, rob_col = self._pos2rowcol(rob_pos)
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (np.array([rob_col, rob_row]) + 0.5) * pix_square_size,
                pix_square_size / 4,
            )

        # Finalmente, las lineas del grid
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
            # Se copia el dibujo de canvas a la ventana
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # El renderizado "human" debe ocurrir con el framrate predefinido.
            # Se añade un delay para mantener el framerate estable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes = (1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()