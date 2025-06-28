import gym
from gym import spaces
import pygame
import numpy as np
import time


class MRTAWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode = None, rows = 8, cols = 8, num_robots = 2, num_tasks = 3):
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
        # Estado de cada robot (ocupado o no)
        # Tipo de robot (baja o alta capacidad)
        # Tiempo transcurrido desde la ultima accion

        self.observation_space = spaces.Dict( #Añadir estados robots
            {
                "robots_positions": spaces.Box(0, self.size - 1, shape=(self.num_robots,), dtype=int),      #Pos in interval (0,size-1)
                "tasks_positions": spaces.Box(0, self.size - 1, shape=(self.num_tasks,), dtype=int),        #Pos in interval (0,size-1)
                # "tasks_states": spaces.MultiDiscrete([3] * self.num_tasks),                                 #0: Pending, 1: Completed, 2: Failed
                "tasks_states": spaces.MultiDiscrete([4] * self.num_tasks),                                 #0: Pending, 1: Completed, 2: Failed, 3: Allocated
                "tasks_allocations": spaces.Box(0, self.num_robots, shape=(self.num_tasks,), dtype=int),    #0: Not allocated, else (n): Allocated to robot n (Probar a cambiar por multidiscrete)
                "busy_robots": spaces.MultiDiscrete([2] * self.num_robots),                                 #0: Free, 1: Busy
                "tasks_types": spaces.MultiDiscrete([3] * self.num_tasks),                                  #0: Easy, 1: Medium, 2: Hard
                "robots_types": spaces.MultiDiscrete([2] * self.num_robots),                                #0: Low capacity, 1: High capacity
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
                "busy_robots": self._busy_robots,
                "tasks_types": self._task_types,
                "robots_types": self._robots_types,
                "time_lapse": self._time_lapse,
                }
    
    def _get_info(self):
        dist_travelled = []
        for i in range(self.num_robots):
            pos0 = self._initial_positions[i]
            pos1 = self._robots_positions[i]
            row0, col0 = self._pos2rowcol(pos0)
            row1, col1 = self._pos2rowcol(pos1)
            dist = abs(row0 - row1) + abs(col0 - col1)
            dist_travelled.append(dist)

        perc_success = np.count_nonzero(self._tasks_states == 1) / self.num_tasks

        return {"dist_travelled": dist_travelled, "time_steps": self._time_lapse, "perc_success": perc_success, "tasks_distribution": self._tasks_distribution}

    def reset(self, seed = None, options = None):
        # Reinicie el numero de steps
        self.steps = 0

        # Semilla de self.np_random
        super().reset(seed = seed)

        # Tipos de robots (0: baja capacidad, 1: alta capacidad)
        self._robots_types = np.random.randint(0, 2, self.num_robots)

        # Al menos, uno de cada tipo
        if np.count_nonzero(self._robots_types) == 0:
            self._robots_types[np.random.randint(0, self.num_robots)] = 1
        elif np.count_nonzero(self._robots_types) == self.num_robots:
            self._robots_types[np.random.randint(0, self.num_robots)] = 0

        # self._robots_types = np.ones(self.num_robots, dtype=int)

        self._fail_prob = np.zeros(self.num_robots, dtype=float)        # Probabilidad de fallo de cada robot
        for rob, robot_type in enumerate(self._robots_types):
            if robot_type == 0:
                self._fail_prob[rob] = 0.3
            else:
                self._fail_prob[rob] = 0.1

        self._tasks_distribution = np.zeros(2, dtype=float)             # Conteo de tareas asignadas a robots de baja y alta capacidad


        # Tiempo aleatorio en completar cada tarea
        self._task_types = np.random.randint(0, 3, self.num_tasks)
        self._T2end = np.zeros(self.num_tasks, dtype=int)

        for task, task_type in enumerate(self._task_types):
            if task_type == 0:
                self._T2end[task] = 3 + np.random.randint(-1, 2)    # Provisional
            elif task_type == 1:
                self._T2end[task] = 5 + np.random.randint(-1, 2)    # Provisional
            else:
                self._T2end[task] = 7 + np.random.randint(-1, 2)    # Provisional

        # self._task_types = np.zeros(self.num_tasks, dtype=float)
        # self._T2end = 3*np.ones(self.num_tasks, dtype=int)


        # Posicion aleatoria inicial de robots y tareas
        self._robots_positions = self.np_random.choice(self.size, size=self.num_robots, replace=False)
        self._tasks_positions = self.np_random.choice(
                                np.setdiff1d(np.arange(self.size), self._robots_positions),  # Sin superposición entre tareas y robots
                                size=self.num_tasks,
                                replace=False
                                )

        # Inicializacion de estados de tareas y robots y de las asignaciones
        self._tasks_states = np.zeros(self.num_tasks, dtype=int)
        self._busy_robots = np.zeros(self.num_robots, dtype=int) 
        self._tasks_allocations = np.zeros(self.num_tasks, dtype=int)
        self._time_lapse = 0.0

        self._initial_positions = self._robots_positions.copy()     # Posiciones iniciales robots

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _pos2rowcol(self, pos):
        """
        Transforma una posición como valor entero a posición como (fila,columna).
        """
        row, col = pos // self.cols, pos % self.cols
        return row, col

    def _rowcol2pos(self, row, col):
        """
        Transforma una posición como (fila, columna) a posición como valor entero.
        """
        return row * self.cols + col  
    
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
        num_tasks_allocated = np.count_nonzero(self._tasks_states == 3)
        num_tasks_to_finish = np.count_nonzero(self._tasks_states == 0) + num_tasks_allocated

        # print("num_robots_assigned",num_robots_assigned)
        # print("num_tasks_to_finish",num_tasks_to_finish)
        # print("num_tasks_allocated",num_tasks_allocated)

        if num_robots_assigned == 0 and num_tasks_allocated == 0:
            reward = -100
            end_step = 1
        elif num_robots_assigned < min(self.num_robots, num_tasks_to_finish):
            # print(f"Asignaciones insuficientes.\nRequeridos: {min(self.num_robots, num_tasks_to_finish)}, Asignados: {num_robots_assigned}")
            reward = -20 * min(self.num_robots - num_robots_assigned, num_tasks_to_finish)
        else:
            reward = 0


        # print(f"Asignaciones anteriores: {self._tasks_allocations} (mrta_world)")
        self._tasks_allocations = action.copy()         # Asignaciones de tareas a robots
        # self._tasks_allocations = np.maximum(action, self._tasks_allocations)
        # print(f"Asignaciones finales: {self._tasks_allocations} (mrta_world)")


        for task, rob in enumerate(self._tasks_allocations):
            if 1 <= self._tasks_states[task] <= 2:      # Tarea completada o fallada
                self._tasks_allocations[task] = 0

            if rob > 0:
                if self._robots_types[rob - 1] == 0 and self._tasks_states[task] != 3:      # Si el robot es de baja capacidad
                    self._tasks_distribution[0] += 1
                elif self._robots_types[rob - 1] == 1 and self._tasks_states[task] != 3:    # Si el robot es de alta capacidad
                    self._tasks_distribution[1] += 1
                
                self._tasks_states[task] = 3            # Tarea asignada a algun robot


        self._initial_positions = self._robots_positions.copy()     # Posiciones iniciales robots
        

        if self.render_mode != "human":
            try:
                min_dist = min(         # Menor de las distancias entre cada robot y su tarea
                    abs(self._pos2rowcol(self._tasks_positions[task])[0] - self._pos2rowcol(self._robots_positions[robot-1])[0]) +
                    abs(self._pos2rowcol(self._tasks_positions[task])[1] - self._pos2rowcol(self._robots_positions[robot-1])[1]) +
                    self._T2end[task]
                    for task, robot in enumerate(self._tasks_allocations) if robot > 0)
            except ValueError:
                min_dist = 0

            self._time_lapse = min_dist

            for i, robot_id in enumerate(self._tasks_allocations):      # Para cada asignacion
                if robot_id > 0:                                        # Si la tarea está asignada
                    task_idx = i                                        # Indice de la tarea
                    task_pos = self._tasks_positions[task_idx]          # Posicion de la tarea
                    robot_pos = self._robots_positions[robot_id - 1]    # Posicion del robot
                    self._busy_robots[robot_id - 1] = 1                 # Si la tarea está asignada, el robot está ocupado

                    task_row, task_col = self._pos2rowcol(task_pos)
                    robot_row, robot_col = self._pos2rowcol(robot_pos)

                    dir_row = np.sign(task_row - robot_row)
                    dir_col = np.sign(task_col - robot_col)

                    robot_row += dir_row * min(min_dist, abs(task_row - robot_row))
                    min_dist_aux = min_dist - min(min_dist, abs(task_row - robot_row))
                    robot_col += dir_col * min(min_dist_aux, abs(task_col - robot_col))
                    min_dist_aux = min_dist_aux - min(min_dist_aux, abs(task_col - robot_col))
                    self._T2end[task_idx] = max(0, self._T2end[task_idx] - min_dist_aux)

                    robot_pos = self._rowcol2pos(robot_row, robot_col)
                    self._robots_positions[robot_id - 1] = robot_pos

            # Comprueba si el robot ha llegado a su tarea (por modificar: probabilidad fallo)
            for task_idx, robot_id in enumerate(self._tasks_allocations):   # Para cada asignacion
                if robot_id > 0:                                            # Si la tarea está asignada
                    task_pos = self._tasks_positions[task_idx]              # Posicion de la tarea
                    robot_pos = self._robots_positions[robot_id - 1]        # Posicion del robot
                    if self._robots_positions[robot_id - 1] == task_pos and self._tasks_states[task_idx] != 1 and self._tasks_states[task_idx] != 2 and self._T2end[task_idx] <= 0:
                        if self.np_random.uniform(0, 1) > self._fail_prob[robot_id - 1]:  # Si el robot no falla
                            reward += 20
                            self._tasks_states[task_idx] = 1        # Tarea completada
                        else:  # Si el robot falla
                            # reward -= 5
                            self._tasks_states[task_idx] = 2        # Tarea fallada
                        self._tasks_allocations[task_idx] = 0       # Tarea no asignada
                        self._busy_robots[robot_id - 1] = 0         # Robot liberado
                        end_step = 1
            
            # Un episodio acaba si todas las tareas se completan o si el numero de steps alcanza el limite (300)
            terminated = np.all(self._tasks_states != 0) and np.all(self._tasks_states != 3)
            truncated = self.steps > self.max_steps
            reward -= 2 * self._time_lapse
            reward += 100 if terminated else 0
            observation = self._get_obs()
            # info = self._get_info()
                
            end_step = end_step or terminated or truncated
                  
        else:
            while not end_step:
                # Actualizacion del time lapse
                self._time_lapse += 1
                # print("tasks_allocations",self._tasks_allocations)

                # Movimiento de los robots a sus tareas
                # for i, robot_id in enumerate(np.asarray(action)):  # Para cada asignacion
                for i, robot_id in enumerate(self._tasks_allocations):  # Para cada asignacion
                    if robot_id > 0:  # Si la tarea está asignada
                        task_idx = i  # Indice de la tarea
                        task_pos = self._tasks_positions[task_idx]          # Posicion de la tarea
                        robot_pos = self._robots_positions[robot_id - 1]    # Posicion del robot
                        self._busy_robots[robot_id - 1] = 1                 # Si la tarea está asignada, el robot está ocupado

                        # print(f"Tarea: {task_idx}, Robot: {robot_id}")

                        # Movimiento de cada robot
                        if robot_pos != task_pos:
                            task_row, task_col = self._pos2rowcol(task_pos)
                            robot_row, robot_col = self._pos2rowcol(robot_pos)

                            dir_row = np.sign(task_row - robot_row)
                            if dir_row == 0:                                # Evitar movimiento diagonal
                                dir_col = np.sign(task_col - robot_col)
                            else:
                                dir_col = 0

                            robot_row += dir_row
                            robot_col += dir_col
                            robot_pos = self._rowcol2pos(robot_row, robot_col)

                            self._robots_positions[robot_id-1] = robot_pos

                        # Comprueba si el robot ha llegado a su tarea (por modificar: probabilidad fallo)
                        if self._robots_positions[robot_id-1] == task_pos and self._tasks_states[task_idx] != 1 and self._tasks_states[task_idx] != 2:
                            if self._T2end[task_idx] <= 0:
                                if self.np_random.uniform(0, 1) > self._fail_prob[robot_id - 1]:  # Si el robot no falla
                                    reward += 20
                                    self._tasks_states[task_idx] = 1        # Tarea completada
                                else:  # Si el robot falla
                                    # reward -= 5
                                    self._tasks_states[task_idx] = 2        # Tarea fallada
                                self._tasks_allocations[task_idx] = 0       # Tarea no asignada
                                self._busy_robots[robot_id - 1] = 0         # Robot liberado
                                end_step = 1
                            elif self._T2end[task_idx] > 0:
                                self._T2end[task_idx] -= 1
                    
                time.sleep(1)

                # Un episodio acaba si todas las tareas se completan o si el numero de steps alcanza el limite (300)
                terminated = np.all(self._tasks_states != 0) and np.all(self._tasks_states != 3)
                truncated = self.steps > self.max_steps
                reward -= 2
                reward += 100 if terminated else 0
                observation = self._get_obs()
                # info = self._get_info()
                
                end_step = end_step or terminated or truncated


                if self.render_mode == "human":
                    self._render_frame()

                # print("reward",reward)

                # if self._time_lapse == 100:
                #         print("Asignaciones",self._tasks_allocations)

        info = self._get_info()
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
            self.window_size / self.rows
        )  # Tamaño en pixeles de una celda

        # Se dibujan primero las tareas
        for task, task_pos in enumerate(self._tasks_positions):
            task_row, task_col = self._pos2rowcol(task_pos)
            if self._tasks_states[task] == 0:                   # Tarea pendiente
                color = (0, 0, 0)
            elif self._tasks_states[task] == 1:                 # Tarea completada    
                color = (0, 255, 0)
            elif self._tasks_states[task] == 2:                 # Tarea fallada
                color = (255, 0, 0)
            else:                                               # Tarea asignada a un robot  
                color = (150,150,150)

            pygame.draw.rect(
                canvas, color,
                pygame.Rect(pix_square_size * np.array([task_col, task_row]), (pix_square_size, pix_square_size)),
            )


        # Luego, los robots
        for rob_pos, robot_type in zip(self._robots_positions, self._robots_types):
            rob_row, rob_col = self._pos2rowcol(rob_pos)
            if robot_type == 0:                             # Robot de baja capacidad
                color = (0, 0, 255)
            else:                                           # Robot de alta capacidad
                color = (255, 0, 255)

            pygame.draw.circle(
                canvas,
                color,
                (np.array([rob_col, rob_row]) + 0.5) * pix_square_size,
                pix_square_size / 4,
            )

        # A continuacion, los tiempos de las tareas
        for task, task_pos in enumerate(self._tasks_positions):
            task_row, task_col = self._pos2rowcol(task_pos)
            txt = str(self._T2end[task])
            font = pygame.font.Font(None, 36)
            text = font.render(txt, True, (255, 255, 255))
            text_rect = text.get_rect(center=(pix_square_size * (task_col + 0.5), pix_square_size * (task_row + 0.5)))
            canvas.blit(text, text_rect)

        # Finalmente, las lineas del grid
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width = 3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width = 3,
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