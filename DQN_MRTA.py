import numpy as np
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import time

# Red neuronal para Q-Learning
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        num_neurons = 16                                # Número de neuronas en las capas ocultas (3 tareas: 16, 4 tareas: 24, 5 tareas: 32) (+4 si prueba con reasignaciones)
        self.fc1 = nn.Linear(state_dim, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.fc3 = nn.Linear(num_neurons, num_neurons)
        self.fc4 = nn.Linear(num_neurons, action_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = F.leaky_relu(self.fc3(x), 0.1)
        x = self.fc4(x)
        return x


# Algoritmo DQN
class DQN:
    def __init__(self, env, state_dim, epsilon = 0.1, alpha = 0.01, gamma = 0.9, batch_size = 64, buffer_size = 100000):
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size

        # Acciones válidas
        self.act_space = self._generate_action_combinations(env.action_space.nvec)
        self.valid_act_space = self._filter_valid_actions(self.act_space)
        self.action_to_idx = {action: idx for idx, action in enumerate(self.valid_act_space)}   # Mapeo de acciones a índices para agilizar el acceso


        self.state_dim = state_dim
        self.action_dim = len(self.valid_act_space)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.QPolicy = QNetwork(self.state_dim, self.action_dim).to(self.device)                # Red neuronal para la política
        self.QTarget = QNetwork(self.state_dim, self.action_dim).to(self.device)                # Red neuronal objetivo
        self.QTarget.load_state_dict(self.QPolicy.state_dict())                                 # Inicializa la red objetivo con los pesos de la red de política
        self.QTarget.eval()                                                                     # Modo evaluación

        self.optimizer = optim.Adam(self.QPolicy.parameters(), lr = self.alpha)                 # Optimizador Adam para la red Q
        self.buffer = deque(maxlen = buffer_size)                                               # Buffer para almacenar experiencias
        self.loss_fn = nn.MSELoss()                                                             # Función de pérdida MSE 
        self.loss_tab = []
        self.qtarget_vals = []                                                          # Lista para almacenar los valores Q de la red objetivo
        self.qpolicy_vals = []                                                          # Lista para almacenar los valores Q de la red de política

        self.steps = 0                          # Recuento de steps
        self.steps_update = 200                 # Cada cuantos steps se actualiza la red objetivo


    def choose_act(self, state, tasks_states, tasks_allocations, busy_robots):
        """
        Elige la acción correspondiente mediante epsilon-greedy.
        """
        valid_actions = [a for a in self.valid_act_space if self._validate_action(a, tasks_states, tasks_allocations, busy_robots)]

        if not valid_actions:
            print("No se encontraron acciones válidas")
            print("Estado:", state)
            print("tasks_states:", tasks_states)
            print("tasks_allocations:", tasks_allocations)
            print("busy_robots:", busy_robots)
            exit()


        if np.random.random() < self.epsilon:
            action =  random.choice(valid_actions)                                                          # Explora una acción aleatoria válida
            return action
        
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)                           # Convierte el estado a tensor y lo envía al dispositivo
            with torch.no_grad():
                q_values = self.QPolicy(state).squeeze(0)                                                           # Obtiene los valores Q de la red de política
                valid_indices = torch.tensor([self.action_to_idx[a] for a in valid_actions], device = self.device)  # Convierte las acciones válidas a índices y los envía al dispositivo
                q_values = q_values[valid_indices]                                                                  # Filtra los valores Q para las acciones válidas
                
                max_q = q_values.max().item()                                                                       # Obtiene el valor Q máximo y su índice
                max_indices = (q_values == max_q).nonzero(as_tuple=True)[0]                                         # Obtiene los índices de las acciones con el valor Q máximo

                chosen_index = max_indices[torch.randint(0, len(max_indices), ())].item()                           # Elige un índice aleatorio entre los máximos
                action = valid_actions[chosen_index]                                                                # Obtiene la acción correspondiente al índice elegido

                return action
        

    def store_experience(self, state, action, reward, next_state, done, obs):
        """
        Almacena la experiencia en el buffer.
        """
        action = tuple(action) if isinstance(action, (list, np.ndarray)) else action
        action_id = self.action_to_idx[action]
        obs_copy = {
                "tasks_states": np.copy(obs["tasks_states"]),
                "tasks_allocations": np.copy(obs["tasks_allocations"]),
                "busy_robots": np.copy(obs["busy_robots"]),
                }
        self.buffer.append((state, action_id, reward, next_state, done, obs_copy))


    def updateQTarget(self):
        '''
        Actualiza la red objetivo con los pesos de la red de política cada ciertos pasos.
        '''
        if len(self.buffer) < self.batch_size:
            return
        
        # Batch aleatorio de experiencias
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones, next_obs = zip(*batch)
        next_obs = {
            "tasks_states": [obs["tasks_states"] for obs in next_obs],
            "tasks_allocations": [obs["tasks_allocations"] for obs in next_obs],
            "busy_robots": [obs["busy_robots"] for obs in next_obs],
        }

        states = torch.tensor(np.array(states), dtype = torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype = torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype = torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype = torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype = torch.float32).unsqueeze(1).to(self.device)
        tasks_states = torch.tensor(np.array(next_obs["tasks_states"]), dtype = torch.float32).to(self.device)
        tasks_allocations = torch.tensor(np.array(next_obs["tasks_allocations"]), dtype = torch.float32).to(self.device)
        busy_robots = torch.tensor(np.array(next_obs["busy_robots"]), dtype = torch.float32).to(self.device)

        q_val_policy = self.QPolicy(states).gather(1, actions)          # Obtiene los valores Q para las distintas acciones tomadas
        self.qpolicy_vals.append(q_val_policy.cpu().detach().numpy()) 

        with torch.no_grad():                                           # Obtención Q-values de la red objetivo
            batch_size = states.size(0)
            valid_mask = torch.zeros((batch_size, self.action_dim), dtype = torch.bool)

            tasks_states = tasks_states.cpu().numpy()
            tasks_allocations = tasks_allocations.cpu().numpy()
            busy_robots = busy_robots.cpu().numpy()

            for  i in range(batch_size):
                valid_actions = [a for a in self.valid_act_space if self._validate_action(a, tasks_states[i], tasks_allocations[i], busy_robots[i])]
                valid_indices = [self.action_to_idx[a] for a in valid_actions]
                valid_mask[i, valid_indices] = True

            q_val_next = self.QTarget(next_states)
            neg_fill = torch.tensor(-1e9, device = self.device)
            q_val_next = q_val_next.masked_fill(~valid_mask.to(self.device), neg_fill)

            q_val_next = q_val_next.max(dim = 1, keepdim = True).values
            q_val_target = rewards + (self.gamma * q_val_next * (1 - dones))
            self.qtarget_vals.append(q_val_target.cpu().detach().numpy())
        
        loss = self.loss_fn(q_val_policy, q_val_target)                 # Calcula la pérdida entre los valores Q actuales y los valores Q objetivo
        self.loss_tab.append(loss.item())                               # Almacena la pérdida para análisis posterior

        self.optimizer.zero_grad()                                      # Limpia los gradientes
        loss.backward()                                                 # Calcula los gradientes

        self.optimizer.step()                                           # Actualiza los pesos de la red de política

        self.steps += 1
        if self.steps % self.steps_update == 0:                         # Actualiza la red objetivo cada ciertos pasos
            self.QTarget.load_state_dict(self.QPolicy.state_dict())
            

    def _generate_action_combinations(self, nvec):
        """
        Genera todas las posibles combinaciones de acciones en un espacio multidiscreto.
        """
        action_combinations = []
        self._generate_combinations_recursive(nvec, 0, [], action_combinations)
        return action_combinations


    def _generate_combinations_recursive(self, nvec, index, current_combination, action_combinations):
        """
        Function recursiva que genera todas las posibles combinaciones de acciones.
        """
        if index == len(nvec):
            action_combinations.append(tuple(current_combination))
            return
        
        for i in range(nvec[index]):
            self._generate_combinations_recursive(nvec, index + 1, current_combination + [i], action_combinations)


    def _filter_valid_actions(self, actions):
        """ 
        Filtra las acciones que siempre serán inválidas (asignando un robot a más de una tarea). 
        """
        return [action for action in actions if not np.any(np.bincount(np.array(action)[np.array(action) > 0])[1:] > 1)]


    def _validate_action(self, action, tasks_states, tasks_allocations, busy_robots):
        """
        Comprueba si alguna de las tareas asignadas ya estan completadas o falladas
        """
        action = np.asarray(action) 
        
        busy_robots_set = set()                         # Conjunto para almacenar los robots ocupados
        for robot, busy in enumerate(busy_robots):
            if busy:
                busy_robots_set.add(robot + 1)          # +1 porque los robots se numeran desde 1 en lugar de 0


        for task, robot in enumerate(action):
            if (tasks_states[task] == 1 or tasks_states[task] == 2) and robot > 0:      # Si la tarea ya está completada (1) o fallada (2) y se le ha asignado un robot
                return False

            # Comentar en prueba con reasignaciones
            if tasks_allocations[task] > 0 and tasks_allocations[task] != robot:        # Si la tarea ya tiene un robot asignado y no es el mismo
                return False
            
            # Comentar en prueba con reasignaciones
            if robot in busy_robots_set and tasks_allocations[task] != robot:           # Si el robot ya está asignado a otra tarea diferente
                return False
        
        return True


