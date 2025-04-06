import numpy as np
import gym

#Algoritmo Q-Learning
class QLearning:
    def __init__(self,env,epsilon=0.1,alpha=0.1,gamma=0.9):
        self.Q={}       # Q-table
        self.env=env
        self.epsilon=epsilon
        self.alpha=alpha
        self.gamma=gamma

        self.act_space = self._generate_action_combinations(env.action_space.nvec)
        self.valid_act_space = self._filter_valid_actions(self.act_space)
        

    def getQ(self,state,action):
        """
        Recibe el valor de Q correspondiente a la tupla (estado, acción).
        """
        action = tuple(action) if isinstance(action, np.ndarray) else action
        return self.Q.get((state,action), 0.0)
    
    def choose_act(self,state,tasks_states,tasks_allocations):
        """
        Elige la acción correspondiente mediante epsilon-greedy. Distingue entre espacio de acciones discreto o multidiscreto.
        """
        valid_actions = [a for a in self.valid_act_space if self._validate_action(a, tasks_states, tasks_allocations)]

        # print(f"Acciones válidas: {valid_actions}")

        if np.random.random() < self.epsilon:
            return valid_actions[np.random.randint(len(valid_actions))]
        else:
            q_values = [self.getQ(state, action) for action in valid_actions]
            max_q = np.max(q_values)
            best_indices = [i for i, q in enumerate(q_values) if q == max_q]
            chosen_index = np.random.choice(best_indices)
            return valid_actions[chosen_index]
        
    def updateQ(self,state,action,reward,next_state):
        '''
        Q(s,a) <- Q(s,a) + alpha*(R + gamma*Qmax(s',a') - Q(s,a))
        '''
        action = tuple(action) if isinstance(action, np.ndarray) else action

        Qant=self.getQ(state,action)
        act=reward + self.gamma*np.max([self.getQ(next_state,next_action) for next_action in self.valid_act_space]) - Qant

        self.Q[(state,action)]=Qant + self.alpha*act

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

    def _validate_action(self,action,tasks_states,tasks_allocations):
        """
        Comprueba si alguna de las tareas asignadas ya estan completadas o falladas
        """
        action = np.asarray(action) 

        # print("Asignaciones",tasks_allocations)
        # print("Estados",tasks_states)
        
        busy_robots = set()
        for i, state in enumerate(tasks_states):
            if state == 3 and tasks_allocations[i] != 0:
                busy_robots.add(tasks_allocations[i])

        # print("robots ocupados",busy_robots)

        for task, robot in enumerate(action):
            if robot > 0: 
                if tasks_states[task] != 0:       # Si la tarea no está pendiente y se le ha asignado un robot
                    # print(f"Tarea {task} no pendiente")
                    return False
                
                if robot in busy_robots: # No poder asignar un robot ya asignado a otra tarea
                    return False

            
        return True


