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

        # Generar todas las combinaciones posibles de acciones
        if isinstance(env.action_space, gym.spaces.MultiDiscrete):
            self.act_space = self._generate_action_combinations(env.action_space.nvec)
        else:
            self.act_space = range(env.action_space.n)

    def getQ(self,state,action):
        """
        Recibe el valor de Q correspondiente a la tupla (estado, acción).
        """
        action = tuple(action) if isinstance(action, np.ndarray) else action
        return self.Q.get((state,action), 0.0)
    
    def choose_act(self,state):
        """
        Elige la acción correspondiente mediante epsilon-greedy. Distingue entre espacio de acciones discreto o multidiscreto.
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = [self.getQ(state, action) for action in self.act_space]
            max_q = np.max(q_values)
            best_indices = [i for i, q in enumerate(q_values) if q == max_q]
            chosen_index = np.random.choice(best_indices)
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                return chosen_index
            else:
                return self.act_space[chosen_index]
        
    def updateQ(self,state,action,reward,next_state):
        '''
        Q(s,a) <- Q(s,a) + alpha*(R + gamma*Qmax(s',a') - Q(s,a))
        '''
        action = tuple(action) if isinstance(action, np.ndarray) else action

        Qant=self.getQ(state,action)
        act=reward + self.gamma*np.max([self.getQ(next_state,next_action) for next_action in self.act_space]) - Qant

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

