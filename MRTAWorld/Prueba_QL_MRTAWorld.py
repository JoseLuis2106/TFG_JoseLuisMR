# Prueba Q-Learning para MRTA
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ubuntu_shared/Prueba_QL')
sys.path.append('/home/ubuntu_shared/Prueba_QL/gym-examples')
# from QLearning import QLearning
from QLearning_MRTA import QLearning
import gym_examples.envs
import time


# Transformaciones de estados para MRTA
def build_state(feats):
    """
    Recibe una lista de observaciones y genera un estado (entero).
    """
    return int("".join(f"{feat:02d}" for feat in feats))

def binarize(value,bins):
    return np.digitize(value,bins=bins)

def pos2rowcol(pos,cols):
        """
        Transforma una posición como valor entero a posición como (fila,columna).
        """
        row, col = pos//cols, pos%cols
        return row, col

class StateTransformer:
    """
    Transforma las observaciones en un estado.
    """
    def __init__(self,max_time=10,rows=8, cols=8):
        self.rows=rows
        self.cols=cols
        self.time_bins=np.linspace(3, int(max_time), 9)
        self.dist_bins = np.linspace(2, np.sqrt(rows**2 + cols**2), 3)

    def transform(self,observation):
        robots_positions = list(np.array(observation["robots_positions"]).astype(int))
        tasks_positions = list(np.array(observation["tasks_positions"]).astype(int))
        tasks_states = list(np.array(observation["tasks_states"]).astype(int))
        # tasks_allocations = list(np.array(observation["tasks_allocations"]).astype(int))
        time_lapse = int(binarize(observation["time_lapse"],self.time_bins))

        tasks_allocations = list(np.zeros(3, dtype=int))

        # return build_state(robots_positions+tasks_positions+tasks_states+tasks_allocations+[time_lapse])

        # Sin tiempo en estado
        # return build_state(robots_positions+tasks_positions+tasks_states+tasks_allocations)

        # Usar distancias en lugar de posiciones
        distances = []
        for robot_pos in robots_positions:
            robot_row, robot_col = pos2rowcol(robot_pos, self.cols)
            for task_pos in tasks_positions:
                task_row, task_col = pos2rowcol(task_pos, self.cols)
                dist = np.sqrt((robot_row - task_row) ** 2 + (robot_col - task_col) ** 2)
                distances.append(binarize(dist, self.dist_bins))

        # return build_state(distances + tasks_states + tasks_allocations + [time_lapse])
            
            # Sin tiempo en estado
        return build_state(distances + tasks_states + tasks_allocations)
    

def plot_avg(data,txt):
    """
    Dibuja gráfica de valores medios (de steps o recompensas).
    """
    N=len(data)
    avg=np.empty(N)
    for i in range(N):
        avg[i]=data[max(0,i-1000):i+1].mean()
    plt.plot(avg)
    plt.title("Evolution of average "+txt)
    plt.show()


# Entrenamiento y pruebas
if __name__=="__main__":
    rows, cols=8, 8
    env = gym.make('gym_examples/MRTAWorld-v0',rows=rows, cols=cols, num_robots=2, num_tasks=3)#, render_mode='human')
    learner=QLearning(env,alpha=1e-2,gamma=0.9)
    ft=StateTransformer(rows=rows, cols=cols)
    train=1

    if train:           # Segun si se desea entrenar un nuevo algoritmo o probar uno existente
        n_eps=9000000
        total_steps=np.empty(n_eps)
        total_reward=np.empty(n_eps)
        tab_eps=[]

        for ep in range(1,n_eps):
            # Epsilon-decay
            if ep<n_eps/10:
                learner.epsilon=1.0
            elif ep<n_eps*3/4:
                learner.epsilon = max(0.1, 0.9999995 * learner.epsilon)
            else:
                learner.epsilon = max(0.01, 0.999998 * learner.epsilon)

            obs,_ = env.reset()
            state=ft.transform(obs)
            done = False
            nsteps=0
            return_=0
            tab_eps.append(learner.epsilon)

            # print("\nEpisodio",ep)
            while not done:
                # print("Nro steps:",nsteps)

                # action = learner.choose_act(state,list(obs["tasks_states"]))
                action = learner.choose_act(state,list(obs["tasks_states"]),list(obs["tasks_allocations"]))
                # print(f"Asignacion:    {action}")
                # print(f"Estado tareas: {obs['tasks_states']}")

                obs, reward, truncated, terminated, info = env.step(action)
                # print(f"Recompensa:{reward}")

                next_state=ft.transform(obs)
                done=terminated or truncated

                learner.updateQ(state,action,reward,next_state)
                state=next_state

                nsteps+=1
                return_+=reward

                # time.sleep(2)

            total_steps[ep]=nsteps
            total_reward[ep]=return_

            if ep%1000==0:
                print(f"Episodio {ep}\nMedia ultimos 100 episodios: {total_steps[max(0,ep-100):ep+1].mean()}")
                print(f"Recompensa media ultimos 100 episodios: {total_reward[max(0,ep-100):ep+1].mean()}\n")


        print(f"Fin entrenamiento\nMedia ultimos 100 episodios: {total_steps[100:].mean()}\n Epsilon final: {learner.epsilon}")
        print(f"Recompensa media ultimos 100 episodios: {total_reward[100:].mean()}\n")
        plt.plot(total_steps[1:])
        plt.title("Total steps")
        plt.show()
        plot_avg(total_steps[1:],"steps")
        plot_avg(total_reward[1:],"reward")
        plt.plot(tab_eps)
        plt.title("Evolucion epsilon")
        plt.show()

    else:
        # learner.Q=np.load("q_table_mrtaworld.npy", allow_pickle=True).item()        # Carga una Q-Table anterior
        data = np.load("q_table_mrtaworld_dists_bins3_noT.npz", allow_pickle=True)        # Carga una Q-Table anterior
        learner.Q = data["Q"].item()

    # Prueba del algoritmo
    env = gym.make('gym_examples/MRTAWorld-v0',rows=8, cols=8, num_robots=2, num_tasks=3, render_mode='human')
    reward_list=[]
    steps_list=[]
    learner.epsilon=0.0

    for i in range(10):
        obs,info = env.reset()
        state=ft.transform(obs)
        done=False
        nsteps=0
        return_=0

        print(f"Prueba visual {i+1}")
        while not done:
            # action = learner.choose_act(state)
            action = learner.choose_act(state,list(obs["tasks_states"]),list(obs["tasks_allocations"]))
            print(f"Asignacion:    {action}")
            print(f"Estado tareas: {obs['tasks_states']}")

            obs, reward, truncated, terminated, info  = env.step(action)
            next_state=ft.transform(obs)
            done=terminated or truncated

            state=next_state

            nsteps+=1
            return_+=reward

            time.sleep(2)

        print(f"Numero steps: {nsteps}\nRecompensa: {return_}\n")
        reward_list.append(return_)
        steps_list.append(nsteps)
    env.close()

    print(f"Media de steps de la prueba: {np.mean(steps_list)}")
    print(f"Media de recompensa de la prueba: {np.mean(reward_list)}")

    # # Tamaño de la Q-Table
    # states=set(state for state, _ in learner.Q.keys())
    # actions=set(action for _, action in learner.Q.keys())
    # num_states = len(states)
    # num_actions = len(actions)

    # print("Size of Q-Table:\n",(num_states,num_actions))

    # Si se resuelve el problema de forma aceptable, se guarda la Q-Table (por decidir qué es aceptable)
    if np.mean(reward_list)>80:
        np.savez_compressed("q_table_mrtaworld.npz", Q=learner.Q)