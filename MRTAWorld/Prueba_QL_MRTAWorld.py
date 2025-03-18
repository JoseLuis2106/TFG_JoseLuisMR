# Prueba Q-Learning para MRTA
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ubuntu_shared/TFG_JoseLuisMR')
sys.path.append('/home/ubuntu_shared/TFG_JoseLuisMR/gym-examples')
from QLearning import QLearning
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

class StateTransformer:
    """
    Transforma las observaciones en un estado.
    """
    def __init__(self,max_time=100):
        self.time_bins=np.linspace(0, int(max_time), 9)

    def transform(self,observation):
        robots_positions = list(np.array(observation["robots_positions"]).astype(int))
        tasks_positions = list(np.array(observation["tasks_positions"]).astype(int))
        tasks_states = list(np.array(observation["tasks_states"]).astype(int))
        tasks_allocations = list(np.array(observation["tasks_allocations"]).astype(int))
        time_lapse = int(binarize(observation["time_lapse"],self.time_bins))

        return build_state(robots_positions+tasks_positions+tasks_states+tasks_allocations+[time_lapse])

def plot_avg(data,txt):
    """
    Dibuja gráfica de valores medios (de steps o recompensas).
    """
    N=len(data)
    avg=np.empty(N)
    for i in range(N):
        avg[i]=data[max(0,i-100):i+1].mean()
    plt.plot(avg)
    plt.title("Evolution of average "+txt)
    plt.show()


# Entrenamiento y pruebas
if __name__=="__main__":
    env = gym.make('gym_examples/MRTAWorld-v0',rows=8, cols=8, num_robots=2, num_tasks=3)#, render_mode='human')
    learner=QLearning(env,alpha=1e-2,gamma=0.9)
    ft=StateTransformer()
    train=1

    if train:           # Segun si se desea entrenar un nuevo algoritmo o probar uno existente
        n_eps=2000000
        total_steps=np.empty(n_eps)
        total_reward=np.empty(n_eps)
        tab_eps=[]

        for ep in range(1,n_eps):
            # Epsilon-decay
            if ep<n_eps/10:
                learner.epsilon=1.0
            elif ep<n_eps*3/4:
                learner.epsilon = max(0.1, 0.999999 * learner.epsilon)
            else:
                learner.epsilon = max(0.01, 0.99999 * learner.epsilon)

            obs,_ = env.reset()
            state=ft.transform(obs)
            done = False
            nsteps=0
            return_=0
            tab_eps.append(learner.epsilon)

            # print("\nEpisodio",ep)
            while not done:
                # print("Nro steps:",nsteps)

                action = learner.choose_act(state)
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


        print(f"Fin entrenamiento\nMedia ultimos 100 episodios: {total_steps[100:].mean()}\n Epsilon final: {learner.epsilon}\n")
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
        learner.Q=np.load("q_table_mrtaworld.npy", allow_pickle=True).item()        # Carga una Q-Table anterior

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

        while not done:
            action = learner.choose_act(state)
            print(f"Asignacion:    {action}")
            print(f"Estado tareas: {obs['tasks_states']}")

            obs, reward, truncated, terminated, info  = env.step(action)
            next_state=ft.transform(obs)
            done=terminated or truncated

            state=next_state

            nsteps+=1
            return_+=reward

            time.sleep(2)

        print(f"Prueba visual {i+1} \nNumero steps: {nsteps}\nRecompensa: {return_}\n")
        reward_list.append(return_)
        steps_list.append(nsteps)
    env.close()

    print(f"Media de steps de la prueba: {np.mean(steps_list)}")
    print(f"Media de recompensa de la prueba: {np.mean(reward_list)}")

    # Si se resuelve el problema de forma aceptable, se guarda la Q-Table (por decidir qué es aceptable)
    # if np.mean(steps_list)<=size*1.5 and np.mean(reward_list)>200:
    #     np.save("q_table_mrtaworld.npy", learner.Q)