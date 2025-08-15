# Prueba Q-Learning para MRTA
import gym
from gym import wrappers
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ubuntu_shared/Prueba_QL')
sys.path.append('/home/ubuntu_shared/Prueba_QL/gym-examples')
from QLearning_MRTA import QLearning
import gym_examples.envs
import time
import pickle


# Transformaciones de estados para MRTA
def build_state(feats):
    """
    Recibe una lista de observaciones y genera un estado (entero).
    """
    return int("".join(f"{feat:1d}" for feat in feats))

def binarize(value,bins):
    return np.digitize(value,bins=bins)

def pos2rowcol(pos,cols):
    """
    Transforma una posición como valor entero a posición como (fila, columna).
    """
    row, col = pos//cols, pos%cols
    return row, col

class StateTransformer:
    """
    Transforma las observaciones en un estado.
    """
    def __init__(self, max_time = 10, rows = 8, cols = 8):
        self.rows=rows
        self.cols=cols
        # self.time_bins=np.linspace(3, int(max_time), 9)
        self.dist_bins = np.linspace(2, np.sqrt(rows**2 + cols**2), 4)

    def transform(self,observation):
        robots_positions = list(np.array(observation["robots_positions"]).astype(int))
        tasks_positions = list(np.array(observation["tasks_positions"]).astype(int))
        tasks_states = list(np.array(observation["tasks_states"]).astype(int))
        tasks_allocations = list(np.array(observation["tasks_allocations"]).astype(int))
        busy_robots = list(np.array(observation["busy_robots"]).astype(int))
        tasks_types = list(np.array(observation["tasks_types"]).astype(int))
        robots_types = list(np.array(observation["robots_types"]).astype(int))

        # Usar distancias en lugar de posiciones
        distances = []
        for robot_pos in robots_positions:
            robot_row, robot_col = pos2rowcol(robot_pos, self.cols)
            for task_pos in tasks_positions:
                task_row, task_col = pos2rowcol(task_pos, self.cols)
                dist = np.sqrt((robot_row - task_row) ** 2 + (robot_col - task_col) ** 2)
                distances.append(binarize(dist, self.dist_bins))

        # print(f"Distancias: {distances}")
        # print(f"Tareas: {tasks_states}")
        # print(f"Robots ocupados: {busy_robots}")

        return build_state(distances + tasks_states + tasks_allocations + tasks_types + robots_types)
        # return build_state(distances + tasks_states + tasks_allocations + tasks_types)                      # Prueba 5
        # return build_state(distances + tasks_states + tasks_allocations + robots_types)                     # Prueba 6  



def plot_avg(data,txt):
    """
    Dibuja gráfica de valores medios (de steps o recompensas).
    """
    N = len(data)
    avg = np.empty(N)
    for i in range(N):
        avg[i] = data[max(0,i-1000):i+1].mean()
    plt.plot(avg)
    plt.title("Evolution of average "+txt)
    # plt.show()
    plt.grid()
    plt.savefig(f"Avg{txt}_{num_tasks}Tasks_CasoX.png")
    plt.clf()


# Entrenamiento y pruebas
if __name__=="__main__":
    rows, cols = 6, 6
    num_robots, num_tasks = 2, 4
    env = gym.make('gym_examples/MRTAWorld-v0',rows = rows, cols = cols, num_robots = num_robots, num_tasks = num_tasks)#, render_mode = 'human')
    learner = QLearning(env, alpha = 1e-2, gamma = 0.9)
    ft = StateTransformer(rows = rows, cols = cols)

    n_eps = 10000000
    # n_eps = 10000
    train = 0
    val = 1

    if train:           # Segun si se desea entrenar un nuevo algoritmo o probar uno existente
        total_steps = np.zeros(n_eps)
        total_reward_tab = np.zeros(n_eps)
        tab_eps = []
        robot_dists = np.zeros((n_eps, num_robots))
        perc_success = np.zeros(n_eps)
        tab_time = np.zeros(n_eps)
        tasks_distribution = np.zeros((n_eps, 2))  # Dos tipos de robots (baja y alta capacidad)
        # tab_qmax = np.empty(n_eps)

        print(f"Iniciando entrenamiento ({int(n_eps/1000000)}M episodios)")
        T_start = time.time()

        for ep in range(1,n_eps+1):
            # Epsilon-decay
            if ep < n_eps / 10:
                learner.epsilon=1.0
            elif ep < n_eps * 3/4:
                learner.epsilon = max(0.1, 0.9999996 * learner.epsilon)
            else:
                learner.epsilon = max(0.01, 0.999999 * learner.epsilon)

            obs, info = env.reset()
            state = ft.transform(obs)
            done = False
            nsteps = 0
            total_reward = 0
            TSteps = 0
            tab_eps.append(learner.epsilon)
            dist_per_robot = np.zeros(num_robots)

            # print("\nEpisodio",ep)

            while not done:
                # print(f"Nro ep: {ep}      Nro step:{nsteps}")

                # action = learner.choose_act(state,list(obs["tasks_states"]))
                # action = learner.choose_act(state, list(obs["tasks_states"]), list(obs["tasks_allocations"]))
                action = learner.choose_act(state, list(obs["tasks_states"]), list(obs["tasks_allocations"]), list(obs["busy_robots"]))
                # print(f"Asignacion:    {action}")
                # print(f"Estado tareas: {obs['tasks_states']}")

                obs, reward, truncated, terminated, info = env.step(action)
                # print(f"Recompensa:{reward}")

                # Distancia recorrida por cada robot cada episodio
                dist_per_robot += np.array(info["dist_travelled"])
                TSteps += info["time_steps"]

                next_state = ft.transform(obs)
                done = terminated or truncated

                learner.updateQ(state,action,reward,next_state)
                state = next_state
                # print(state)

                nsteps += 1
                total_reward += reward

                # time.sleep(2)


            # try:
            #     qmax = max(learner.Q.values())
            # except:
            #     qmax = -1

            total_steps[ep-1] = nsteps
            total_reward_tab[ep-1] = total_reward
            robot_dists[ep-1] = dist_per_robot
            tab_time[ep-1] = TSteps
            perc_success[ep-1] = info["perc_success"]
            tasks_distribution[ep-1] = info["tasks_distribution"]
            # tab_qmax[ep-1] = qmax

            if ep % 2000 == 0:
                print(f"Episodio {ep}\nMedia ultimos 1000 episodios: {total_steps[max(0,ep-1000):ep+1].mean():.2f}")
                print(f"Recompensa media ultimos 1000 episodios: {total_reward_tab[max(0,ep-1000):ep+1].mean():.2f}")
                print(f"Porcentaje de tareas completadas con exito (ultimos 1000 episodios): {perc_success[max(0,ep-1000):ep+1].mean()*100:.2f}%")
                print(f"Tiempo medio para completar tareas en time-steps ultimos 1000 eps: {tab_time[max(0,ep-1000):ep+1].mean():.2f}\n")

        # Guardar Q-Table
        np.savez_compressed(f"q_table_mrtaworld_{num_tasks}Tasks_CasoX.npz", Q = learner.Q)
        # with open(f"q_table_mrtaworld_{num_tasks}Tasks_CasoX.pkl", "wb") as f:
        #     pickle.dump(learner.Q, f, protocol=pickle.HIGHEST_PROTOCOL)


        # Metricas del entrenamiento (añadir porcentaje de tareas completadas con exito)
        T_end = time.time()
        T_train = T_end-T_start
        print("Fin entrenamiento")
        print(f"Tiempo de entrenamiento: {T_train} s")
        for i in range(num_robots): #Hacer media de ambos robots
            print(f"Distancia media recorrida por robot {i+1} ultimos 1000 eps: {robot_dists[-1000:, i].mean()}")

        print(f"Tiempo medio para completar tareas en time-steps (ultimos 1000 episodios): {tab_time[-1000:].mean()}")

        print(f"Porcentaje de tareas completadas con exito (ultimos 1000 episodios): {perc_success[-1000:].mean()*100:.2f}%")

        print(f"Media ultimos 1000 episodios: {total_steps[1000:].mean()}\nEpsilon final: {learner.epsilon}")
        print(f"Recompensa media ultimos 1000 episodios: {total_reward_tab[1000:].mean()}\n")

        plt.plot(total_steps)
        plt.title("Total steps")
        # plt.show()
        plt.grid()
        plt.savefig(f"TotalSteps_{num_tasks}Tasks_CasoX.png")
        plt.clf()

        # plot_avg(total_steps,"steps")
        plot_avg(total_steps,"Steps")

        # plot_avg(total_reward_tab,"reward")
        plot_avg(total_reward_tab,"Reward")

        plt.plot(tab_eps)
        plt.title("Evolucion epsilon")
        # plt.show()
        plt.grid()
        plt.savefig(f"EvolEps_{num_tasks}Tasks_CasoX.png")
        plt.clf()

        for i in range(num_robots):
            # plot_avg(robot_dists[:, i], f"distance by robot {i+1}")
            plot_avg(robot_dists[:, i], f"Dist_Rob{i+1}")

        # plot_avg(tab_time,"time to complete all tasks (time-steps)")
        plot_avg(tab_time,f"Time2Complete")

        # plot_avg(perc_success,"percentage of success")
        plot_avg(perc_success,"PercSuccess")

        # plot_avg(tasks_distribution,"Tasks distribution to low capacity robots")
        plot_avg(tasks_distribution[:, 0],"Tasks_distribution_low_capacity")
        # plot_avg(tasks_distribution,"Tasks distribution to high capacity robots")
        plot_avg(tasks_distribution[:, 1],"Tasks_distribution_high_capacity")

        # plt.plot(tab_qmax)
        # plt.title("Evolucion max Q")
        # plt.show()


    else:
        data = np.load(f"q_table_mrtaworld_{num_tasks}Tasks_Caso2.npz", allow_pickle=True)        # Carga una Q-Table anterior
        learner.Q = data["Q"].item()
        # with open(f"q_table_mrtaworld_{num_tasks}Tasks_CasoX.pkl", "rb") as f:
        #     learner.Q = pickle.load(f)

    # Evaluación del algoritmo
    if val:
        env = gym.make('gym_examples/MRTAWorld-v0', rows = rows, cols = cols, num_robots = num_robots, num_tasks = num_tasks)
        reward_list = []
        steps_list = []
        n_eps = int(n_eps / 4)
        learner.epsilon = 0.0
        total_steps = np.empty(n_eps)
        total_reward_tab = np.empty(n_eps)
        tab_eps = []
        robot_dists = np.zeros((n_eps, num_robots))
        perc_success = np.zeros(n_eps)
        tab_time = np.empty(n_eps)
        tasks_distribution = np.zeros((n_eps, 2))  # Dos tipos de robots (baja y alta capacidad)

        print(f"Iniciando validación ({n_eps/1000000}M episodios)")

        for ep in range(1,n_eps+1):
            obs, info = env.reset()
            state = ft.transform(obs)
            done = False
            nsteps = 0
            total_reward = 0
            dist_per_robot = np.zeros(num_robots)
            TSteps = 0
            
            while not done:
                # action = learner.choose_act(state)
                # action = learner.choose_act(state, list(obs["tasks_states"]), list(obs["tasks_allocations"]))
                action = learner.choose_act(state, list(obs["tasks_states"]), list(obs["tasks_allocations"]), list(obs["busy_robots"]))
                # print(f"Asignacion:    {action}")
                # print(f"Estado tareas: {obs['tasks_states']}")

                obs, reward, truncated, terminated, info  = env.step(action)
                next_state = ft.transform(obs)
                done = terminated or truncated

                state=next_state
                # print(state)

                nsteps += 1
                total_reward += reward

                TSteps += info["time_steps"]
                dist_per_robot += np.array(info["dist_travelled"])

                # time.sleep(2)

            if ep % int(n_eps/5) == 0:
                print(f"Episodio {ep}")
            
            total_steps[ep-1] = nsteps
            total_reward_tab[ep-1] = total_reward
            robot_dists[ep-1] = dist_per_robot
            tab_time[ep-1] = TSteps
            perc_success[ep-1] = info["perc_success"]
            tasks_distribution[ep-1] = info["tasks_distribution"]

            # print(f"Tiempo para completar tareas en time-steps: {TSteps}")
            # print(f"Distancia recorrida por cada robot: {dist_per_robot}")

            # print(f"Porcentaje de tareas completadas con exito: {info['perc_success']*100:.2f}%")

            # print(f"Numero steps: {nsteps}\nRecompensa: {total_reward}\n")

        print("Fin evaluación")
        for i in range(num_robots): #Hacer media de ambos robots
            print(f"Distancia media recorrida por robot {i+1}: {robot_dists[:, i].mean()}")

        print(f"Media de tareas asignadas a robot baja capacidad: {tasks_distribution[:, 0].mean()}")
        print(f"Media de tareas asignadas a robot alta capacidad: {tasks_distribution[:, 1].mean()}")

        print(f"Tiempo medio para completar tareas en time-steps: {tab_time.mean()}")

        print(f"Porcentaje de tareas completadas con éxito: {perc_success.mean()*100:.2f}%")

        print(f"Media steps: {total_steps.mean()}")
        print(f"Recompensa media: {total_reward_tab.mean()}\n")


    # Prueba del algoritmo
    env = gym.make('gym_examples/MRTAWorld-v0',rows = rows, cols = cols, num_robots = num_robots, num_tasks = num_tasks, render_mode = 'human')
    reward_list = []
    steps_list = []
    learner.epsilon = 0.0

    # Iniciar prueba al pulsar boton
    input("Pulsar Enter para iniciar la prueba")

    for ep in range(10):
        obs, info = env.reset()
        state = ft.transform(obs)
        done = False
        nsteps = 0
        total_reward = 0
        dist_per_robot = np.zeros(num_robots)
        TSteps = 0        

        print(f"Prueba visual {ep+1}")
        while not done:
            # action = learner.choose_act(state)
            # action = learner.choose_act(state, list(obs["tasks_states"]), list(obs["tasks_allocations"]))
            action = learner.choose_act(state, list(obs["tasks_states"]), list(obs["tasks_allocations"]), list(obs["busy_robots"]))
            # print(f"Asignacion:    {action}")
            # print(f"Estado tareas: {obs['tasks_states']}")

            obs, reward, truncated, terminated, info  = env.step(action)
            next_state = ft.transform(obs)
            done = terminated or truncated

            state=next_state
            # print(state)

            nsteps += 1
            total_reward += reward

            TSteps += info["time_steps"]
            dist_per_robot += np.array(info["dist_travelled"])

            # time.sleep(2)

        print(f"Tiempo para completar tareas en time-steps: {TSteps}")
        print(f"Distancia recorrida por cada robot: {dist_per_robot}")

        print(f"Porcentaje de tareas completadas con exito: {info['perc_success']*100:.2f}%")

        print(f"Numero steps: {nsteps}\nRecompensa: {total_reward}\n")
        reward_list.append(total_reward)
        steps_list.append(nsteps)

    env.close()


    print(f"Media de steps de la prueba: {np.mean(steps_list)}")
    print(f"Media de recompensa de la prueba: {np.mean(reward_list)}")


    # Si se resuelve el problema de forma aceptable, se guarda la Q-Table (por decidir qué es aceptable)
    # if np.mean(reward_list) > 100:
    #     np.savez_compressed("q_table_mrtaworld_T2End.npz", Q = learner.Q)
