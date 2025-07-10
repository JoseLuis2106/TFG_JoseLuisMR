# Prueba DQN para MRTA (transformar de QL a DQN)
import gym
from gym import wrappers
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('/home/ubuntu_shared/Prueba_QL')
sys.path.append('/home/ubuntu_shared/Prueba_QL/gym-examples')
from DQN_MRTA import DQN
import gym_examples.envs
import time


# Transformaciones de estados para MRTA
def pos2rowcol(pos,cols):
        """
        Transforma una posición como valor entero a posición como (fila,columna).
        """
        row, col = pos//cols, pos%cols
        return row, col

def build_state(feats):
    """
    Construye el estado a partir de las características.
    """
    return np.concatenate([np.array(feat).flatten() for feat in feats])

class StateTransformer:
    """
    Transforma las observaciones en un estado.
    """
    def __init__(self, rows = 8, cols = 8, num_robots = 2, num_tasks = 3):
        self.rows=rows
        self.cols=cols
        self.num_robots = num_robots
        self.num_tasks = num_tasks

    def state_dim(self):
        """
        Devuelve la dimensión del estado.
        """
        # return self.num_robots * 3 + self.num_tasks * 5                                 # Si posiciones
        # return self.num_robots * self.num_tasks + self.num_tasks * 3 + self.num_robots  # Si distancias
        return self.num_robots * self.num_tasks

    def transform(self,observation):
        robots_positions = list(np.array(observation["robots_positions"]).astype(float))
        tasks_positions = list(np.array(observation["tasks_positions"]).astype(float))
        tasks_states = list((np.array(observation["tasks_states"]).astype(float) / 1.5) - 1)                                # Normalizar estados de tareas
        tasks_allocations = list((np.array(observation["tasks_allocations"]).astype(float) / (self.num_robots / 2)) -1)     # Normalizar asignaciones de tareas
        busy_robots = list((np.array(observation["busy_robots"]).astype(float) / 0.5) - 1)                                  # Normalizar robots ocupados
        tasks_types = list((np.array(observation["tasks_types"]).astype(float) / 1) - 1)                                    # Normalizar tipos de tareas
        robots_types = list((np.array(observation["robots_types"]).astype(float) / 0.5) - 1)                                # Normalizar tipos de robots

        # Usar distancias en lugar de posiciones
        distances = []
        max_dist = np.sqrt((self.rows - 1) ** 2 + (self.cols - 1) ** 2)
        for robot_pos in robots_positions:
            robot_row, robot_col = pos2rowcol(robot_pos, self.cols)
            for task_pos in tasks_positions:
                task_row, task_col = pos2rowcol(task_pos, self.cols)
                dist = (np.sqrt((robot_row - task_row) ** 2 + (robot_col - task_col) ** 2) / (max_dist / 2)) - 1            # Normalizar la distancia
                distances.append(dist)

        # Usar posiciones en lugar de distancias
        # robrows, robcols = zip(*[pos2rowcol(pos, self.cols) for pos in robots_positions]) / max(self.rows, self.cols)   # Normalizar filas y columnas de robots
        # taskrows, taskcols = zip(*[pos2rowcol(pos, self.cols) for pos in tasks_positions]) / max(self.rows, self.cols)  # Normalizar filas y columnas de tareas

        # print(f"Distancias: {distances}")
        # print(f"Tareas: {tasks_states}")
        # print(f"Robots ocupados: {busy_robots}")

        # return build_state([robrows, robcols, taskrows, taskcols, tasks_states, tasks_allocations, tasks_types, robots_types])  # Si posiciones
        # return build_state([distances, tasks_states, tasks_allocations, tasks_types, robots_types])                             # Si distancias
        # return build_state([distances, tasks_states, tasks_allocations, tasks_types])
        # return build_state([distances, tasks_states, tasks_allocations, robots_types])
        return build_state([distances])


def plot_avg(data,txt):
    """
    Dibuja gráfica de valores medios (de steps o recompensas).
    """
    N = len(data)
    avg = np.empty(N)
    for i in range(N):
        avg[i] = data[max(0,i-1000):i+1].mean()
    plt.plot(avg)
    # plt.title("Evolution of average "+txt)
    # plt.show()
    plt.savefig(f"Avg{txt}_{num_tasks}Tasks_CasoX.png")
    plt.clf()


# Entrenamiento y pruebas
if __name__=="__main__":
    rows, cols = 8, 8
    num_robots, num_tasks = 2, 3
    env = gym.make('gym_examples/MRTAWorld-v0', rows = rows, cols = cols, num_robots = num_robots, num_tasks = num_tasks)#, render_mode = 'human')
    
    ft = StateTransformer(rows = rows, cols = cols, num_robots = num_robots, num_tasks = num_tasks)
    state_dim = ft.state_dim()

    learner = DQN(env = env, state_dim = state_dim, alpha = 1e-4, gamma = 0.9, batch_size = 64, buffer_size = 100000)
    
    n_eps = 500000
    # n_eps = 1000
    train = 1
    val = 1

    if train:           # Segun si se desea entrenar un nuevo algoritmo o probar uno existente
        total_steps = np.empty(n_eps)
        total_reward_tab = np.empty(n_eps)
        tab_eps = []
        robot_dists = np.zeros((n_eps, num_robots))
        perc_success = np.zeros(n_eps)
        tab_time = np.empty(n_eps)
        tasks_distribution = np.zeros((n_eps, 2))  # Dos tipos de robots (baja y alta capacidad)
        # tab_qmax = np.empty(n_eps)

        print(f"Estructura de la red neuronal: {learner.QPolicy}") 

        print(f"Iniciando entrenamiento ({n_eps/1000}k episodios)")
        T_start = time.time()

        for ep in range(1,n_eps+1):
            # Epsilon-decay
            if ep < n_eps / 10:
            # if ep == 1:
                learner.epsilon = 1.0
            elif ep < n_eps * 3/4:
                # learner.epsilon = max(0.1, 0.999996 * learner.epsilon)
                learner.epsilon = max(0.1, 0.999992 * learner.epsilon)
            else:
                # learner.epsilon = max(0.01, 0.99999 * learner.epsilon)
                learner.epsilon = max(0.01, 0.99998 * learner.epsilon)
            learner.epsilon = 1

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
                # print("Nro steps:",nsteps)

                # print(f"Estado: {state}")

                # action = learner.choose_act(state,list(obs["tasks_states"]))
                action = learner.choose_act(state, list(obs["tasks_states"]), list(obs["tasks_allocations"]), list(obs["busy_robots"]))
                # print(f"Asignacion:    {action}")
                # print(f"Estado tareas: {obs['tasks_states']}")

                # print(f"Accion: {action}")
                # time.sleep(5)

                obs, reward, truncated, terminated, info = env.step(action)

                # reward /= 10
                
                # print(f"Recompensa:{reward}")

                # obs = {"robots_positions": [18, 38],
                #        "tasks_positions": [21, 35, 42],
                #        "tasks_states": [0, 0, 0],
                #        "tasks_allocations": [0, 0, 0],
                #        "busy_robots": [0, 0],
                #        "tasks_types": [0, 1, 2],
                #        "robots_types": [0, 1]}

                # Distancia recorrida por cada robot cada episodio
                dist_per_robot += np.array(info["dist_travelled"])
                TSteps += info["time_steps"]

                next_state = ft.transform(obs)
                done = terminated or truncated
                
                learner.store_experience(state, action, reward, next_state, done, obs)
                learner.updateQTarget()
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

            if ep % 1000 == 0:
                print(f"Episodio {ep}\nMedia ultimos 100 episodios: {total_steps[max(1,ep-100):ep+1].mean():.2f}")
                print(f"Recompensa media ultimos 100 episodios: {total_reward_tab[max(1,ep-100):ep+1].mean():.2f}")
                print(f"Porcentaje de tareas completadas con exito (ultimos 1000 episodios): {perc_success[max(1,ep-1000):ep+1].mean()*100:.2f}%")
                print(f"Tiempo medio para completar tareas en time-steps ultimos 1000 eps: {tab_time[max(1,ep-1000):ep+1].mean():.2f}\n")

        # Guardar modelo DQN
        torch.save(learner.QTarget.state_dict(), f"DQN_model_{num_tasks}Tasks_CasoX.pth")
        learner.QPolicy.load_state_dict(learner.QTarget.state_dict())                           # Copiar pesos de la red objetivo a la red de política
        loss_tab = np.array(learner.loss_tab)
        qpolicy_tab = np.array(learner.qpolicy_vals)
        qtarget_tab = np.array(learner.qtarget_vals)
        
        # Metricas del entrenamiento (añadir porcentaje de tareas completadas con exito)
        T_end = time.time()
        T_train = T_end-T_start
        print("Fin entrenamiento")
        print(f"Tiempo de entrenamiento: {T_train} s")
        for i in range(num_robots): #Hacer media de ambos robots
            print(f"Distancia media recorrida por robot {i+1} ultimos 1000 eps: {robot_dists[-1000:, i].mean()}")

        print(f"Tiempo medio para completar tareas en time-steps (ultimos 1000 episodios): {tab_time[-1000:].mean()}")

        print(f"Porcentaje de tareas completadas con exito (ultimos 1000 episodios): {perc_success[-1000:].mean()*100:.2f}%")

        print(f"Media ultimos 100 episodios: {total_steps[100:].mean()}\nEpsilon final: {learner.epsilon}")
        print(f"Recompensa media ultimos 100 episodios: {total_reward_tab[100:].mean()}\n")

        plt.plot(total_steps)
        # plt.title("Total steps")
        # plt.show()
        plt.savefig(f"TotalSteps_{num_tasks}Tasks_CasoX.png")
        plt.clf()

        # plot_avg(total_steps,"steps")
        plot_avg(total_steps,"Steps")

        # plot_avg(total_reward_tab,"reward")
        plot_avg(total_reward_tab,"Reward")

        plt.plot(tab_eps)
        plt.title("Evolucion epsilon")
        # plt.show()
        plt.savefig(f"EvolEps.png")
        plt.clf()

        for i in range(num_robots):
            # plot_avg(robot_dists[:, i], f"distance by robot {i+1}")
            plot_avg(robot_dists[:, i], f"Dist_Rob{i+1}")

        # plot_avg(tab_time,"time to complete all tasks (time-steps)")
        plot_avg(tab_time,f"Time2Complete")

        # plot_avg(perc_success,"percentage of success")
        plot_avg(perc_success,"PercSuccess")

        # plot_avg(tasks_distribution[:, 0],"Tasks distribution to low capacity robots")
        plot_avg(tasks_distribution[:, 0],"Tasks_distribution_low_capacity")
        # plot_avg(tasks_distribution[:, 1],"Tasks distribution to high capacity robots")
        plot_avg(tasks_distribution[:, 1],"Tasks_distribution_high_capacity")

        plot_avg(loss_tab,"Loss")
        plot_avg(qpolicy_tab, "QPolicy")
        plot_avg(qtarget_tab, "QTarget")

        # plt.plot(tab_qmax)
        # plt.title("Evolucion max Q")
        # plt.show()


    else:
        learner.QPolicy.load_state_dict(torch.load(f"DQN_model_{num_tasks}Tasks_CasoX.pth"))  #Cargar modelo DQN entrenado
        learner.QPolicy.eval()
        learner.epsilon = 0.0

    
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

        print(f"Iniciando validación ({n_eps/1000}k episodios)")

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

                state = next_state
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
    env = gym.make('gym_examples/MRTAWorld-v0', rows = rows, cols = cols, num_robots = num_robots, num_tasks = num_tasks, render_mode = 'human')
    reward_list = []
    steps_list = []
    learner.epsilon=0.0

    # Iniciar prueba al pulsar boton
    input("Pulsar Enter para iniciar la prueba")

    for i in range(10):
        obs, info = env.reset()
        state = ft.transform(obs)
        done = False
        nsteps = 0
        total_reward = 0
        dist_per_robot = np.zeros(num_robots)
        TSteps = 0        

        print(f"Prueba visual {i+1}")
        while not done:
            # action = learner.choose_act(state)
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

