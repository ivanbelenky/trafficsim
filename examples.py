import simpy
import numpy as np
from trafficsim import TrafficSimulator, DT_IN_SECONDS
from tqdm import tqdm


def fixed_rho(tf):
    env = simpy.Environment()
    nodes = [0, 1]
    roads = {
        (0,1): (3000, 30, 100),
        (1,0): (3000, 50, 10)
    }
    
    rho = np.array([[0.0, 0.7, 0.3],
                    [0.5, 0.0, 0.5]])
    
    simulator = TrafficSimulator(rho, nodes, roads, env)
    simulator.run(tf)
    simulator.plot_travel_times()


def dynamic_rho(tf):
    env = simpy.Environment()
    nodes = [0, 1]
    roads = {
        (0,1): (100, 10, 3),
        (1,0): (100, 10, 3)
    }

    def rho(t):
        if t//7200 % 2 == 0:
            return np.array([[0.0, 0.25, 0.75],
                            [0.15, 0.0, 0.85]])
        return np.array([[0.0, 0.15, 0.85],
                        [0.25, 0.0, 0.75]])
    
    simulator = TrafficSimulator(rho, nodes, roads, env)
    simulator.run(tf)
    simulator.plot_travel_times()



if __name__ == "__main__":
    #fixed_rho()
    dynamic_rho(int(3600*24))