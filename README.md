# **TrafficSim**

```python
env = simpy.Environment()
nodes = [0, 1]
roads = {
    (0,1): (3000, 100, 50),
    (1,0): (3000, 40, 60)
}
rho = np.array([[0.0, 0.1, 0.9],
                [0.2, 0.0, 0.8]])

simulator = TrafficSimulator(rho, nodes, roads, env)
env.process(simulator.run())
env.run(until=100)

```

# **Basic idea and motivation**
- generate traffic data given a $\color{orange} \rho_{i,j}(t) $ of cars going from point $\color{orange}i$ to point $\color{orange}j$. This is a row-wise probability measure, where for each $\color{orange}n \in Nodes$ there is a $\color{orange}p_{i,j}$ of a vehicle appearing and wanting to make the trip $\color{orange}i-j$. So the idea is that this matrix performs some evolution over time.


# **Ambitions**

- given $\color{orange}\rho_{ij}(t)$ we should be able to generate a simulation where we can track each car in each point of the route. In this way we are going to be able to generate traffic volume, traffic speeds, and some other metrics related to each load at each time step.
- an ambitious goal would be to study the ability to regenerate with simulated data $\color{orange}\rightarrow \rho (t)$. So given traffic information, find the pseudo probability matrix that generated it.


# **Model**

- the city is a `DiGraph` were the `edges` are `Roads`. 
- each road holds information
  - `speed`
  - `distance`
  - `capacity` 
  - `vehicles` 
- each road connects to `RoadNodes`
- each `RoadNode` $\color{orange}a$ holds the following information:
  - queue of vehicles wanting to start their journey at $\color{orange}a$ and going to $\color{orange}b$. That is, a queue of vehicles wanting to go through `Road` $\color{orange}a-b$, where $\color{orange}b$ corresponds to each direct edge from a.

# Rules for time evolution

```python
def run(self):
    for a in self.nodes:
        self.new_vehicles(a)
        a.release_queues()
    for road in self.roads:
        road.move_vehicles()
    yield self.env.timeout(1)
```

- add new vehicles following the probability matrix.
- release all queues, one car at a time per queue
- move vehicles in that road
- make time update


# Examples

2 hour periodic discrete shifts.

```python
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
    env.process(simulator.run())
    for i in tqdm(range(1,tf)):
        env.run(until=i)

    simulator.plot_travel_times()
```

![](https://github.com/ivanbelenky/trafficsim/blob/master/assets/images/travel_time.png)


# Next steps

- [ ] at the moment there is no use for simpy really, but could come handly for semaphors
