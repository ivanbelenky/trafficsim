from typing import NewType, Tuple, Sequence, Any, Dict, Union, Callable

import numpy as np
from numpy import random as rd
import networkx as nx
import simpy
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.style.use("dark_background")

RoadInfo = NewType('RoadInfo', Tuple[float, float, int])

DT_IN_SECONDS = 1

class Road:
    '''
    Roads are the main component of the RoadMap, this are the edges of the 
    graph, they hold information on length lanes and any other parameters
    that could be used to model vehicle transport through those lanes. 
    '''
    def __init__(self, key, info: RoadInfo, roadmap):
        length, speed_limit, capacity = info
        self.a, self.b = key
        self.length = length
        self.speed_limit = speed_limit
        self.capacity = capacity
        self.roadmap = roadmap

        self.vehicles = []

    @property
    def capacity_left(self):
        return self.capacity - len(self.vehicles)

    def add_vehicle(self, vehicle):
        vehicle.distance = 0
        vehicle.waiting = False
        vehicle.speed = self.speed_limit
        self.vehicles.append(vehicle)

    def move_vehicles(self):
        max_d = self.length
        for vehicle in reversed(self.vehicles): 
            delta_d = vehicle.speed * DT_IN_SECONDS
            new_distance = min(vehicle.distance + delta_d, max_d)

            # Check when reaching node b
            if new_distance == self.length:
                # If we reached the destination, it dissapears
                # without causing congestion, magic abduction
                if vehicle.reached_dst(self.b):
                    vehicle.finish()
                    self.vehicles.remove(vehicle) 
                elif vehicle.waiting:
                    # If it is waiting, we dont have to do anything
                    max_d = max_d - self.length/self.capacity
                else:        
                    # If it doesn't reach dst, we look at the next
                    # path, and we enque to the next road
                    c = vehicle.route[-1] # c := next node
                    vehicle.distance = max_d
                    max_d = max_d-self.length/self.capacity
                    self.roadmap.nodes[self.b].to[c].enqueue(vehicle, self)
                    # By enqueuing the road we make sure that whenever the
                    # vehicle is released from the queue, is going to be removed
                    # as well from the current road. This is, while this vehicle
                    # coming from a, reaching b, is waiting to get into road b-c 
                    # it is going to wait occupying space in road a-b.
                    vehicle.waiting = True
                
            if new_distance < max_d:
                vehicle.distance = new_distance


class RoadNode:
    '''
    RoadNodes hold information of the particular node, and for each of
    the edges that this nodes is connected to, it creates a queue
    that could be populated by cars reaching the specified node willing
    to go to node b.
    '''
    def __init__(self, node: Any, connected_to: Sequence[Any]=None, roadmap=None):
        self.a = node
        self.roadmap = roadmap
        self.to = {b: RoadNodeQueue() for _, b in connected_to}

    def release_queue(self):
        # Liberating queues may need randomization to average the 
        # effect of multiple connected nodes liberating always at
        # the same order
        for b, queue in self.to.items(): 
            road = self.roadmap.roads[(self.a, b)]
            if road.capacity_left > 0 and not queue.is_empty():
                vehicle, prev_road = queue.dequeue()
                if prev_road:
                    prev_road.vehicles.remove(vehicle)
                vehicle.route.pop()
                road.add_vehicle(vehicle)

class RoadMap:
    '''
    RoadMap holds the graph version of the roads for a particular
    section of a map. It holds RoadNodes, being interesting points
    for discretizing this graph, commonly interpreted as common
    sources or destinations. 
    '''
    def __init__(self, nodes: Sequence[Any], roads: Dict[Any, RoadInfo]):
        self.roads = {(a, b): Road((a,b),road, self) for (a, b), road in roads.items()}
        self.graph = self._create_graph(nodes, self.roads)
        self.nodes = {node: RoadNode(node, connected_to=self.graph.edges(node), roadmap=self) 
                      for node in nodes}
        
    def _create_graph(self, nodes, roads) -> nx.Graph:
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for ab, road in roads.items():
            a, b = ab  
            if a not in nodes or b not in nodes:
                raise ValueError(f"Node {a} or {b} not in nodes")
            
            graph.add_edge(a, b, length=road.length, capacity=road.capacity,
                speed_limit=road.speed_limit)
            
        return graph


class RoadNodeQueue:
    def __init__(self):
        self.vehicles = []
    
    def enqueue(self, vehicle, road=None):
        self.vehicles.insert(0, (vehicle, road))
    
    def dequeue(self):
        return self.vehicles.pop()
    
    def is_empty(self):
        return self.vehicles == []


class Vehicle:
    def __init__(self, a, b, route, env):
        self.src = a
        self.dst = b
        self.route = route
        self.env = env
        self.start = env.now
        self.speed = 0
        self.distance = 0
        self.waiting = True
        self.end = None

    def reached_dst(self, c):
        return c == self.dst

    def finish(self):
        self.end = self.env.now
        
    def travel_time(self):
        if self.end:
            return self.end - self.start


class TrafficSimulator:
    def __init__(self, rho: Union[np.ndarray, Callable], nodes: Sequence[Any], 
        roads:Dict[Any, RoadInfo], env: simpy.Environment):
        
        self._validate_attr(rho, nodes)
        self._rho = rho
        
        self.roadmap = RoadMap(nodes, roads)
        self.roads = self.roadmap.roads
        self.nodes = self.roadmap.nodes
        self.graph = self.roadmap.graph
        
        self.vehicles = []
        self.env = env
        self.time = 0
    
    @property
    def rho(self):
        if isinstance(self._rho, np.ndarray):
            return self._rho
        return self._rho(self.env.now*DT_IN_SECONDS)

    def _validate_attr(self, rho, nodes):
        n = len(nodes)
        if not isinstance(rho, (np.ndarray, Callable)):
            raise ValueError("Rho must be a fixed array or a callable")

        if isinstance(rho, np.ndarray):
            if rho.shape != (n, n+1):
                raise ValueError(f"Rho matrix must be of shape {(n, n+1)}")

        if isinstance(rho, Callable):
            if rho.__code__.co_argcount != 1:
                raise ValueError("Rho function must have (t) as argument")

    def new_vehicles(self, a):
        # Last column is Pr{no new vehicle}
        p = self.rho[a,:-1] 
        p_new = p.sum()
        # Could have multiple new vehicles
        while rd.random() < p_new:
            b = rd.choice(list(self.nodes.keys()), p=p/p_new)
            route = nx.shortest_path(self.graph, a, b)[::-1] # should/could vary
            new_vehicle = Vehicle(a, b, route, self.env)
            self.nodes[a].to[b].enqueue(new_vehicle)
            
            self.vehicles.append(new_vehicle)

    def get_statistics(self):
        ab_travel_times = {}
        for v in self.vehicles:
            ab = (v.src, v.dst)
            if ab in ab_travel_times:
                ab_travel_times[ab].append((v.start, v.end))
            else:
                ab_travel_times[ab] = [(v.start, v.end)]
        
        return ab_travel_times

    def plot_travel_times(self, nodes: Union[Sequence[Any], None]=None):
        if not nodes:
            nodes = self.nodes.keys()
        plt.figure(figsize=(10, 10))
        ab_travel_times = self.get_statistics()
        for ab, times in ab_travel_times.items():
            ab_scatter = []
            for start, end in times:
                if end:
                    ab_scatter.append([start*DT_IN_SECONDS/60, 
                                       (end-start)*DT_IN_SECONDS/60])
            ab_scatter = np.array(ab_scatter)
            plt.scatter(ab_scatter[:,0], ab_scatter[:,1], s=1, label=f"{ab}")
        plt.xlabel("Time (minutes)"); plt.ylabel("Travel time (minutes)")
        plt.legend(loc=1); plt.show()

    def _run(self):
        while True:
            for a, node in self.nodes.items():
                self.new_vehicles(a)
                node.release_queue()
            for ab, road in self.roads.items():
                road.move_vehicles()
            yield self.env.timeout(1)

    def run(self, sim_time):
        self.env.process(self._run())
        for i in tqdm(range(1, sim_time+1)):
            self.env.run(until=i)