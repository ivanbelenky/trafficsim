from typing import NewType, Tuple, Sequence, Any, Dict

import numpy as np
from numpy import random as rd
import networkx as nx
import simpy


import networkx as nx

RoadInfo = NewType('RoadInfo', Tuple[float, float, int])

DT_IN_SECONDS = 10

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
                    vehicle.distance = self.length
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
        self.speed = 0
        self.distance = 0
        self.waiting = True

    def reached_dst(self, c):
        return c == self.dst


class TrafficSimulator:
    def __init__(self, rho: np.ndarray, nodes: Sequence[Any], 
        roads:Dict[Any, RoadInfo], env: simpy.Environment):
        
        self._validate_attr(rho, nodes)
        self.rho = rho
        
        self.roadmap = RoadMap(nodes, roads)
        self.roads = self.roadmap.roads
        self.nodes = self.roadmap.nodes
        self.graph = self.roadmap.graph
        
        self.vehicles = []
        self.env = env
        self.time = 0
    
    def _validate_attr(self, rho, nodes):
        n = len(nodes)
        if rho.shape != (n, n+1):
            raise ValueError(f"Rho matrix must be of shape {(n, n+1)}")

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

    def run(self):
        while True:
            for a, node in self.nodes.items():
                self.new_vehicles(a)
                node.release_queue()
            for ab, road in self.roads.items():
                road.move_vehicles()
            yield self.env.timeout(1)


if __name__ == "__main__":
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

    print([v.distance for v in simulator.vehicles])