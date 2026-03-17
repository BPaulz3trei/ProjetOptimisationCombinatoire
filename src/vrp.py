import numpy as np
import matplotlib.pyplot as plt

class Client:
    def __init__(self, id, x, y, demand):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand

class VRPInstance:

    def __init__(self, name, depot, clients, capacity):
        self.name = name
        self.depot = depot
        self.clients = clients
        self.capacity = capacity
        self.dist_matrix = self._compute_distances()

    def _compute_distances(self):
        n = len(self.clients) + 1
        coords = np.zeros((n, 2))
        coords[0] = self.depot
        for c in self.clients:
            coords[c.id] = (c.x, c.y)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i][j] = np.sqrt(
                    (coords[i][0] - coords[j][0])**2 +
                    (coords[i][1] - coords[j][1])**2
                )
        return dist

    @property
    def n_clients(self):
        return len(self.clients)

    def total_demand(self):
        return sum(c.demand for c in self.clients)


class VRPSolution:

    def __init__(self, routes, instance):
        self.routes = routes
        self.instance = instance

    def total_distance(self):
        total = 0.0
        for route in self.routes:
            for i in range(len(route) - 1):
                total += self.instance.dist_matrix[route[i]][route[i + 1]]
        return total

    def is_feasible(self):
        visited = set()
        for route in self.routes:
            load = 0
            for node in route[1:-1]:
                if node in visited:
                    return False
                visited.add(node)
                load += self.instance.clients[node - 1].demand
                if load > self.instance.capacity:
                    return False
        all_clients = {c.id for c in self.instance.clients}
        return visited == all_clients

    def n_vehicles(self):
        return len(self.routes)

    def __repr__(self):
        return (f"VRPSolution("
                f"distance={self.total_distance():.2f}, "
                f"vehicles={self.n_vehicles()}, "
                f"feasible={self.is_feasible()})")


def generate_random_instance(n_clients, capacity, seed=42):
    rng = np.random.default_rng(seed)
    depot = (50.0, 50.0)
    clients = []
    for i in range(1, n_clients + 1):
        clients.append(Client(
            id=i,
            x=float(rng.uniform(0, 100)),
            y=float(rng.uniform(0, 100)),
            demand=float(rng.uniform(1, 15))
        ))
    return VRPInstance(
        name=f"instance_{n_clients}clients",
        depot=depot,
        clients=clients,
        capacity=capacity
    )