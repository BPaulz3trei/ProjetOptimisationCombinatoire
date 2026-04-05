import numpy as np
import time

from src.vrp import VRPInstance, VRPSolution


class ACO:

    def __init__(self, instance, n_ants=20, n_iter=100,
                 alpha=1.0, beta=2.0, rho=0.1, seed=42):

        self.instance = instance
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.rng = np.random.default_rng(seed)

        # meilleure solution et historique
        self.best_solution = None
        self.history = []

        # initialisation des phéromones
        n = instance.n_clients + 1
        self.tau_max = 1.0
        self.tau_min = 0.01
        self.pheromones = np.full((n, n), self.tau_max)

        # heuristique η(i,j) = 1 / distance(i,j)
        with np.errstate(divide='ignore'):
            self.heuristic = np.where(
                instance.dist_matrix > 0,
                1.0 / instance.dist_matrix,
                0.0
            )

    def _build_solution(self):
        inst = self.instance
        unvisited = set(range(1, inst.n_clients + 1))
        routes = []
        current_route = [0]
        current_load = 0.0
        current_node = 0

        while unvisited:
            feasible = [j for j in unvisited
                        if current_load + inst.clients[j-1].demand <= inst.capacity]

            if not feasible:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
                current_load = 0.0
                current_node = 0
                continue

            next_node = self._choose_next(current_node, feasible)
            current_route.append(next_node)
            current_load += inst.clients[next_node - 1].demand
            current_node = next_node
            unvisited.remove(next_node)

        current_route.append(0)
        routes.append(current_route)
        return VRPSolution(routes, inst)

    def _choose_next(self, current, feasible):
        scores = np.array([
            (self.pheromones[current][j] ** self.alpha) *
            (self.heuristic[current][j] ** self.beta)
            for j in feasible
        ])

        total = scores.sum()
        if total == 0:
            return self.rng.choice(feasible)

        probs = scores / total
        return feasible[self.rng.choice(len(feasible), p=probs)]

    def _update_pheromones(self, best_iter):
        # 1. évaporation
        self.pheromones *= (1 - self.rho)

        # 2. renforcement : seule la meilleure dépose
        if self.best_solution is None:
            depositor = best_iter
        else:
            if best_iter.total_distance() < self.best_solution.total_distance():
                depositor = best_iter
            else:
                depositor = self.best_solution

        delta = 1.0 / depositor.total_distance()
        for route in depositor.routes:
            for i in range(len(route) - 1):
                self.pheromones[route[i]][route[i+1]] += delta
                self.pheromones[route[i+1]][route[i]] += delta

        # 3. bornes MMAS
        self.pheromones = np.clip(self.pheromones, self.tau_min, self.tau_max)

    def _update_tau_bounds(self):
        best_dist = self.best_solution.total_distance()
        self.tau_max = 1.0 / (self.rho * best_dist)
        self.tau_min = self.tau_max / (2 * self.instance.n_clients)

    def run(self, verbose=False):
        start_time = time.time()

        for iteration in range(self.n_iter):
            solutions = [self._build_solution() for _ in range(self.n_ants)]
            best_iter = min(solutions, key=lambda s: s.total_distance())

            if (self.best_solution is None or
                    best_iter.total_distance() < self.best_solution.total_distance()):
                self.best_solution = best_iter
                self._update_tau_bounds()

            self._update_pheromones(best_iter)
            self.history.append(self.best_solution.total_distance())

            if verbose and (iteration + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Itération {iteration+1:4d}/{self.n_iter} | "
                      f"Meilleure : {self.best_solution.total_distance():.2f} | "
                      f"Temps : {elapsed:.1f}s")

        return self.best_solution