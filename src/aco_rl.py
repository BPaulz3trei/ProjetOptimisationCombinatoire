import numpy as np
from src.aco import ACO
from src.rl_agent import QLearningAgent


class ACO_RL(ACO):
    """
    Version hybride ACO + Q-Learning.
    Hérite de ACO et surcharge uniquement la méthode run()
    pour y brancher l'agent RL qui ajuste rho et beta dynamiquement.
    """

    def __init__(self, instance, n_ants=20, n_iter=100,
                 alpha=1.0, beta=2.0, rho=0.1,
                 rl_alpha=0.1, rl_gamma=0.9, rl_epsilon=0.2,
                 seed=42):
        super().__init__(instance, n_ants, n_iter,
                         alpha, beta, rho, seed)

        self.agent = QLearningAgent(
            alpha=rl_alpha,
            gamma=rl_gamma,
            epsilon=rl_epsilon,
            seed=seed
        )

        self.rho_history  = []
        self.beta_history = []

    def run(self, verbose=False):
        """
        Boucle principale ACO+RL.
        À chaque itération :
          1. ACO construit ses solutions
          2. L'agent observe l'état
          3. L'agent choisit une action
          4. L'action modifie rho et beta
          5. L'agent reçoit sa récompense et met à jour Q
        """
        stagnation = 0

        for iteration in range(self.n_iter):

            # ── 1. ACO : construction des solutions ──────────────────
            solutions = [self._build_solution() for _ in range(self.n_ants)]
            best_iter = min(solutions, key=lambda s: s.total_distance())

            prev_best = (self.best_solution.total_distance()
                         if self.best_solution else float('inf'))

            if (self.best_solution is None or
                    best_iter.total_distance() < self.best_solution.total_distance()):
                self.best_solution = best_iter
                self._update_tau_bounds()
                stagnation = 0
            else:
                stagnation += 1

            self._update_pheromones(best_iter)
            self.history.append(self.best_solution.total_distance())

            # ── 2. Agent RL : observer l'état courant ────────────────
            current_dist = best_iter.total_distance()
            best_dist    = self.best_solution.total_distance()

            state = self.agent.encode_state(
                iteration, self.n_iter,
                best_dist, current_dist, stagnation
            )

            # ── 3. Agent RL : choisir une action ─────────────────────
            action_idx = self.agent.choose_action(state)

            # ── 4. Appliquer l'action sur rho et beta ─────────────────
            self.rho, self.beta = self.agent.apply_action(
                action_idx, self.rho, self.beta
            )
            self.rho_history.append(self.rho)
            self.beta_history.append(self.beta)

            # ── 5. Agent RL : récompense et mise à jour Q ─────────────
            new_best = self.best_solution.total_distance()
            reward   = self.agent.compute_reward(prev_best, new_best, stagnation)

            next_state = self.agent.encode_state(
                iteration + 1, self.n_iter,
                new_best, current_dist, stagnation
            )

            self.agent.update(state, action_idx, reward, next_state)

            # Réduire l'exploration au fil du temps
            self.agent.epsilon = max(0.05, self.agent.epsilon * 0.995)

            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Itération {iteration+1:4d}/{self.n_iter} | "
                      f"Meilleure : {self.best_solution.total_distance():.2f} | "
                      f"rho={self.rho:.3f} beta={self.beta:.2f} | "
                      f"stagnation={stagnation}")

        return self.best_solution