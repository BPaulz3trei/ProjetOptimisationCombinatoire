import numpy as np

# ──────────────────────────────────────────────
#  Définition des actions disponibles
# ──────────────────────────────────────────────
ACTIONS = [
    "increase_rho",   # plus d'évaporation → plus d'exploration
    "decrease_rho",   # moins d'évaporation → plus d'exploitation
    "increase_beta",  # favoriser les clients proches
    "decrease_beta",  # moins d'importance à la distance
    "do_nothing",     # statu quo
]
N_ACTIONS = len(ACTIONS)

# Bornes pour éviter des valeurs aberrantes
RHO_MIN,  RHO_MAX  = 0.01, 0.5
BETA_MIN, BETA_MAX = 1.0,  5.0
DELTA_RHO  = 0.05
DELTA_BETA = 0.5


class QLearningAgent:
    """
    Agent Q-Learning qui pilote les paramètres d'ACO (rho, beta)
    en observant l'état de la recherche à chaque itération.
    """

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, seed=42):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.rng     = np.random.default_rng(seed)

        # Table Q : dictionnaire état → tableau de N_ACTIONS valeurs
        self.q_table = {}

        # Historique pour analyse
        self.action_history = []
        self.reward_history = []

    def _get_q_values(self, state):
        """Retourne les valeurs Q pour un état, initialisées à zéro si inconnu."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(N_ACTIONS)
        return self.q_table[state]

    def encode_state(self, iteration, n_iter, best_dist, current_dist, stagnation):
        """
        Transforme les infos d'ACO en un état discret (tuple de 3 entiers).
        Retourne : (progression, qualite, stagnation_level)
        """
        # Dimension 1 — Progression
        ratio = iteration / max(n_iter - 1, 1)
        if ratio < 0.33:
            prog = 0
        elif ratio < 0.66:
            prog = 1
        else:
            prog = 2

        # Dimension 2 — Qualité relative
        if best_dist > 0:
            gap = (current_dist - best_dist) / best_dist
        else:
            gap = 0.0

        if gap < 0.02:
            qualite = 0
        elif gap < 0.10:
            qualite = 1
        else:
            qualite = 2

        # Dimension 3 — Stagnation
        if stagnation < 5:
            stag = 0
        elif stagnation < 15:
            stag = 1
        else:
            stag = 2

        return (prog, qualite, stag)