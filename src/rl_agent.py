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
    def choose_action(self, state):
        """
        Choisit une action selon la politique epsilon-greedy.

        Avec probabilité epsilon  → action aléatoire (exploration)
        Avec probabilité 1-epsilon → meilleure action connue (exploitation)

        Paramètres
        ----------
        state : tuple (prog, qualite, stag) retourné par encode_state()

        Retourne
        --------
        action_idx : int, index de l'action choisie dans ACTIONS
        """
        if self.rng.random() < self.epsilon:
            # Exploration : action aléatoire
            action_idx = int(self.rng.integers(0, N_ACTIONS))
        else:
            # Exploitation : meilleure action connue pour cet état
            q_values = self._get_q_values(state)
            action_idx = int(np.argmax(q_values))

        # Enregistrer pour analyse
        self.action_history.append(ACTIONS[action_idx])

        return action_idx
        
    def compute_reward(self, prev_best, new_best, stagnation):
        """
        Calcule la récompense reçue après une action.

        Paramètres
        ----------
        prev_best  : meilleure distance AVANT l'action
        new_best   : meilleure distance APRÈS l'action
        stagnation : nb d'itérations sans amélioration

        Retourne
        --------
        reward : float
        """
        if new_best < prev_best:
            # L'action a permis une amélioration → bonne action
            reward = 1.0
        elif stagnation > 10:
            # ACO est bloqué depuis trop longtemps → mauvaise action
            reward = -0.5
        else:
            # Ni amélioration ni stagnation forte → neutre
            reward = 0.0

        self.reward_history.append(reward)
        return reward

    def update(self, state, action_idx, reward, next_state):
        """
        Met à jour la table Q avec la formule Q-Learning :
        Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') − Q(s,a)]

        Paramètres
        ----------
        state      : état AVANT l'action (tuple)
        action_idx : index de l'action choisie
        reward     : récompense reçue
        next_state : état APRÈS l'action (tuple)
        """
        q_values      = self._get_q_values(state)
        q_values_next = self._get_q_values(next_state)

        # Valeur Q actuelle
        q_current = q_values[action_idx]

        # Meilleure valeur Q atteignable depuis le prochain état
        q_next_max = np.max(q_values_next)

        # Formule de mise à jour Q-Learning
        q_values[action_idx] = q_current + self.alpha * (
            reward + self.gamma * q_next_max - q_current
        )

    def apply_action(self, action_idx, rho, beta):
        """
        Traduit l'action choisie en modification concrète de rho et beta.

        Paramètres
        ----------
        action_idx : int, index de l'action dans ACTIONS
        rho        : valeur actuelle du taux d'évaporation
        beta       : valeur actuelle du poids heuristique

        Retourne
        --------
        new_rho, new_beta : float, paramètres mis à jour et bornés
        """
        action = ACTIONS[action_idx]

        if action == "increase_rho":
            rho = rho + DELTA_RHO
        elif action == "decrease_rho":
            rho = rho - DELTA_RHO
        elif action == "increase_beta":
            beta = beta + DELTA_BETA
        elif action == "decrease_beta":
            beta = beta - DELTA_BETA
        # "do_nothing" → on ne touche à rien

        # Bornes : on empêche des valeurs aberrantes
        rho  = float(np.clip(rho,  RHO_MIN,  RHO_MAX))
        beta = float(np.clip(beta, BETA_MIN, BETA_MAX))

        return rho, beta