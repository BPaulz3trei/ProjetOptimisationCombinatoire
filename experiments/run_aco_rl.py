import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from src.vrp import generate_random_instance
from src.aco_rl import ACO_RL

# ── Paramètres identiques à run_aco.py de ta camarade ──────────────────────
N_RUNS = 10
N_ANTS = 20
N_ITER = 200

ACO_PARAMS = {
    'alpha': 1.0,
    'beta':  2.0,
    'rho':   0.1,
}

RL_PARAMS = {
    'rl_alpha':   0.1,
    'rl_gamma':   0.9,
    'rl_epsilon': 0.2,
}

# Exactement les mêmes instances que ta camarade
INSTANCES_CONFIG = [
    (10, 50, 1),
    (15, 60, 2),
    (20, 70, 3),
    (25, 80, 4),
]

# ── Chargement des résultats baseline ──────────────────────────────────────
with open("results/aco_results.json", "r") as f:
    baseline_results = json.load(f)
print("✓ Baseline chargée !")

# ── Expériences ACO+RL ──────────────────────────────────────────────────────
all_results = {}

print("=" * 60)
print("  EXPÉRIENCES ACO + RL")
print("=" * 60)

for (n_clients, capacity, inst_seed) in INSTANCES_CONFIG:
    instance = generate_random_instance(
        n_clients=n_clients,
        capacity=capacity,
        seed=inst_seed
    )
    name = instance.name
    print(f"\nInstance : {name} ({n_clients} clients, capacité={capacity})")

    run_distances = []
    run_times     = []
    best_convergence = None
    best_rho_history  = None
    best_beta_history = None
    best_dist = float('inf')

    for run in range(N_RUNS):
        seed = run * 100 + inst_seed  # mêmes seeds que ta camarade
        t0 = time.time()

        aco_rl = ACO_RL(
            instance=instance,
            n_ants=N_ANTS,
            n_iter=N_ITER,
            seed=seed,
            **ACO_PARAMS,
            **RL_PARAMS
        )
        solution = aco_rl.run(verbose=False)
        elapsed  = time.time() - t0
        dist     = solution.total_distance()

        run_distances.append(dist)
        run_times.append(elapsed)

        if dist < best_dist:
            best_dist         = dist
            best_convergence  = aco_rl.history
            best_rho_history  = aco_rl.rho_history
            best_beta_history = aco_rl.beta_history

        print(f"  Run {run+1:2d}/{N_RUNS} | "
              f"Distance: {dist:.2f} | "
              f"Temps: {elapsed:.2f}s")

    # ── Statistiques ────────────────────────────────────────────────────────
    distances = np.array(run_distances)
    times     = np.array(run_times)
    stats = {
        'mean':   float(np.mean(distances)),
        'std':    float(np.std(distances)),
        'min':    float(np.min(distances)),
        'max':    float(np.max(distances)),
        'median': float(np.median(distances)),
    }

    # ── Calcul du gain vs baseline ───────────────────────────────────────────
    baseline_mean = baseline_results[name]['stats']['mean']
    gain = (baseline_mean - stats['mean']) / baseline_mean * 100

    print(f"\n  → Résumé {name} :")
    print(f"     ACO seul  : {baseline_mean:.2f}")
    print(f"     ACO+RL    : {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"     Gain      : {gain:+.2f}%")

    all_results[name] = {
        'n_clients':        n_clients,
        'capacity':         capacity,
        'distances':        run_distances,
        'times':            run_times,
        'stats':            stats,
        'gain_vs_baseline': gain,
        'best_convergence': best_convergence,
        'best_rho_history': best_rho_history,
        'best_beta_history': best_beta_history,
        'params':           {**ACO_PARAMS, **RL_PARAMS},
    }

# ── Sauvegarde ───────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
with open("results/aco_rl_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# ── Tableau récapitulatif final ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TABLEAU COMPARATIF FINAL")
print("=" * 60)
print(f"\n{'Instance':<25} {'ACO seul':>12} {'ACO+RL':>15} {'Gain':>8}")
print("-" * 65)

for name, r in all_results.items():
    baseline_mean = baseline_results[name]['stats']['mean']
    print(f"{name:<25} "
          f"{baseline_mean:>12.2f} "
          f"{r['stats']['mean']:>8.2f} ± {r['stats']['std']:<5.2f} "
          f"{r['gain_vs_baseline']:>+7.2f}%")

print("\n✓ Résultats sauvegardés : results/aco_rl_results.json")