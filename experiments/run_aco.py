import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.vrp import generate_random_instance
from src.aco import ACO



N_RUNS = 10        # nombre de fois qu'on lance ACO par instance
N_ANTS = 20        # nombre de fourmis
N_ITER = 100       # nombre d'itérations

ACO_PARAMS = {
    'alpha': 1.0,  # importance des phéromones
    'beta':  2.0,  # importance de la distance
    'rho':   0.1,  # taux d'évaporation
}

#  (n_clients, capacité, seed)
INSTANCES_CONFIG = [
    (10, 50,  1),
    (15, 60,  2),
    (20, 70,  3),
    (25, 80,  4),
]


#  Expériences
all_results = {}

print("=" * 60)
print("  EXPÉRIENCES MMAS (BASELINE)")
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
    run_times = []
    best_convergence = None
    best_dist = float('inf')

    for run in range(N_RUNS):
        seed = run * 100 + inst_seed
        t0 = time.time()

        aco = ACO(instance=instance, n_ants=N_ANTS, n_iter=N_ITER,
                  seed=seed, **ACO_PARAMS)
        solution = aco.run(verbose=False)

        elapsed = time.time() - t0
        dist = solution.total_distance()
        run_distances.append(dist)
        run_times.append(elapsed)

        if dist < best_dist:
            best_dist = dist
            best_convergence = aco.history

        print(f"  Run {run+1:2d}/{N_RUNS} | "
              f"Distance: {dist:.2f} | "
              f"Temps: {elapsed:.2f}s")

    # calcul des statistiques sur les 10 runs
    distances = np.array(run_distances)
    times = np.array(run_times)

    stats = {
        'mean':   float(np.mean(distances)),
        'std':    float(np.std(distances)),
        'min':    float(np.min(distances)),
        'max':    float(np.max(distances)),
        'median': float(np.median(distances)),
    }

    print(f"\n  → Résumé {name} :")
    print(f"     Distance  : {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"     Min/Max   : {stats['min']:.2f} / {stats['max']:.2f}")
    print(f"     Temps moy : {np.mean(times):.2f}s")

    all_results[name] = {
        'n_clients':       n_clients,
        'capacity':        capacity,
        'distances':       run_distances,
        'times':           run_times,
        'stats':           stats,
        'best_convergence': best_convergence,
        'params':          ACO_PARAMS,
    }

# Sauvegarde des résultats 
os.makedirs("results", exist_ok=True)

# sauvegarder les résultats en JSON
with open("results/aco_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 60)
print("  RÉSULTATS SAUVEGARDÉS")
print("=" * 60)

# afficher le tableau récapitulatif final
print(f"\n{'Instance':<25} {'Moy ± Std':>15} {'Min':>10} {'Max':>10} {'Temps':>8}")
print("-" * 70)
for name, r in all_results.items():
    s = r['stats']
    t = np.mean(r['times'])
    print(f"{name:<25} "
          f"{s['mean']:>8.2f} ± {s['std']:<5.2f} "
          f"{s['min']:>10.2f} "
          f"{s['max']:>10.2f} "
          f"{t:>7.2f}s")

print("\n✓ Fichier sauvegardé : results/aco_results.json")