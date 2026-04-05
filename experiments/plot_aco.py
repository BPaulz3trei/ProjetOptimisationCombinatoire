import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# charger les résultats
with open("results/aco_results.json", "r") as f:
    results = json.load(f)

print("✓ Résultats chargés !")
print(f"  Instances disponibles : {list(results.keys())}")

colors = ['steelblue', 'darkorange', 'green', 'red']

# ── Graphe 1 : Courbes de convergence ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

for i, (name, data) in enumerate(results.items()):
    convergence = data['best_convergence']
    iterations = list(range(1, len(convergence) + 1))
    ax.plot(iterations, convergence,
            color=colors[i],
            linewidth=2,
            label=f"{name} (min={data['stats']['min']:.1f})")

ax.set_xlabel("Itération", fontsize=12)
ax.set_ylabel("Meilleure distance trouvée (km)", fontsize=12)
ax.set_title("Convergence de MMAS sur 4 instances", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/convergence_mmas.png", dpi=150)
plt.close()

print("✓ Graphe sauvegardé : results/convergence_mmas.png")

# ── Graphe 2 : Boxplot des distances ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

data_boxplot = [data['distances'] for data in results.values()]
labels = [f"{data['n_clients']} clients" for data in results.values()]

bp = ax.boxplot(data_boxplot,
                tick_labels=labels,
                patch_artist=True,
                notch=False)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for i, (name, data) in enumerate(results.items()):
    mean = data['stats']['mean']
    ax.text(i + 1, data['stats']['max'] * 1.01,
            f"μ={mean:.1f}",
            ha='center', fontsize=9)

ax.set_xlabel("Instance", fontsize=12)
ax.set_ylabel("Distance totale (km)", fontsize=12)
ax.set_title("Distribution des distances MMAS sur 10 runs", fontsize=14)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("results/boxplot_mmas.png", dpi=150)
plt.close()

print("✓ Graphe sauvegardé : results/boxplot_mmas.png")
print("\n Tous les graphes générés dans results/")