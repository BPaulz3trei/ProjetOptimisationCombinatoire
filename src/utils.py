import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def load_results(aco_path="results/aco_results.json",
                 aco_rl_path="results/aco_rl_results.json"):
    """Charge les deux fichiers de résultats."""
    with open(aco_path, "r") as f:
        aco_results = json.load(f)
    with open(aco_rl_path, "r") as f:
        aco_rl_results = json.load(f)
    return aco_results, aco_rl_results


def plot_convergence_comparison(aco_results, aco_rl_results, save_path="results/convergence_comparison.png"):
    """
    Courbes de convergence ACO vs ACO+RL sur les 4 instances.
    Une sous-figure par instance.
    """
    instances = list(aco_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, name in enumerate(instances):
        ax = axes[i]
        n_clients = aco_results[name]['n_clients']

        aco_conv    = aco_results[name]['best_convergence']
        aco_rl_conv = aco_rl_results[name]['best_convergence']

        iterations_aco    = list(range(1, len(aco_conv) + 1))
        iterations_aco_rl = list(range(1, len(aco_rl_conv) + 1))

        ax.plot(iterations_aco,    aco_conv,
                color='steelblue', linewidth=2, label='ACO seul')
        ax.plot(iterations_aco_rl, aco_rl_conv,
                color='darkorange', linewidth=2, label='ACO + RL', linestyle='--')

        gain = aco_rl_results[name]['gain_vs_baseline']
        ax.set_title(f"{n_clients} clients  (gain={gain:+.2f}%)", fontsize=12)
        ax.set_xlabel("Itération", fontsize=10)
        ax.set_ylabel("Meilleure distance", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Convergence ACO vs ACO+RL", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Sauvegardé : {save_path}")


def plot_boxplot_comparison(aco_results, aco_rl_results, save_path="results/boxplot_comparison.png"):
    """
    Boxplots côte à côte ACO vs ACO+RL pour chaque instance.
    """
    instances = list(aco_results.keys())
    n = len(instances)

    fig, ax = plt.subplots(figsize=(13, 6))

    x      = np.arange(n)
    width  = 0.35
    labels = [f"{aco_results[name]['n_clients']} clients" for name in instances]

    for i, name in enumerate(instances):
        d_aco    = aco_results[name]['distances']
        d_aco_rl = aco_rl_results[name]['distances']

        bp1 = ax.boxplot(d_aco,
                         positions=[x[i] - width/2],
                         widths=width * 0.9,
                         patch_artist=True,
                         manage_ticks=False)
        bp2 = ax.boxplot(d_aco_rl,
                         positions=[x[i] + width/2],
                         widths=width * 0.9,
                         patch_artist=True,
                         manage_ticks=False)

        bp1['boxes'][0].set_facecolor('steelblue')
        bp1['boxes'][0].set_alpha(0.7)
        bp2['boxes'][0].set_facecolor('darkorange')
        bp2['boxes'][0].set_alpha(0.7)

    # Légende manuelle
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue',  alpha=0.7, label='ACO seul'),
        Patch(facecolor='darkorange', alpha=0.7, label='ACO + RL'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Instance", fontsize=12)
    ax.set_ylabel("Distance totale", fontsize=12)
    ax.set_title("Distribution des distances sur 10 runs — ACO vs ACO+RL",
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Sauvegardé : {save_path}")


def plot_rl_parameters(aco_rl_results, save_path="results/rl_parameters.png"):
    """
    Évolution de rho et beta au fil des itérations (meilleur run).
    Montre que le RL pilote bien les paramètres.
    """
    instances = list(aco_rl_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, name in enumerate(instances):
        ax = axes[i]
        n_clients = aco_rl_results[name]['n_clients']

        rho_hist  = aco_rl_results[name]['best_rho_history']
        beta_hist = aco_rl_results[name]['best_beta_history']
        iterations = list(range(1, len(rho_hist) + 1))

        ax2 = ax.twinx()  # second axe Y pour beta

        ax.plot(iterations,  rho_hist,
                color='steelblue', linewidth=2, label='rho (ρ)')
        ax2.plot(iterations, beta_hist,
                 color='darkorange', linewidth=2, label='beta (β)', linestyle='--')

        ax.set_title(f"{n_clients} clients", fontsize=12)
        ax.set_xlabel("Itération", fontsize=10)
        ax.set_ylabel("rho (ρ)", fontsize=10, color='steelblue')
        ax2.set_ylabel("beta (β)", fontsize=10, color='darkorange')
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.suptitle("Évolution des paramètres pilotés par le RL", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Sauvegardé : {save_path}")


def print_summary_table(aco_results, aco_rl_results):
    """Affiche le tableau comparatif final dans le terminal."""
    print("\n" + "=" * 70)
    print("  TABLEAU COMPARATIF FINAL")
    print("=" * 70)
    print(f"{'Instance':<25} {'ACO seul':>12} {'ACO+RL':>15} {'Gain':>8} {'Std RL':>8}")
    print("-" * 70)

    for name in aco_results.keys():
        s_aco    = aco_results[name]['stats']
        s_aco_rl = aco_rl_results[name]['stats']
        gain     = aco_rl_results[name]['gain_vs_baseline']

        print(f"{name:<25} "
              f"{s_aco['mean']:>12.2f} "
              f"{s_aco_rl['mean']:>12.2f} "
              f"{gain:>+8.2f}% "
              f"{s_aco_rl['std']:>8.2f}")

    print("=" * 70)