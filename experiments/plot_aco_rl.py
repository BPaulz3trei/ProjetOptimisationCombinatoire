import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (load_results, plot_convergence_comparison,
                        plot_boxplot_comparison, plot_rl_parameters,
                        print_summary_table)

aco_results, aco_rl_results = load_results()

plot_convergence_comparison(aco_results, aco_rl_results)
plot_boxplot_comparison(aco_results, aco_rl_results)
plot_rl_parameters(aco_rl_results)
print_summary_table(aco_results, aco_rl_results)

print("\n✓ Tous les graphes générés dans results/")