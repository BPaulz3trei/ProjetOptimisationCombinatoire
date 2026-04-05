import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.vrp import generate_random_instance, VRPSolution, Client, VRPInstance
from src.aco import ACO

def test_instance_creation():
    inst = generate_random_instance(n_clients=10, capacity=50, seed=42)
    
    assert inst.n_clients == 10, "Nombre de clients incorrect"
    assert inst.capacity == 50, "Capacité incorrecte"
    assert inst.depot == (50.0, 50.0), "Dépôt incorrect"
    assert inst.dist_matrix.shape == (11, 11), "Matrice de distances incorrecte"
    
    print(" test_instance_creation : OK")


def test_distance_matrix():
    inst = generate_random_instance(n_clients=5, capacity=50, seed=1)
    
    # la distance d'un noeud vers lui-même doit être 0
    for i in range(inst.n_clients + 1):
        assert inst.dist_matrix[i][i] == 0, \
            f"Distance de {i} vers lui-même doit être 0"
    
    # la matrice doit être symétrique : d(i,j) = d(j,i)
    for i in range(inst.n_clients + 1):
        for j in range(inst.n_clients + 1):
            assert abs(inst.dist_matrix[i][j] - inst.dist_matrix[j][i]) < 1e-9, \
                f"Matrice non symétrique en ({i},{j})"
    
    # les distances doivent être positives
    for i in range(inst.n_clients + 1):
        for j in range(inst.n_clients + 1):
            assert inst.dist_matrix[i][j] >= 0, \
                f"Distance négative en ({i},{j})"
    
    print(" test_distance_matrix : OK")


def test_solution_feasibility():
    inst = generate_random_instance(n_clients=5, capacity=100, seed=2)

    # solution valide : un véhicule par client
    routes_valides = [[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0], [0, 5, 0]]
    sol_valide = VRPSolution(routes_valides, inst)
    assert sol_valide.is_feasible() == True, \
        "Solution valide détectée comme invalide"

    # solution invalide : client 1 visité deux fois
    routes_invalides = [[0, 1, 1, 0], [0, 2, 3, 0], [0, 4, 5, 0]]
    sol_invalide = VRPSolution(routes_invalides, inst)
    assert sol_invalide.is_feasible() == False, \
        "Solution invalide détectée comme valide"

    print(" test_solution_feasibility : OK")


def test_total_distance():
    inst = generate_random_instance(n_clients=3, capacity=100, seed=3)

    routes = [[0, 1, 2, 3, 0]]
    sol = VRPSolution(routes, inst)

    # calculer manuellement la distance
    distance_manuelle = (
        inst.dist_matrix[0][1] +  # dépôt → client 1
        inst.dist_matrix[1][2] +  # client 1 → client 2
        inst.dist_matrix[2][3] +  # client 2 → client 3
        inst.dist_matrix[3][0]    # client 3 → dépôt
    )

    assert abs(sol.total_distance() - distance_manuelle) < 1e-9, \
        "Distance totale incorrecte"

    # la distance doit être positive
    assert sol.total_distance() > 0, \
        "Distance totale doit être positive"

    print(" test_total_distance : OK")


def test_aco_solution_valid():
    inst = generate_random_instance(n_clients=10, capacity=50, seed=42)

    aco = ACO(instance=inst, n_ants=10, n_iter=20, seed=0)
    solution = aco.run(verbose=False)

    # la solution doit exister
    assert solution is not None, \
        "ACO n'a pas retourné de solution"

    # la solution doit être faisable
    assert solution.is_feasible() == True, \
        "La solution ACO n'est pas faisable"

    # la distance doit être positive
    assert solution.total_distance() > 0, \
        "La distance ACO doit être positive"

    # l'historique doit avoir autant d'entrées que d'itérations
    assert len(aco.history) == 20, \
        "Historique ACO incorrect"

    print(" test_aco_solution_valid : OK")

def test_aco_convergence():
    inst = generate_random_instance(n_clients=10, capacity=50, seed=42)

    aco = ACO(instance=inst, n_ants=10, n_iter=50, seed=0)
    aco.run(verbose=False)

    # la distance ne doit jamais remonter
    for i in range(1, len(aco.history)):
        assert aco.history[i] <= aco.history[i-1] + 1e-9, \
            f"La distance a augmenté à l'itération {i} : \
            {aco.history[i-1]:.2f} → {aco.history[i]:.2f}"

    # la dernière distance doit être inférieure à la première
    assert aco.history[-1] <= aco.history[0], \
        "ACO n'a pas amélioré la solution initiale"

    print("test_aco_convergence : OK")

if __name__ == "__main__":
    print("\n=== Tests VRP ===")
    test_instance_creation()
    test_distance_matrix()
    test_solution_feasibility()
    test_total_distance()

    print("\n=== Tests ACO ===")
    test_aco_solution_valid()
    test_aco_convergence()

    print("\n Tous les tests passent !")