import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def analyze_stationary_behavior(transition_matrix, states, n_steps=100):
    """Analyse complète du comportement stationnaire"""
    
    # 1. Distribution stationnaire théorique
    eigenvals, eigenvecs = linalg.eig(transition_matrix.T)
    stationary = eigenvecs[:, np.argmax(eigenvals.real)].real
    pi = stationary / stationary.sum()
    
    # 2. Convergence vers la distribution stationnaire
    initial_dist = np.zeros(len(states))
    initial_dist[0] = 1  # Commencer dans le premier état
    
    distributions = [initial_dist]
    current_dist = initial_dist
    
    for _ in range(n_steps):
        current_dist = np.dot(current_dist, transition_matrix)
        distributions.append(current_dist)
    
    distributions = np.array(distributions)
    
    # Visualisation
    plt.figure(figsize=(15, 10))
    
    # Plot de la convergence
    for i in range(len(states)):
        plt.plot(distributions[:, i], label=states[i])
    
    plt.axhline(y=pi[0], color='r', linestyle='--', alpha=0.3)
    plt.title('Convergence vers la Distribution Stationnaire')
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('Probabilité')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Afficher les résultats
    print("\nDistribution stationnaire :")
    for state, prob in zip(states, pi):
        print(f"{state}: {prob:.3f}")
    
    # Temps de mélange (mixing time)
    distances = np.max(np.abs(distributions - pi), axis=1)
    mixing_time = np.where(distances < 0.01)[0][0]
    print(f"\nTemps de mélange (ε=0.01): {mixing_time} étapes")

    return pi, mixing_time
