import numpy as np
from visualization.matrix_plots import plot_transition_matrix, plot_markov_graph
from visualization.simulation_plots import (plot_simulation,
                                            plot_convergence,
                                            plot_stationary_distribution)
from utils.calculations import (compute_stationary_distribution,
                              calculate_mixing_time)

class WeatherMarkovAnalysis:
    def __init__(self):
        self.states = ["Soleil", "Nuages", "Pluie"]
        self.transition_matrix = np.array([
            [0.65, 0.25, 0.10],
            [0.20, 0.50, 0.30],
            [0.15, 0.35, 0.50]
        ])

    def simulate_weather(self, n_days, initial_state=0):
        states = np.zeros(n_days, dtype=int)
        states[0] = initial_state
        
        for t in range(1, n_days):
            states[t] = np.random.choice(
                len(self.states),
                p=self.transition_matrix[states[t-1]]
            )

        return states

    def analyze_stationary_behavior(self, n_steps=100):
        initial_dist = np.zeros(len(self.states))
        initial_dist[0] = 1
        
        distributions = [initial_dist]
        current_dist = initial_dist
        
        for _ in range(n_steps):
            current_dist = np.dot(current_dist, self.transition_matrix)
            distributions.append(current_dist)
        
        distributions = np.array(distributions)
        pi = compute_stationary_distribution(self.transition_matrix)
        mixing_time = calculate_mixing_time(distributions, pi)
        
        plot_convergence(distributions, self.states, pi)
        
        print("\nDistribution stationnaire :")
        for state, prob in zip(self.states, pi):
            print(f"{state}: {prob:.3f}")
        print(f"\nTemps de mélange (ε=0.01): {mixing_time} étapes")
        
        return pi, mixing_time

    def run_complete_analysis(self):
        print("1. Matrice de Transition")
        plot_transition_matrix(self.transition_matrix, self.states)
        
        print("\n2. Graphe de Markov")
        plot_markov_graph(self.transition_matrix, self.states)
        
        print("\n3. Simulation sur 30 jours")
        states = self.simulate_weather(30)
        plot_simulation(states, self.states)
        
        print("\n4. Distribution Stationnaire")
        pi = compute_stationary_distribution(self.transition_matrix)
        plot_stationary_distribution(pi, self.states)

        print("\n5. Analyse du Comportement Stationnaire")
        self.analyze_stationary_behavior()
        
        print("\nDistribution stationnaire calculée:")
        for state, prob in zip(self.states, pi):
            print(f"{state}: {prob:.2f}")
