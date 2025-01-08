import matplotlib.pyplot as plt

def plot_simulation(states, state_labels, n_days=30):
    plt.figure(figsize=(15, 6))
    plt.plot(states, marker='o', linestyle='-', markersize=10)
    plt.yticks(range(len(state_labels)), state_labels)
    plt.xlabel('Jours')
    plt.ylabel('État météorologique')
    plt.title(f'Simulation de la météo sur {n_days} jours')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_convergence(distributions, states, pi):
    plt.figure(figsize=(15, 10))
    for i in range(len(states)):
        plt.plot(distributions[:, i], label=states[i])
    
    plt.axhline(y=pi[0], color='r', linestyle='--', alpha=0.3)
    plt.title('Convergence vers la Distribution Stationnaire')
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('Probabilité')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_stationary_distribution(pi, states):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(states, pi)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title('Distribution Stationnaire des États Météorologiques')
    plt.ylabel('Probabilité')
    plt.ylim(0, 1)
    plt.show()
