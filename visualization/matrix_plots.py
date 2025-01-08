import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def plot_transition_matrix(transition_matrix, states):
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, 
                annot=True, 
                fmt='.2f',
                xticklabels=states,
                yticklabels=states,
                cmap='Blues')
    plt.title('Matrice de Transition Météorologique')
    plt.xlabel('État suivant')
    plt.ylabel('État actuel')
    plt.tight_layout()
    plt.show()

def plot_markov_graph(transition_matrix, states):
    G = nx.DiGraph()
    
    for state in states:
        G.add_node(state)
        
    for i, state1 in enumerate(states):
        for j, state2 in enumerate(states):
            if transition_matrix[i,j] > 0:
                G.add_edge(state1, state2, 
                          weight=transition_matrix[i,j],
                          label=f'{transition_matrix[i,j]:.2f}')
    
    plt.figure(figsize=(12, 8))
    pos = nx.circular_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000)
    
    for edge in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, 
                             edgelist=[(edge[0], edge[1])],
                             width=edge[2]['weight']*3,
                             alpha=0.6,
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20)
    
    nx.draw_networkx_labels(G, pos)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    plt.title("Graphe de Markov des Transitions Météorologiques")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
