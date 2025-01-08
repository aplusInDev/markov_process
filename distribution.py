import pandas as pd
import numpy as np

def define_weather_state(row):
    """Define weather state based on temperature and humidity"""
    temp = row['meantemp']
    humidity = row['humidity']
    
    if temp >= 20 and humidity < 60:
        return "Soleil"
    elif temp >= 15 and humidity >= 60:
        return "Nuages"
    else:
        return "Pluie"

def process_weather_data(file_path):
    df = pd.read_csv(file_path)
    df['weather_state'] = df.apply(define_weather_state, axis=1)
    
    states = sorted(df['weather_state'].unique())
    n_states = len(states)
    
    transition_matrix = np.zeros((n_states, n_states))
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    
    for i in range(len(df)-1):
        current_state = df['weather_state'].iloc[i]
        next_state = df['weather_state'].iloc[i+1]
        current_idx = state_to_idx[current_state]
        next_idx = state_to_idx[next_state]
        transition_matrix[current_idx][next_idx] += 1
    
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = np.divide(transition_matrix, row_sums[:, np.newaxis],
                                where=row_sums[:, np.newaxis] != 0)
    
    return states, transition_matrix

def calculate_stationary_distribution(P):
    """Calculate the stationary distribution for a given transition matrix P"""
    n = P.shape[0]
    A = np.append(P.T - np.eye(n), [np.ones(n)], axis=0)
    b = np.append(np.zeros(n), 1)
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi

if __name__ == "__main__":
    from core.markov import WeatherMarkovAnalysis
    
    # Process general weather data
    states, transition_matrix = process_weather_data('data/DailyDelhiClimateTest.csv')
    
    # Calculate stationary distribution for general weather states
    pi = calculate_stationary_distribution(transition_matrix)
    
    # Define transition matrices for summer (été) and winter (hiver)
    P_ete = np.array([[0.70, 0.20, 0.10],
                      [0.25, 0.50, 0.25],
                      [0.20, 0.30, 0.50]])
    
    P_hiver = np.array([[0.60, 0.30, 0.10],
                        [0.20, 0.55, 0.25],
                        [0.15, 0.35, 0.50]])
    
    # Calculate stationary distributions for summer and winter
    pi_ete = calculate_stationary_distribution(P_ete)
    pi_hiver = calculate_stationary_distribution(P_hiver)
    
    # Print results
    print("Stationary distribution (π):", pi)
    print("Stationary distribution for summer (π_été):", pi_ete)
    print("Stationary distribution for winter (π_hiver):", pi_hiver)
    
    # Run complete analysis
    analysis = WeatherMarkovAnalysis()
    analysis.states = states
    analysis.transition_matrix = transition_matrix
    analysis.run_complete_analysis()
