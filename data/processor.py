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
