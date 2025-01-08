if __name__ == "__main__":
    from data.processor import process_weather_data
    from core.markov import WeatherMarkovAnalysis


    file_path = 'data/DailyDelhiClimateTrain.csv'
    states, transition_matrix = process_weather_data(file_path)

    analysis = WeatherMarkovAnalysis()
    analysis.states = states
    analysis.transition_matrix = transition_matrix
    analysis.run_complete_analysis()
