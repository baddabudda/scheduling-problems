def print_aco_stats(model_dictionary, edd_schedule, edd_score, aco_scedule, aco_score):
    print()
    print("=== Model stats ===")
    print(f"Pheromone power (alpha): {model_dictionary['pheromone_power']}")
    print(f"Pheromone decrease rate (rho): {model_dictionary['pheromone_decrease_rate']}")
    print(f"Suitability power (beta): {model_dictionary['suitability_power']}")
    print(f"Job selection rule threshold (q): {model_dictionary['job_selection_rule_threshold']}")
    print()

    print("=== Colony info ===")
    print(f"Number of iterations: {model_dictionary['iterations']}")
    print(f"Colony size (ants): {model_dictionary['colony_size']}")
    print()

    print("=== EDD schedule ===")
    print(f"Schedule: {str(edd_schedule)}")
    print(f"Score: {edd_score}")
    print()

    print("=== ACO schedule ===")
    print(f"Schedule: {str(aco_scedule)}")
    print(f"Score: {aco_score}")
    print()

def print_classic_stats(schedule, score):
    print()
    print("=== Classic schedule ===")
    print(f"Schedule: {str(schedule)}")
    print(f"Score: {score}")
    print()