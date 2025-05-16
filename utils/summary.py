def print_aco_stats(model_dictionary, heuristic_schedule, heuristic_score, aco_schedule, aco_score):
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

    print("=== Heuristic schedule ===")
    print(f"Schedule: {str(heuristic_schedule)}")
    print(f"Score: {heuristic_score}")
    print()

    print("=== ACO schedule ===")
    print(f"Schedule: {str(aco_schedule)}")
    print(f"Score: {aco_score}")
    print()

def print_classic_stats(schedule, score):
    print()
    print("=== Classic schedule ===")
    print(f"Schedule: {str(schedule)}")
    print(f"Score: {score}")
    print()

def print_parallel_flow_answer(answer):
    print()
    print("=== Parallel Flow ===")

    if answer == True:
        print("There exists a schedule respecting all time windows.")
    else:
        print("There is no schedule respecting all time windows.")
    
    print()
    