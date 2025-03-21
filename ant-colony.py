# Task: Ant Colony Optimization for 1 || SUM(T_j)

import random
import numpy as np

class Ant:
    schedule = [-1, -1, -1, -1]
    score = -1
    visited = set()

    def calculate_probabilities(self) -> list:
        # TODO: add entire code here
        # if job has been visited, prop = 0
        # otherwise calculate via formula
        prob_vector = [-1, -1, -1, -1]

        for i in range(0, SCHEDULE_LENGTH):
            if i not in self.visited:
                # TODO: calculate it via formula
                prob_vector[i] = 100
            else:
                # make it 0
                prob_vector[i] = 0
        
        return prob_vector

    def pick_job(self, prob_vector):
        q = random.random()
        
        if (q < JOB_SELECTION_RULE_TRESHOLD):
            # TODO: argmax
            return
        else:
            # TODO: random weighted sample
            return

    def update_pheromones_local(self):
        # TODO: update local pheromones
        return

    def make_schedule(self):
        for position in range(0, len(self.schedule)):
            prob_vector = self.calculate_probabilities()
            self.pick_job(prob_vector)
            self.update_pheromones_local()

def init_jobs() -> np.array:
    types = [('job_number', int), ('processing_time', int), ('deadline', int)]
    values = [(1, 3, 5), (2, 1, 2), (3, 1, 3), (4, 2, 1)]

    return np.array(values, dtype=types)

def init_suitability_matrix(size) -> np.array:
    result_matrix = np.zeros((size, size))

    for i in range(0, SCHEDULE_LENGTH):
        for j in range(0, SCHEDULE_LENGTH):
            result_matrix[i][j] = 1 / JOBS[j][2]

    return result_matrix

def calculate_edd_delay() -> float:
    sorted_by_edd = np.sort(JOBS, order='deadline')
    finish_times = np.zeros(SCHEDULE_LENGTH) # C_j
    finish_times[0] = sorted_by_edd[0][1]

    for i in range(1, len(finish_times)):
        finish_times[i] = finish_times[i-1] + sorted_by_edd[i][1]

    edd_delay = 0
    
    for i in range(0, len(finish_times)):
        delay = finish_times[i] - sorted_by_edd[i][2]
        if (delay > 0):
            edd_delay += delay
    
    return edd_delay

def init_pheromone_matrix(size) -> np.array:
    delay = calculate_edd_delay()
    fill_value = 1 / (ANTS_NUMBER * delay)

    return np.full((size, size), fill_value)

# ===== Initialization step =====

# 1. Define model hyperparameters and other constans
PHEROMONE_POWER = 1 # alpha
PHEROMONE_DECREASE_RATE = 0.6 # rho in [0, 1]
SUITABILITY_POWER = 1 # beta
JOB_SELECTION_RULE_TRESHOLD = 0.5 # q in [0, 1]

ITERATIONS_NUMBER = 20
ANTS_NUMBER = 20

JOBS = init_jobs()
SCHEDULE_LENGTH = len(JOBS)

SUITABILITY_MATRIX = init_suitability_matrix(SCHEDULE_LENGTH)

# 2. Initialize pheromone matrix
pheromone_matrix = init_pheromone_matrix(SCHEDULE_LENGTH)


def run_ACO():
    # TODO: init ant colony
    print(SUITABILITY_MATRIX)

print(calculate_edd_delay())