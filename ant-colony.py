# Task: Ant Colony Optimization for 1 || SUM(T_j)

import random
import numpy as np

# ===== Definition of Ant class =====
# Each ant stores:
# - schedule size
# - schedule, an array of JOBS' indices
# - available_jobs, a set of unpicked jobs
# - delay_score 
class Ant:
    def __init__(self, size):
        self.schedule_size = size
        self.schedule = np.full(self.schedule_size, -1)
        self.available_jobs = {job for job in range(self.schedule_size)}
        self.delay_score = -1
    
    def calculate_job_strength(self, schedule_pos, job_number):
        return pheromone_matrix[schedule_pos][job_number] ** PHEROMONE_POWER * \
                SUITABILITY_MATRIX[schedule_pos][job_number] ** SUITABILITY_POWER

    def calculate_overall_job_strength(self, schedule_pos) -> float:
        sum = 0
        for job in self.available_jobs:
            sum += self.calculate_job_strength(schedule_pos, job)
        
        return sum

    def pick_by_argmax(self, schedule_pos) -> int:
        max_strength = -1
        max_job = -1
        for job in self.available_jobs:
            candidate_max = self.calculate_job_strength(schedule_pos, job)
            if (candidate_max > max_strength):
                max_strength = candidate_max
                max_job = job
        
        return max_job

    def pick_job(self, prob_vector, schedule_pos) -> int:
        q = random.random()
        
        if (q < JOB_SELECTION_RULE_TRESHOLD):
            return self.pick_by_argmax(schedule_pos)
        else:
            picked_job = np.random.choice(JOBS, p=prob_vector)
            return picked_job[0] - 1

    def calculate_probabilities(self, schedule_pos) -> np.array:
        prob_vector = np.full(self.schedule_size, 0.0)
        overall_job_strength = self.calculate_overall_job_strength(schedule_pos)

        for job in self.available_jobs:
            prob_vector[job] = self.calculate_job_strength(schedule_pos, job) / \
                                overall_job_strength
        
        return prob_vector

    def make_schedule(self):
        for schedule_pos in range(self.schedule_size):
            prob_vector = self.calculate_probabilities(schedule_pos)
            picked_job = self.pick_job(prob_vector, schedule_pos)

            self.schedule[schedule_pos] = picked_job
            self.available_jobs.remove(picked_job)

        mapped_schedule = self.map_to_schedule()
        self.delay_score = calculate_edd_delay(mapped_schedule)
    
    def reset_state(self):
        self.schedule = np.full(self.schedule_size, -1)
        self.delay_score = -1
        self.available_jobs = {job for job in range(self.schedule_size)}
    
    def map_to_schedule(self) -> np.array:
        types = [('job_number', int), ('processing_time', int), ('deadline', int)]
        list = []

        for schedule_pos in range(self.schedule_size):
            job_id = self.schedule[schedule_pos]
            list.append(JOBS[job_id])
        
        return np.array(list, dtype=types)

# ===== Definition of global methods =====

def init_jobs() -> np.array:
    types = [('job_number', int), ('processing_time', int), ('deadline', int)]
    values = [(1, 3, 5), (2, 1, 2), (3, 1, 3), (4, 2, 1)]

    return np.array(values, dtype=types)

def init_suitability_matrix(size) -> np.array:
    result_matrix = np.zeros((size, size))

    for i in range(SCHEDULE_LENGTH):
        for j in range(SCHEDULE_LENGTH):
            result_matrix[i][j] = 1 / JOBS_EDD[j][2]

    return result_matrix

def calculate_edd_delay(schedule) -> int:
    finish_times = np.zeros(SCHEDULE_LENGTH) # C_j
    finish_times[0] = schedule[0][1]

    for i in range(1, len(finish_times)):
        finish_times[i] = finish_times[i-1] + schedule[i][1]
        # print(finish_times[i])

    edd_delay = 0
    
    for i in range(0, len(finish_times)):
        delay = finish_times[i] - schedule[i][2]
        if (delay > 0):
            edd_delay += delay
    
    return edd_delay

def init_edd_schedule() -> np.array:
    return np.sort(JOBS, order='deadline')

def init_pheromone_matrix(size) -> np.array:
    delay = calculate_edd_delay(JOBS_EDD)
    fill_value = 1 / (ANTS_NUMBER * delay)

    return np.full((size, size), fill_value)

def init_ant_colony() -> list:
    # return np.full(ANTS_NUMBER, Ant(SCHEDULE_LENGTH))
    result = []
    for ant in range(ANTS_NUMBER):
        result.append(Ant(SCHEDULE_LENGTH))
    
    return result

def pick_best(colony) -> Ant:
    best_ant = colony[0]
    for ant in colony:
        if (ant.delay_score <= best_ant.delay_score):
            best_ant = ant

    return best_ant

def update_feromones(ant, pheromone_matrix):
    for i in range(SCHEDULE_LENGTH):
        for j in range(SCHEDULE_LENGTH):
            if (ant.schedule[i] == j):
                pheromone_matrix[i][j] *= 1 - PHEROMONE_DECREASE_RATE
                pheromone_matrix[i][j] += PHEROMONE_DECREASE_RATE / ant.delay_score
            else:
                pheromone_matrix[i][j] *= 1 - PHEROMONE_DECREASE_RATE

def reset_ants():
    for ant in colony:
        ant.reset_state()

def print_result(best_schedule, best_delay_score):
    print("Best schedule is: " + str(best_schedule))
    print("Best schedule's delay score is: " + str(best_delay_score))

def print_edd(edd_schedule, edd_delay_score):
        print("Best EDD schedule is: " + str(edd_schedule))
        print("Best EDD schedule's delay score is: " + str(edd_delay_score))

# ===== Initialization step =====

# Initialize model hyperparameters and other constans
PHEROMONE_POWER = 1 # alpha
PHEROMONE_DECREASE_RATE = 0.8 # rho in [0, 1]
SUITABILITY_POWER = 30 # beta
JOB_SELECTION_RULE_TRESHOLD = 0.3 # q in [0, 1]

# Initialize number of jobs and size of schedule
JOBS = init_jobs()
SCHEDULE_LENGTH = len(JOBS)
JOBS_EDD = init_edd_schedule()
EDD_SCORE = calculate_edd_delay(JOBS_EDD)

print_edd(JOBS_EDD, EDD_SCORE)

# Initialize number of ants and iterations
ITERATIONS_NUMBER = 200
ANTS_NUMBER = 30
colony = init_ant_colony()

# Initialize suitability matrix (Eta) and pheromone matrix (Tau)
SUITABILITY_MATRIX = init_suitability_matrix(SCHEDULE_LENGTH)
pheromone_matrix = init_pheromone_matrix(SCHEDULE_LENGTH)

# ===== Define ACO caller function =====
def run_ACO():
    # Initialize variables for best schedule
    best_schedule = np.full(SCHEDULE_LENGTH, 0)
    best_delay_score = -1

    for iter in range(ITERATIONS_NUMBER):
        for ant in colony:
            ant.make_schedule()
    
        best_ant = pick_best(colony)
        best_schedule = best_ant.map_to_schedule()
        best_delay_score = best_ant.delay_score

        update_feromones(best_ant, pheromone_matrix)
        reset_ants()
    
    print_result(best_schedule, best_delay_score)

# ===== Run ACO =====
run_ACO()