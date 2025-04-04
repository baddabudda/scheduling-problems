# Task: Ant Colony Optimization for 1 || SUM(T_j)

import random
import numpy as np
import input_parser as parser
import summary

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
        return pheromone_matrix[schedule_pos][job_number] ** pheromone_power * \
                suitability_matrix[schedule_pos][job_number] ** suitability_power

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
        
        if (q < job_selection_rule_threshold):
            return self.pick_by_argmax(schedule_pos)
        else:
            picked_job = np.random.choice(jobs, p=prob_vector)
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
        # types = [('job_number', int), ('processing_time', int), ('deadline', int)]
        list = []

        for schedule_pos in range(self.schedule_size):
            job_id = self.schedule[schedule_pos]
            list.append(jobs_edd[job_id])
        
        return np.array(list, dtype=parser.JOB_STORAGE_TYPES)

# ===== Definition of global methods =====

# def init_jobs() -> np.array:
#     types = [('job_number', int), ('processing_time', int), ('deadline', int)]
#     values = [(1, 3, 5), (2, 1, 2), (3, 1, 3), (4, 2, 1)]
#     values = [(1, 1, 2), (2, 2, 4), (3, 4, 3), (4, 1, 1)]

#     return np.array(values, dtype=types)

def init_suitability_matrix(size) -> np.array:
    result_matrix = np.zeros((size, size))

    for i in range(schedule_length):
        for j in range(schedule_length):
            result_matrix[i][j] = 1 / jobs_edd[j][2]

    return result_matrix

def calculate_edd_delay(schedule) -> int:
    finish_times = np.zeros(schedule_length) # C_j
    finish_times[0] = schedule[0][1]

    for i in range(1, len(finish_times)):
        finish_times[i] = finish_times[i-1] + schedule[i][1]

    edd_delay = 0
    
    for i in range(0, len(finish_times)):
        delay = finish_times[i] - schedule[i][2]
        if (delay > 0):
            edd_delay += delay
    
    return edd_delay

def init_edd_schedule() -> np.array:
    return np.sort(jobs, order='deadline')

def init_pheromone_matrix(size) -> np.array:
    delay = calculate_edd_delay(jobs_edd)
    fill_value = 1 / (colony_size * delay)

    return np.full((size, size), fill_value)

def init_ant_colony() -> list:
    result = []
    for ant in range(colony_size):
        result.append(Ant(schedule_length))
    
    return result

def pick_best(colony) -> Ant:
    best_ant = colony[0]
    for ant in colony:
        if (ant.delay_score <= best_ant.delay_score):
            best_ant = ant

    return best_ant

def update_pheromones(ant, pheromone_matrix):
    for i in range(schedule_length):
        for j in range(schedule_length):
            if (ant.schedule[i] == j):
                pheromone_matrix[i][j] *= 1 - pheromone_decrese_rate
                pheromone_matrix[i][j] += pheromone_decrese_rate / ant.delay_score
            else:
                pheromone_matrix[i][j] *= 1 - pheromone_decrese_rate

def reset_ants():
    for ant in colony:
        ant.reset_state()

def print_aco_result(best_schedule, best_delay_score):
    print("Best schedule is: " + str(best_schedule))
    print("Best schedule's delay score is: " + str(best_delay_score))

def print_edd_result(edd_schedule, edd_delay_score):
    print("Best EDD schedule is: " + str(edd_schedule))
    print("Best EDD schedule's delay score is: " + str(edd_delay_score))

# ===== Initialization step =====

# Initialize model hyperparameters and other constans
model = parser.init_model_params()
pheromone_power = model["pheromone_power"] # alpha
pheromone_decrese_rate = model["pheromone_decrease_rate"] # rho in [0, 1]
suitability_power = model["suitability_power"] # beta
job_selection_rule_threshold = model["job_selection_rule_threshold"] # q in [0, 1]
iterations_number = model["iterations"]
colony_size = model["colony_size"]

# Initialize number of jobs and size of schedule
jobs = parser.init_jobs()
schedule_length = len(jobs)
jobs_edd = init_edd_schedule()
edd_score = calculate_edd_delay(jobs_edd)

# print_edd_result(jobs_edd, edd_score)

# Initialize number of ants and iterations
colony = init_ant_colony()

# Initialize suitability matrix (Eta) and pheromone matrix (Tau)
suitability_matrix = init_suitability_matrix(schedule_length)
pheromone_matrix = init_pheromone_matrix(schedule_length)

# ===== Define ACO caller function =====
def run_ACO():
    # Initialize variables for best schedule
    best_schedule = np.full(schedule_length, 0)
    best_delay_score = -1

    for iter in range(iterations_number):
        for ant in colony:
            ant.make_schedule()
    
        best_ant = pick_best(colony)
        best_schedule = best_ant.map_to_schedule()
        best_delay_score = best_ant.delay_score

        update_pheromones(best_ant, pheromone_matrix)
        reset_ants()
    
    # print_aco_result(best_schedule, best_delay_score)
    summary.print_stats(model, jobs_edd, edd_score, best_schedule, best_delay_score)

# ===== Run ACO =====
run_ACO() # run aco