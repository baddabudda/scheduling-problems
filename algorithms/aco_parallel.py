import random
import numpy as np
from utils.summary import print_aco_stats

# ===== Definition of Ant class =====
# Each ant stores:
# - schedule size
# - schedule, an array of JOBS' indices
# - available_jobs, a set of unpicked jobs
# - delay_score 
class Ant:
    def __init__(self, schedule_length):
        self.schedule_length = schedule_length 
        self.schedule = np.full(self.schedule_length, -1)
        self.available_jobs = {job for job in range(self.schedule_length)}
        self.penalty_score = -1
    
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
        prob_vector = np.full(self.schedule_length, 0.0)
        overall_job_strength = self.calculate_overall_job_strength(schedule_pos)

        for job in self.available_jobs:
            prob_vector[job] = self.calculate_job_strength(schedule_pos, job) / \
                                overall_job_strength
        
        return prob_vector

    def make_schedule(self):
        # TODO: update to parallel machines
        for schedule_pos in range(self.schedule_length):
            prob_vector = self.calculate_probabilities(schedule_pos)
            picked_job = self.pick_job(prob_vector, schedule_pos)

            self.schedule[schedule_pos] = picked_job
            self.available_jobs.remove(picked_job)

        mapped_schedule = self.map_to_schedule()
        self.penalty_score = calculate_penalty_score(mapped_schedule, machines)
    
    def reset_state(self):
        self.schedule = np.full(self.schedule_length, -1)
        self.penalty_score = -1
        self.available_jobs = {job for job in range(self.schedule_length)}
    
    def map_to_schedule(self) -> np.array:
        types = [('job_number', int), ('processing_time', int), ('deadline', int), ('penalty', int)]
        list = []

        for schedule_pos in range(self.schedule_length):
            job_id = self.schedule[schedule_pos]
            list.append(naive_schedule[job_id])
        
        return np.array(list, dtype=types)

# ===== Define ACO caller function =====
def run_ACO_parallel(data):
    model_params = data['model_params']

    # Initialize model hyperparameters
    global pheromone_power
    global pheromone_decrese_rate
    global suitability_power
    global job_selection_rule_threshold
    pheromone_power = model_params['pheromone_power'] # alpha
    pheromone_decrese_rate = model_params['pheromone_decrease_rate'] # rho in [0, 1]
    suitability_power = model_params['suitability_power'] # beta
    job_selection_rule_threshold = model_params["job_selection_rule_threshold"] # q in [0, 1]

    iterations_number = model_params["iterations"]
    colony_size = model_params["colony_size"]
    global machines
    machines = model_params['number_of_machines']

    # Initialize job-related variables
    global jobs
    jobs = data['jobs']
    schedule_length = len(jobs)

    # Calculate naive schedule
    global naive_schedule
    naive_schedule = create_naive_schedule(jobs)
    naive_score = calculate_penalty_score(naive_schedule, machines)

    # Initialize colony
    colony = init_ant_colony(colony_size, schedule_length)
    
    # Initialize matrices
    global suitability_matrix
    global pheromone_matrix
    suitability_matrix = init_suitability_matrix(naive_schedule)
    pheromone_matrix = init_pheromone_matrix(schedule_length, naive_score, colony_size)

    # Initialize variables for best schedule
    best_schedule = np.full(schedule_length, 0)
    best_penalty_score = -1

    for iter in range(iterations_number):
        for ant in colony:
            ant.make_schedule()
    
        best_ant = pick_best(colony)
        best_schedule = best_ant.map_to_schedule()
        best_penalty_score = best_ant.penalty_score

        update_pheromones(best_ant, schedule_length, pheromone_matrix)
        reset_ants(colony)
    
    # Print result
    print_aco_stats(data['model_params'], naive_schedule, naive_score, best_schedule, best_penalty_score)

def create_naive_schedule(jobs) -> np.array:
    """
    Create naive schedule.
    Jobs are ordered based on penalties in decreasing order.
    """
    sorted_by_penalties = np.sort(jobs, order='penalty')
    return np.array(sorted_by_penalties[::-1])

def calculate_penalty_score(schedule, machines) -> int:
    """
    Calculate overall penalty for the given schedule.
    """
    overall_penalty = 0
    
    cur_time = -1
    for i in range(len(schedule)):
        machine_id = i % machines
        if machine_id == 0:
            cur_time += 1
        # print('Cur time: {0}, Deadline: {1}, Sum: {2}'.format(cur_time, schedule[i][2], cur_time + schedule[i][1]))
        if schedule[i][2] < cur_time + schedule[i][1]:
            overall_penalty += schedule[i][3]
    
    return overall_penalty

def init_ant_colony(colony_size, schedule_length) -> list:
    result = []
    for ant in range(colony_size):
        result.append(Ant(schedule_length))
    
    return result

def init_suitability_matrix(heuristic_schedule) -> np.array:
    """
    Create suitability matrix based on heuristic.
    """
    schedule_length = len(heuristic_schedule)
    result_matrix = np.zeros((schedule_length, schedule_length))

    for i in range(schedule_length):
        for j in range(schedule_length):
            result_matrix[i][j] = 1 / heuristic_schedule[j][3]

    return result_matrix

def init_pheromone_matrix(schedule_length, penalty_score, colony_size) -> np.array:
    fill_value = 1 / (colony_size * penalty_score)

    return np.full((schedule_length, schedule_length), fill_value)

def pick_best(colony) -> Ant:
    best_ant = colony[0]
    for ant in colony:
        if (ant.penalty_score <= best_ant.penalty_score):
            best_ant = ant

    return best_ant

def update_pheromones(ant, schedule_length, pheromone_matrix):
    for i in range(schedule_length):
        for j in range(schedule_length):
            if (ant.schedule[i] == j):
                pheromone_matrix[i][j] *= 1 - pheromone_decrese_rate
                pheromone_matrix[i][j] += pheromone_decrese_rate / ant.penalty_score
            else:
                pheromone_matrix[i][j] *= 1 - pheromone_decrese_rate

def reset_ants(colony):
    for ant in colony:
        ant.reset_state()
