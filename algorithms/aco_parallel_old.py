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
        self.delay_score = calculate_overall_penalty(mapped_schedule)
    
    def reset_state(self):
        self.schedule = np.full(self.schedule_size, -1)
        self.delay_score = -1
        self.available_jobs = {job for job in range(self.schedule_size)}
    
    def map_to_schedule(self) -> np.array:
        types = [('job_number', int), ('processing_time', int), ('deadline', int)]
        list = []

        for schedule_pos in range(self.schedule_size):
            job_id = self.schedule[schedule_pos]
            list.append(jobs_edd[job_id])
        
        return np.array(list, dtype=types)

# ===== Definition of global methods =====
def init_suitability_matrix(size) -> np.array:
    """
    Calculate heuristics matrix (eta_ij) based on penalties.
    """
    result_matrix = np.zeros((size, size))

    # sorted_by_penalty = np.sort(jobs, order='penalty')
    # sorted_by_penalty = sorted_by_penalty[::-1]


    for i in range(number_of_jobs):
        for j in range(number_of_jobs):
            result_matrix[i][j] = 1 / naive_schedule[]

    return result_matrix

def calculate_overall_penalty(schedule) -> int:
    """
    Calculate penalty score for the given schedule.
    """
    overall_penalty = 0

    for i in range(time_slots):
        for j in range(machines):
            if schedule[i][j] != None and schedule[i][j][2] < i + schedule[i][j][1]:
                overall_penalty += schedule[i][j][3]
    
    return overall_penalty

def init_penalty_schedule() -> list:
    """
    Create penalty-based schedule.
    Sort jobs by ascending penalties.
    """
    schedule = [[None for i in range(machines)] for i in range(time_slots)]
    sorted_by_deadline = [job for job in np.sort(jobs, order='deadline').tolist()]
    current_time = 0
    optimal_set = set()
    leftover_jobs = set()

    for job in sorted_by_deadline:
        if job[2] <= current_time:
            # tweak optimal set
            job_to_remove = min(optimal_set)
            if job_to_remove[3] < job[3]:
                replace_with(schedule, job_to_remove, job)
                optimal_set.remove(job_to_remove)
                leftover_jobs.add(job_to_remove)
                optimal_set.add(job)
        else:
            optimal_set.add(job)
            machine_id = add_to_schedule(schedule, current_time, job)
            if machine_id == machines - 1:
                current_time += 1
    
    # print(leftover_jobs)
    if len(leftover_jobs) != 0:
        fill_with_leftovers(schedule, leftover_jobs)

    # OLD: DELETE
    # return np.sort(jobs, order='deadline')
    return schedule

def replace_with(schedule, job_to_remove, job_to_add):
    for i in range(len(schedule)):
        for j in range(machines):
            if job_to_remove[0] == schedule[i][j][0]:
                schedule[i][j] = job_to_add
                return
    
    print('no match found')
    return

def add_to_schedule(schedule, time_slot, job_to_add) -> int:
    for i in range(machines):
        if schedule[time_slot][i] == None:
            schedule[time_slot][i] = job_to_add 
            return i

def fill_with_leftovers(schedule, leftovers):
    for i in range(time_slots):
        for j in range(machines):
            if schedule[i][j] == None and len(leftovers) != 0:
                schedule[i][j] = leftovers.pop()

def create_naive_schedule() -> list:
    sorted_by_penalty = np.sort(jobs, order='penalty')
    sorted_by_penalty = sorted_by_penalty[::-1]
    schedule = [[None for i in range(machines)] for j in range(time_slots)]

    cur_slot = 0
    cur_machine = 0
    for i in range(len(sorted_by_penalty)):
        new_machine = i % machines
        if new_machine == 0 and cur_machine != 0:
            cur_slot += 1
        cur_machine = new_machine
        schedule[cur_slot][cur_machine] = sorted_by_penalty[i]

    return schedule

def init_pheromone_matrix(size) -> np.array:
    """
    Calculate pheromone matrix.
    Penalty score is calculated best on 
    """
    fill_value = 1 / (colony_size * penalty_score)

    return np.full((size, size), fill_value)

def init_ant_colony() -> list:
    result = []
    for ant in range(colony_size):
        result.append(Ant(number_of_jobs))
    
    return result

def pick_best(colony) -> Ant:
    best_ant = colony[0]
    for ant in colony:
        if (ant.delay_score <= best_ant.delay_score):
            best_ant = ant

    return best_ant

def update_pheromones(ant, pheromone_matrix):
    for i in range(number_of_jobs):
        for j in range(number_of_jobs):
            if (ant.schedule[i] == j):
                pheromone_matrix[i][j] *= 1 - pheromone_decrese_rate
                pheromone_matrix[i][j] += pheromone_decrese_rate / ant.delay_score
            else:
                pheromone_matrix[i][j] *= 1 - pheromone_decrese_rate

def reset_ants():
    for ant in colony:
        ant.reset_state()

# ===== Initialization step =====
def init_data(data):
    model_params = data['model_params']
    # Initialize model hyperparameters and other constants
    global pheromone_power 
    pheromone_power = model_params['pheromone_power'] # alpha
    global pheromone_decrese_rate
    pheromone_decrese_rate = model_params['pheromone_decrease_rate'] # rho in [0, 1]
    global suitability_power
    suitability_power = model_params['suitability_power'] # beta
    global job_selection_rule_threshold
    job_selection_rule_threshold = model_params["job_selection_rule_threshold"] # q in [0, 1]
    global iterations_number
    iterations_number = model_params["iterations"]
    global colony_size
    colony_size = model_params["colony_size"]
    global machines 
    machines = model_params['number_of_machines']

    # Initialize number of jobs and size of the schedule
    global jobs
    jobs = data['jobs']
    global number_of_jobs
    number_of_jobs = len(jobs)
    global time_slots
    time_slots = number_of_jobs // machines
    if number_of_jobs % machines > 0:
        time_slots += 1

    global naive_schedule
    naive_schedule = create_naive_schedule()
    global penalty_score
    penalty_score = calculate_overall_penalty(naive_schedule)

    # classic algorithm
    global jobs_by_penalty
    jobs_by_penalty = init_penalty_schedule()

    # Initialize colony
    # global colony
    # colony = init_ant_colony()
    
    # Initialize matrices
    global suitability_matrix
    suitability_matrix = init_suitability_matrix(number_of_jobs)
    print(suitability_matrix)
    global pheromone_matrix
    pheromone_matrix = init_pheromone_matrix(number_of_jobs)

# ===== Define ACO caller function =====
def run_ACO_parallel(data):
    init_data(data)

    # Initialize variables for best schedule
    # best_schedule = np.full(number_of_jobs, 0)
    # best_delay_score = -1

    # for iter in range(iterations_number):
    #     for ant in colony:
    #         ant.make_schedule()
    
    #     best_ant = pick_best(colony)
    #     best_schedule = best_ant.map_to_schedule()
    #     best_delay_score = best_ant.delay_score

    #     update_pheromones(best_ant, pheromone_matrix)
    #     reset_ants()
    
    # # Print result
    # print_aco_stats(data['model_params'], jobs_edd, edd_score, best_schedule, best_delay_score)
