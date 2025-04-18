import numpy as np
import math
from utils.summary import print_classic_stats

def calculate_total_tardiness(start_time, jobs) -> int:
    finish_times = np.zeros(len(jobs))
    finish_times[0] = start_time + jobs[0][1]

    for i in range(1, len(jobs)):
        finish_times[i] = finish_times[i-1] + jobs[i][1]

    total_tardiness = 0

    for i in range(len(jobs)):
        current_tardiness = finish_times[i] - jobs[i][2]
        if (current_tardiness > 0):
            total_tardiness += current_tardiness

    return total_tardiness

def find_pivot(jobs) -> int:
    best_pivot = (-1, -1, -1)
    
    for i in range(len(jobs)):
        if jobs[i][1] > best_pivot[1]:
            best_pivot = jobs[i]
        elif jobs[i][1] == best_pivot[1] and jobs[i][2] > best_pivot[2]:
            best_pivot = jobs[i]

    # proc_times = np.array([job[1] for job in jobs])
    # max_proc_idx = np.argmax(proc_times)
    return best_pivot[0] - 1 

def calculate_start_time(start_time, previous_jobs):
    if len(previous_jobs) == 0:
        return start_time
    
    result_time = start_time
    for job in previous_jobs:
        result_time += job[1]
    
    return result_time

def sequence(start_time, jobs) -> np.array:
    if (len(jobs) == 0):
        return jobs
    elif (len(jobs) == 1):
        return jobs
    
    best_total_tardiness = math.inf
    best_schedule = jobs
    pivot = find_pivot(jobs)

    for j in range(pivot, len(jobs)):
        sub_sched_before = np.concatenate((jobs[:pivot], jobs[pivot+1:j+1]))
        start_time_before = start_time
        sub_sched_before_ordered = sequence(start_time_before, sub_sched_before)

        sub_sched_after = jobs[j+1:len(jobs)]
        start_time_after = calculate_start_time(start_time, np.append(sub_sched_before, jobs[pivot]))
        sub_sched_after_ordered = sequence(start_time_after, sub_sched_after)

        sched_merged = np.concatenate((np.append(sub_sched_before_ordered, jobs[pivot]), sub_sched_after_ordered))
        current_total_tardiness = calculate_total_tardiness(start_time, sched_merged)

        if current_total_tardiness < best_total_tardiness:
            best_total_tardiness = current_total_tardiness
            best_schedule = sched_merged
        
    return best_schedule 

def run_classic(data):
    jobs = data['jobs']
    schedule = sequence(0, jobs)
    total_tardiness = calculate_total_tardiness(0, schedule)
    print_classic_stats(schedule, total_tardiness)
