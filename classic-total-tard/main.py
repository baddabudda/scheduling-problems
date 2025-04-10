import numpy as np
import math

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
    proc_times = np.array([job[1] for job in jobs])
    max_proc_idx = np.argmax(proc_times)
    return max_proc_idx

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


def init_jobs() -> np.array:
    types = [('job_number', int), ('processing_time', int), ('deadline', int)]
    jobs = [(4, 83, 336), (1, 121, 260), (2, 79, 266), (3, 147, 266), (5, 130, 337)]
    jobs_array = np.sort(np.array(jobs, dtype=types), order='deadline')

    return jobs_array

def run_classic():
    jobs = init_jobs()
    schedule = sequence(0, jobs)
    print(schedule)

run_classic()