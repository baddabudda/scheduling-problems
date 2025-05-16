import numpy as np
from utils.summary import print_parallel_flow_answer

def run_parallel_flow(data):
    machines, jobs = init_data(data)

    # initialize interval vertices 
    release_deadline_list = jobs[['release_date', 'deadline']]
    interval_vertices = create_interval_vertices(len(jobs) + 1, release_deadline_list)

    # initialize adjacency and capacity matrix for max flow problem
    adjacency_matrix = create_adjacency_matrix(jobs, interval_vertices)
    capacity_matrix = create_capacity_matrix(adjacency_matrix, jobs, interval_vertices, machines)

    # find max flow in the network
    flow = find_max_flow(adjacency_matrix, capacity_matrix)
    # print(flow)

    # calculate threshold value: sum of processing times
    possibility_threshold = sum([job['processing_time'] for job in jobs])
    # print(possibility_threshold)

    print_parallel_flow_answer(flow == possibility_threshold)

def init_data(data) -> (int, np.array):
    """
    Parse input data into algorithm entities.
    """
    machines = data['model_params']['number_of_machines']
    jobs = data['jobs']

    return machines, jobs

def create_interval_vertices(start_id, rel_due_list) -> dict:
    """
    Create interval vertices (I_k) from release and due dates.
    """
    time_stamps = order_timestamps(rel_due_list)
    interval_vertices = dict()
    for i in range(len(time_stamps) - 1):
        # data format: { vertex_id: (time_start, time_end, interval_length) }
        interval_vertices[start_id + i] = (time_stamps[i], time_stamps[i+1], time_stamps[i+1] - time_stamps[i])
    
    # debug purposes
    # print(time_vertices)
    
    return interval_vertices

def order_timestamps(rel_due_list) -> list:
    """
    Flatten the list of (release_date, deadline) tuples into a list with unique timestamps, sorted in ascending order.
    """
    flattened = [time for time_tuple in rel_due_list for time in time_tuple]
    flattened.sort()
    unique_only = list(dict.fromkeys(flattened))

    return unique_only

def create_adjacency_matrix(jobs, intervals) -> list:
    """
    Create ajacency matrix (unidirected graph) for the given array of jobs and interval vertices.
    Adjacency matrix indexing looks as follows (n - number of jobs, r - number of interval vertices):

    [0, 1, ..., n, n+1, ..., n+r-1, n+r]
    
    0 index is reserved for source; n+r index is reserved for sink.
    Indices from 1 to n are reserved for job vertices.
    Each job vertex is adjacent to the source.

    Indices from n+1 to n+r-1 are reserved for interval vertices.
    Each interval vertex is adjacent either to job vertices or the sink.
    Also, interval vertex is adjacent to a specific job vertex J_i iff:

    r_i <= t_k && t_(k+1) <= d_i, I_k = [t_k, t_(k+1)]
    """
    adjacency_matrix = [list() for i in range(2 + len(jobs) + len(intervals))]

    # fill source -> J_i adjacency
    for i in range(len(jobs)):
        adjacency_matrix[0].append(i + 1)
        adjacency_matrix[1 + i].append(0)
    
    # fill J_i -> I_k adjacency
    start_id = 1 + len(jobs)
    for i in range(len(jobs)):
        for j in range(len(intervals)):
            if jobs[i]['release_date'] <= intervals[start_id + j][0] and intervals[start_id + j][1] <= jobs[i]['deadline']:
                adjacency_matrix[1 + i].append(1 + len(jobs) + j)
                adjacency_matrix[1 + len(jobs) + j].append(1 + i)
    
    # fill I_k -> sink adjacency
    for i in range(len(intervals)):
        adjacency_matrix[start_id + i].append(len(adjacency_matrix) - 1)
        adjacency_matrix[len(adjacency_matrix) - 1].append(start_id + i)
    
    # debug purposes
    # print(adjacency_matrix)
    
    return adjacency_matrix

def create_capacity_matrix(adj_matrix, jobs, intervals, machines) -> list:
    """
    Create capacity matrix based on adjacency matrix.
    Capacity matrix is a (2+n+r)*(2+n+r) matrix, where n - number of jobs, r - number of intervals.

    For each arc (source -> J_i) capacity equals to job's processing time.
    For each arc (J_i -> I_k) capacity equals to interval's length, t_(k+1) - t_k.
    For each arc (I_k -> sink) capacity equals to number_of_machines * interval_length.

    In other cases the default value is 0.
    """
    capacity_matrix = [[0 for i in range(len(adj_matrix))] for j in range(len(adj_matrix))]

    # fill s -> Ji capacity: processing time of job i 
    for i in range(len(jobs)):
        capacity_matrix[0][1 + i] = jobs[i]['processing_time']
    
    # fill Ji -> Ik capacity: interval's length Tk = time_end - time_start
    start_id = 1 + len(jobs)
    for i in range(len(jobs)):
        for j in range(len(adj_matrix[1 + i]) - 1):
            current_interval_id = adj_matrix[1 + i][1 + j]
            current_interval_length = intervals[current_interval_id][2]
            capacity_matrix[1 + i][current_interval_id] = current_interval_length
    
    # fill Ik -> t capacity: m * Tk, where m - number of machines
    for i in range(len(intervals)):
        current_interval_id = start_id + i
        capacity_matrix[current_interval_id][len(adj_matrix) - 1] = machines * intervals[current_interval_id][2]
    
    # debug purposes
    # for i in range(len(capacity_matrix)):
    #     print(capacity_matrix[i])

    return capacity_matrix

def find_max_flow(adj_matrix, cap_matrix) -> int:
    """
    Find maximum flow for the constructed graph (Ford-Fulkerson method).
    """
    flow = 0
    parent = [-1 for i in range(len(adj_matrix))]

    source_vertex = 0
    sink_vertex = len(adj_matrix) - 1

    while True:
        new_flow = bfs(source_vertex, sink_vertex, parent, adj_matrix, cap_matrix)
        if new_flow <= 0:
            break

        flow += new_flow

        current_vertex = sink_vertex

        while current_vertex != source_vertex:
            previous_vertex = parent[current_vertex]
            cap_matrix[previous_vertex][current_vertex] -= new_flow
            cap_matrix[current_vertex][previous_vertex] += new_flow
            current_vertex = previous_vertex

        
    return flow 

def bfs(source, sink, parent, adj_matrix, cap_matrix) -> int:
    """
    Perform Breadth-First-Search to find a new augmenting path. 
    """
    for i in range(len(parent)):
        parent[i] = -1
    parent[source] = -2

    queue = []
    queue.append((source, np.inf))

    while len(queue) != 0:
        current_vertex, flow = queue.pop(0)

        for neighbor in adj_matrix[current_vertex]:
            if parent[neighbor] == -1 and cap_matrix[current_vertex][neighbor] > 0:
                parent[neighbor] = current_vertex
                new_flow = min((flow, cap_matrix[current_vertex][neighbor]))
                if neighbor == sink:
                    return new_flow
                queue.append((neighbor, new_flow))
    
    return 0
