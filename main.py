from utils.parse import read_data
from utils.greeter import greet
from algorithms.aco import run_ACO
from algorithms.classic import run_classic
from algorithms.parallel_flow import run_parallel_flow
from algorithms.aco_parallel import run_ACO_parallel

if __name__ == '__main__':
    # algorithm_to_run = greet()
    algorithm_to_run = 'aco_parallel'

    if algorithm_to_run == 'aco': 
        input_filename = 'aco.json'
        data = read_data(input_filename)
        run_ACO(data)
    elif algorithm_to_run == 'classic': 
        input_filename = 'classic.json'
        data = read_data(input_filename)
        run_classic(data)
    elif algorithm_to_run == 'parallel_flow':
        input_filename = 'parallel-flow.json'
        data = read_data(input_filename)
        run_parallel_flow(data)
    elif algorithm_to_run == 'aco_parallel':
        input_filename = 'aco-parallel.json'
        data = read_data(input_filename)
        run_ACO_parallel(data)
