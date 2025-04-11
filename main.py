from utils.parse import read_data
from utils.greeter import greet
from algorithms.aco import run_ACO
from algorithms.classic import run_classic

if __name__ == '__main__':
    algorithm_to_run = greet()

    if algorithm_to_run == 1: 
        input_filename = 'aco.json'
        data = read_data(input_filename)
        run_ACO(data)
    elif algorithm_to_run == 2: 
        input_filename = 'classic.json'
        data = read_data(input_filename)
        run_classic(data)
