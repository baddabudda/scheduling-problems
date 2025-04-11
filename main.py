from utils.parse import read_data
from algorithms.aco import run_ACO
from algorithms.classic import run_classic

if __name__ == '__main__':
    # input_filename = 'aco-input.json'
    # data = read_data(input_filename)
    # run_ACO(data)
    input_filename = 'classic-input.json'
    data = read_data(input_filename)
    run_classic(data)
