import os

def greet() -> int:
    algorithms = dict() 
    code = 1
    for filename in os.listdir('./algorithms'):
        if filename.startswith(('.', '__')):
            continue
        algorithms[code] = filename[:-3]
        code += 1

    print()
    print('Hello! Type in the code of the algorithm that you\'d like to run:')
    for entry in algorithms:
        print('{0} - {1}'.format(entry, algorithms[entry]))
    
    while True:
        selected_code = int(input('Code: '))
        if selected_code in algorithms.keys():
            return selected_code
        else:
            print('Wrong input. Try again!')
