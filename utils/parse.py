import numpy as np
import json
import os

def read_data(filename) -> dict:
    input_path = './input/{0}'.format(filename)
    data = dict()

    with open(input_path) as file:
        d = json.load(file)

        if ('model_params' in d):
            model_params = dict()

            for param in d['model_params']:
                model_params[param] = d['model_params'][param]
            
            data['model_params'] = model_params
        
        if ('jobs' in d):
            job_list = []
            batch_no = 0

            if len(d['jobs']) > 1:
                message = 'Multiple job batches found: {0}. Pick one: '.format(len(d['jobs']))
                batch_no = int(input(message))
            
            types = parse_types(d['jobs'][batch_no][0])

            for job in d['jobs'][batch_no]:
                # job_tuple = (job['job_no'], job['proc_time'], job['deadline'])
                job_tuple = tuple(job.values())
                job_list.append(job_tuple)
            
            # types = [('job_number', int), ('processing_time', int), ('deadline', int)]
            data['jobs'] = np.array(job_list, dtype=types)
        
        if len(data) == 0:
            raise Exception('Invalid input file format. ' \
                            'Supported input file parameters:\n' \
                            '- model_params (optional): object\n' \
                            '- jobs (required): array = [[{job_no, proc_time, deadline}, ...], [{job_no, proc_time, deadline}, ...], ...] ')
        
    return data

def parse_types(job_sample) -> list:
    if 'release_date' in job_sample.keys():
        return [('job_number', int), ('processing_time', int), ('release_date', int), ('deadline', int)]

    return [('job_number', int), ('processing_time', int), ('deadline', int)]
