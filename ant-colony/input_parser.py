import numpy as np
import json
import os
from dotenv import load_dotenv

JOB_STORAGE_TYPES = [('job_number', int), ('processing_time', int), ('deadline', int)]

load_dotenv()

cwd = os.path.dirname(__file__)
input_file = os.getenv('PATH_TO_MODEL_DATA')
input_path = os.path.join(cwd, input_file)

def init_model_params() -> dict:
    model = dict()

    with open(input_path) as file:
        d = json.load(file)
        model["pheromone_power"] = d["model_params"]["pheromone_power"]
        model["pheromone_decrease_rate"] = d["model_params"]["pheromone_decrease_rate"]
        model["suitability_power"] = d["model_params"]["suitability_power"]
        model["job_selection_rule_threshold"] = d["model_params"]["job_selection_rule_threshold"]
        model["iterations"] = d["model_params"]["iterations"]
        model["colony_size"] = d["model_params"]["colony_size"]
    
    return model

def init_jobs() -> np.array:
    job_list = []

    with open(input_path) as file:
        d = json.load(file)
        
        for job in d['jobs']:
            job_tuple = (job['job_number'], job['processing_time'], job['deadline'])
            job_list.append(job_tuple)

    return np.array(job_list, dtype=JOB_STORAGE_TYPES)