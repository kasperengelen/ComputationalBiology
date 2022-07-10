"""
    File that contains fitting algorithms for finding the value of parameters.

    Copyright (C) 2019  Kasper Engelen, Lotte Leys, William Verbiest

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from math import pow
import csv

import matplotlib.pyplot as plt

import numpy as np

from data_processing import process_dataset
from BdModelTotalSEIRZ import BdModelTotalSEIRZ
from BdModelTotalSEISZ import BdModelTotalSEISZ

def fitter(measure, measure_args, param_candidates, print_progress=True, progress_steps = None):
    """
        Find the best parameter for the specified model.

        Params:
            'measure': The measure that will be used to determine the quality of the parameter. It takes a parameter value as first argument. A lower return value indicates a better parameter value.
            'measure_args': Extra parameters that will be passed to the measure function.
            'param_candidates': An iterable that contains the parameter candidates. The len() function should be applyable to this iterable.
            'print_progress': Whether or not to print how many candidates have been tried.
            'progress_steps': Number of candidates in between printouts.
    """

    # don't take len() if we don't print output
    if print_progress and progress_steps is None:
        progress_steps = len(param_candidates) / 10

    current_best = None
    current_lowest_score = None

    current_candidate_nr = 0

    for candidate in param_candidates:
        param_error = measure(candidate, *measure_args)

        if current_best == None or current_lowest_score > param_error:
            current_best = candidate
            current_lowest_score = param_error

        if print_progress and current_candidate_nr % progress_steps == 0:
            print("Processing... [{}/{}] Current best value: {}, Current best score: {}".format(current_candidate_nr+1,
                                                                                                len(param_candidates),
                                                                                                current_best,
                                                                                                current_lowest_score))

        current_candidate_nr += 1

    return current_best

def sum_squared_residuals(model_values, data_values):
    """
        Determine the sum of squared residuals between the specified model output
        and the specified data.
    """

    if not len(model_values) == len(data_values):
        raise Exception("Error when calculating RSS: {} model values, and {} data values specified. ".format(len(model_values), len(data_values)))

    SSR = sum(pow(model_values[i] - data_values[i], 2) for i in range(0, len(model_values)))

    return SSR

def residuals(model, pop):
    """
        Gives an iterable that contains the residuals.
    """

    if not model in ["SEIRZ", "SEISZ"]:
        raise Exception("Invalid model type: {}".format(model_type))

    if not pop in ["RIP", "TER"]:
        raise Exception("Invalid population: {}".format(population))

    # retrieve riparian
    data_rip = process_dataset('riparian')

    # retrieve terrestrial
    data_ter = process_dataset('terrestrial')

    if model == "SEIRZ":
        model_class = BdModelTotalSEIRZ
    else:
        model_class = BdModelTotalSEISZ

    model = model_class(rip_init_pop = data_rip['start_pop_1998'], ter_init_pop = data_ter['start_pop_1998'])
    model.run()
    #model.plot()

    time_vec = model.getTimeticks()

    if pop == "RIP":
        result = model.getRiparianTotal()
        target = data_rip['fit_total'](time_vec)
    else:
        result = model.getTerrestrialTotal()
        target = data_ter['fit_total'](time_vec)

    return [(result[i] - target[i]) for i in range(0, len(time_vec))]

def residuals_prepost_outbreak(model, pop):
    """
        Returnes two iterables that returns two residuals for pre and post outbreak.
    """

    if not model in ["SEIRZ", "SEISZ"]:
        raise Exception("Invalid model type: {}".format(model_type))

    if not pop in ["RIP", "TER"]:
        raise Exception("Invalid population: {}".format(population))

    # retrieve riparian
    data_rip = process_dataset('riparian')

    # retrieve terrestrial
    data_ter = process_dataset('terrestrial')

    if model == "SEIRZ":
        model_class = BdModelTotalSEIRZ
    else:
        model_class = BdModelTotalSEISZ

    model = model_class(rip_init_pop = data_rip['start_pop_1998'], ter_init_pop = data_ter['start_pop_1998'])
    model.run()
    #model.plot()

    time_vec = model.getTimeticks()

    time_vec_pre = time_vec[time_vec <= data_rip['outbreak_date']]
    time_vec_post = time_vec[time_vec > data_rip['outbreak_date']]

    if pop == "RIP":
        result = model.getRiparianTotal()
        target_pre  = data_rip['fit_total'](time_vec_pre)
        target_post = data_rip['fit_total'](time_vec_post)
    else:
        result = model.getTerrestrialTotal()
        target_pre  = data_ter['fit_total'](time_vec_pre)
        target_post = data_ter['fit_total'](time_vec_post)

    result_pre = np.asarray(result[:len(time_vec_pre)])
    result_post = np.asarray(result[len(time_vec_pre):])

    return ([(result_pre[i] - target_pre[i]) for i in range(0, len(result_pre))], [(result_post[i] - target_post[i]) for i in range(0, len(result_post))])

### lambda fitting ###

def lambda_measure(lambda_value, model_class, data_ter, data_rip, population):
    """
        Determine how well the specified population growth parameter works.

        Params:
            'lambda_value': The population growth for the population in absolute numbers relative to the original population.
            'model_class': The class of the model that the parameter will be fitted for.
            'data_ter': Information about the terrestrial population.
            'data_rip': Information about the riparian population.
            'population': Which population the growth will be fitted for.
    """

    model = model_class(rip_init_pop = data_rip['start_pop_1998'], ter_init_pop = data_ter['start_pop_1998'])

    if population == "RIP":
        model.parameters.lamb_rip = lambda_value
    else:
        model.parameters.lamb_ter = lambda_value

    model.run()

    time_vec = model.getTimeticks()
    
    if population == "RIP":
        result = model.getRiparianTotal()
        target = data_rip['fit_total'](time_vec)
    else:
        result = model.getTerrestrialTotal()
        target = data_ter['fit_total'](time_vec)
    
    SSR = sum_squared_residuals(result, target)

    return SSR


def find_best_lambda(model_type, population):
    """
        Find the optimal lambda value for the specified model type and the specified population.

        Params:
            'model_type': 'SEIRZ' or 'SEISZ'
            'population': 'RIP' or 'TER'
    """

    if not model_type in ["SEIRZ", "SEISZ"]:
        raise Exception("Invalid model type: {}".format(model_type))

    if not population in ["RIP", "TER"]:
        raise Exception("Invalid population: {}".format(population))

    # retrieve riparian
    data_rip = process_dataset('riparian')

    # retrieve terrestrial
    data_ter = process_dataset('terrestrial')

    if model_type == "SEIRZ":
        model_class = BdModelTotalSEIRZ
    else:
        model_class = BdModelTotalSEISZ

    measure_args = [model_class, data_ter, data_rip, population]
    measure = lambda_measure

    if population == "RIP" and model_type == "SEIRZ":
        param_range = np.linspace(0.95, 0.99, num=500)
    elif population == "TER" and model_type == "SEIRZ":
        param_range = np.linspace(0.55, 0.59, num=500)
    elif population == "RIP" and model_type == "SEISZ":
        param_range = np.linspace(1.18, 1.22, num=500)
    elif population == "TER" and model_type == "SEISZ":
        param_range = np.linspace(0.66, 0.70, num=500)

    return fitter(measure, measure_args, param_candidates = param_range, print_progress = True, progress_steps = 50)


def find_lambda():
    """
        Find optimal lambda values for SEIRZ and SEISZ, for riparian and terrestrial.
    """
    SEIRZ_RIP = find_best_lambda("SEIRZ", "RIP")
    print("SEIRZ, riparian:", SEIRZ_RIP)
    SEIRZ_TER = find_best_lambda("SEIRZ", "TER")
    print("SEIRZ, terrestrial:", SEIRZ_TER)
    SEISZ_RIP = find_best_lambda("SEISZ", "RIP")
    print("SEISZ, riparian:", SEISZ_RIP)
    SEISZ_TER = find_best_lambda("SEISZ", "TER")
    print("SEISZ, terrestrial:", SEISZ_TER)

def write_residuals_to_file():
    """
        Determine the residual values for SEIRZ and SEISZ, for riparian and terrestrial and print them to
        a CSV file.
    """
    for pop in ["TER", "RIP"]:
        for model in ["SEIRZ", "SEISZ"]:
            filename = pop + "_" + model + ".csv"
            residuals = residuals(model = model, pop = pop)

            print(residuals)

            with open(filename, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows([[residual] for residual in residuals])

def write_residuals_to_file_prepost_outbreak():
    """
        Determine the residual values for SEIRZ and SEISZ for riparian and terrestrial and print them to
        a CSV file.
    """
    for pop in ["TER", "RIP"]:
        for model in ["SEIRZ", "SEISZ"]:
            filename_pre = pop + "_" + model + "_PRE.csv"
            filename_post = pop + "_" + model + "_POST.csv"
            residuals = residuals_prepost_outbreak(model = model, pop = pop)

            with open(filename_pre, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows([[residual] for residual in residuals[0]])

            with open(filename_post, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows([[residual] for residual in residuals[1]])

def calc_SEISZ_model_accuracy():
    """
        Determine the residual for the model using least-squares.
    """

    # retrieve riparian
    data_rip = process_dataset('riparian')

    # retrieve terrestrial
    data_ter = process_dataset('terrestrial')

    model = BdModelTotalSEISZ(rip_init_pop = data_rip['start_pop_1998'], ter_init_pop = data_ter['start_pop_1998'])
    model.run()
    model.plot()

    time_vec = model.getTimeticks()


    result_rip = model.getRiparianTotal()
    target_rip = data_rip['fit_total'](time_vec)

    result_ter = model.getTerrestrialTotal()
    target_ter = data_ter['fit_total'](time_vec)

    SSR_rip = sum_squared_residuals(result_rip, target_rip)
    SSR_ter = sum_squared_residuals(result_ter, target_ter)

    print("SSR riparian:", SSR_rip)
    print("SSR terrestrial:", SSR_ter)
