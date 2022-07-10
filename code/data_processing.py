"""
    File that contains data processing facilities.

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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.optimize

DATASETS = {
    'riparian':    pd.read_csv("data/riparian.csv", header=None),
    'terrestrial': pd.read_csv("data/terrestrial_filtered.csv", header=None)
}

# note: in meters
TRAIL_WIDTH = {
    'terrestrial': 3.0,
    'riparian'   : 4.0
}

# note: in meters
TOTAL_AREA = 252000000

AREA = {
    'terrestrial': TOTAL_AREA * 0.95,
    'riparian'   : TOTAL_AREA * 0.05
}

def filter_data_by_year(data, year_begin, year_end):
    """
        Extract stats from the specified data from between the two specified years.
        Note: years may be float values.
    """

    return data[(data[0] >= year_begin) & (data[0] <= year_end)]

def process_dataset(dataset_name):
    """
        Process the specified dataset. This will return a dict with the following entries:

        'timesteps': The different time steps at which the population was measured.
        'population': The population sizes through time. For each values in 'timesteps' there is a value in 'population'.
        'outbreak_date': The year at which the outbreak becomes visible.
        'fit_healthy': Linear fit for the pre-outbreak population.
        'fit_disease': Linear fit for the post-outbreak population.
        'fit_total':   Fit that combines fit_healthy and fit_disease.
        'start_pop_1998': The population size in 1998.
        'growth_abs': The absolute population growth per year.
        'growth_rel': The population growth per year relative to the startingpopulation in 1998.
    """

    # retrieve dataset
    dataset = DATASETS[dataset_name]

    # make unique and sort dataset
    dataset.drop_duplicates(subset=0, keep='first', inplace=True)
    dataset.sort_values(by=0, inplace=True)

    # survey dimensions
    area        = AREA[dataset_name]
    trail_width = TRAIL_WIDTH[dataset_name]

    # calculate population sizes throughout the time
    time_values      = dataset[0]
    population_sizes = np.exp(dataset[1]) * area / trail_width

    # determine outbreak time
    # note: this is not when the infection is introduced, this is when the
    # disease becomes visible in the data.
    outbreak_time = 2003.8

    # extract pre-outbreak data
    dataset_pre_outbreak    = filter_data_by_year(dataset, 0, outbreak_time)
    timevalues_pre_outbreak = dataset_pre_outbreak[0]
    pop_pre_outbreak        = (np.exp(dataset_pre_outbreak[1]) / trail_width) * area
    fit_pre_outbreak        = np.poly1d(np.polyfit(timevalues_pre_outbreak, pop_pre_outbreak, 1))

    # extract post-outbreak data
    dataset_post_outbreak    = filter_data_by_year(dataset, outbreak_time, 10000)
    timevalues_post_outbreak = dataset_post_outbreak[0]
    pop_post_outbreak        = (np.exp(dataset_post_outbreak[1]) / trail_width) * area

    # linear function through one point
    # this line will begin at the last healthy population size
    def post_outbreak_linear(x, a):
        x0 = outbreak_time
        y0 = fit_pre_outbreak(outbreak_time)
        return a*(x-x0) + y0

    # perform fitting, and use parameters in linear
    popt, pcov = scipy.optimize.curve_fit(post_outbreak_linear, timevalues_post_outbreak, pop_post_outbreak)
    fit_post_outbreak_a = popt[0]
    fit_post_outbreak_b = post_outbreak_linear(0, fit_post_outbreak_a)
    fit_post_outbreak = np.poly1d([fit_post_outbreak_a, fit_post_outbreak_b])

    total_fit_range = np.linspace(-2.5, 2.5, 6)

    def fit_total(year):
        """
            Fit for the entire dataset.
        """

        # piecewise evaluation
        pre_outbreak_f = fit_pre_outbreak(year[year <= outbreak_time])
        post_outbreak_f = fit_post_outbreak(year[year >= outbreak_time])

        return np.concatenate([pre_outbreak_f, post_outbreak_f])

    # some population parameters such as start population and growth
    start_pop_1998 = fit_pre_outbreak(1998)
    growth_abs = fit_pre_outbreak[1]
    growth_rel = growth_abs / start_pop_1998

    return {
        'timesteps': time_values,
        'outbreak_date': outbreak_time,
        'fit_total':   fit_total,
        'start_pop_1998': start_pop_1998,
        'growth_abs': growth_abs,
        'growth_rel': growth_rel,
        'population': population_sizes
    }

def plot_dataset(dataset_name):
    """
        Plot the dataset with the specified name. 'riparian' or 'terrestrial'.
    """
    dataset = process_dataset(dataset_name)

    timesteps = dataset['timesteps']
    population = dataset['population']

    time_range = np.linspace(1998, 2005, num=400)
    fit_total = dataset['fit_total']

    plt.plot(timesteps, population, 'v', label="Survey")
    plt.plot(time_range, fit_total(time_range), linewidth = 4.0, color='crimson', label="Linear Fit")
    plt.ylim(bottom=-2000)
    plt.gca().ticklabel_format(style='plain', useOffset=False) # disable scientific notation or offsets
    plt.xlabel("Time")
    plt.ylabel("Population")

    plt.legend()
    plt.show()
