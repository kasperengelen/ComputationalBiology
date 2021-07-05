"""
    Script for simulating compartment models in epidemiology. This script provides several classes that represent the different components of the model.
"""

import copy
import matplotlib.pyplot as plt
import numpy as np

class CompartmentalModel:
    """
    Class that represents an entire model.
    """
    def __init__(self, init_time, y_max = None, y_min = None):
        """
        Constructor. This initialises all the compartments and the parameters.

        Params:
            'init_time': The time at which the models begins the simulation. This will be used for plotting.
        """

        self.compartments = VariableContainer()
        self.parameters   = VariableContainer()
        self.time_ticks = [init_time]
        self.y_max = y_max
        self.y_min = y_min

    def update(self, real_time_passed):
        """
            Update the model over a time period.

            Params:
                'real_time_passed': the amount of real time that has passed. This will be used for plotting.
        """
        # keep track of time units
        self.time_ticks += [self.time_ticks[-1] + real_time_passed]

    def plot(self, show=True):
        """
        Returns a matplotlib figure that contains a visualisation of the model.
        """

        # add each compartment to the plot
        for name, values in self.compartments:
            if len(self.time_ticks) != len(values):
                raise Exception("Error when plotting compartment '{}': compartment has {} values while model performed {} timesteps.".format(name, len(values), len(self.time_ticks)))

            plot_args = self.compartments.getVarPlotArgs(name)

            label = name

            if plot_args['label'] is not None:
                label = plot_args['label']

            if plot_args['do_plot']:
                plt.plot(self.time_ticks, values, plot_args['pattern'], linewidth=plot_args['linewidth'], color=plot_args['color'], label=label)
        
        if self.y_min is not None:
            plt.ylim(bottom=self.y_min)

        if self.y_max is not None:
        	plt.ylim(top = self.y_max)

        plt.gca().ticklabel_format(style='plain', useOffset=False) # disable scientific notation or offsets
        plt.xlabel("Time")
        plt.ylabel("Individuals")
        #plt.legend()

        if show:
            plt.show()

    def getTimeticks(self):
        return np.asarray(self.time_ticks)


class VariableContainer:
    """
    Class that keeps track of a set of variables and their values, both current and previous.
    Variables can be created using "x.varname = value"
    Variables can be accessed using "x.varname", this will return the most current value of the variable.
    To retrieve the complete vector of values of a certain variable use "x.getVarValues(varname)".
    To retrieve a permanent copy of the variables for later reference, use "x.copyVars()"
    """

    def __init__(self, variable_dict = None):
        if variable_dict is None:
            variable_dict = dict()
        self.__dict__["vars"] = variable_dict
        self.__dict__["var_plot_args"] = dict()

    def copyVars(self):
        """
            Returns a deep-copy of the variable container.
        """

        variable_dict = dict()

        for name, value in self:
            variable_dict[name] = copy.copy(value)

        return VariableContainer(variable_dict)

    def __getattr__(self, name):
        if name not in self.vars:
            raise Exception("Variable with name '{}' does not exist.".format(name))
    
        return self.vars[name][-1]

    def __setattr__(self, name, value):
        if name not in self.vars:
            self.vars[name] = []
            self.__dict__["var_plot_args"][name] = {
                "color": 'black',
                "linewidth": 1.0,
                "pattern": '-',
                "do_plot": True
            }
        
        self.vars[name].append(value)

    def __iter__(self):
    	for name, value in self.vars.items():
    	    yield name, value

    def getVarHistory(self, varname):
        return self.__dict__["vars"][varname]

    def getVarPlotArgs(self, varname):
        return self.__dict__["var_plot_args"][varname]

    def setVarPlotArgs(self, varname, do_plot, color, linewidth, pattern, label=None):

        self.__dict__["var_plot_args"][varname] = {
            "label": label,
            "color": color,
            "linewidth": linewidth,
            "pattern": pattern,
            "do_plot": do_plot
        }



    	    
