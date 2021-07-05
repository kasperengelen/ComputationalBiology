
from CompartmentalModel import CompartmentalModel
from data_processing import process_dataset
import matplotlib.pyplot as plt
import numpy as np

np.warnings.filterwarnings('ignore')

class BdModelTotalSEIRZ(CompartmentalModel):
    """
        A compartmental model for the population, both terrestrial and riparian. This model
        uses the SEIR(Z) compartment configuration.
    """

    def __init__(self, rip_init_pop, ter_init_pop, plot_all = False):
        """
            Constructor.

            Params:
                'ter_init_pop': The initial terrestrial population.
                'rip_init_pop': The initial riparian population.
        """

        super(BdModelTotalSEIRZ, self).__init__(init_time = 1998, y_max = 8000000, y_min = -50000)

        # 1-1-1998 => 1-1-2005: 7 years
        self.simulation_length = 7 * 365

        # first infection year: 2003.67
        # the data says that the outbreak happened at 2003.8, but there is a delay
        # so the outbreak occurred earlier than the data shows.
        # 2003.67 - 1998 = 5.67 years
        # 5.67 years * 365 days/year = 2070 days
        self.day0 = 2070

        # set initial populations
        self.rip_init_pop = rip_init_pop
        self.ter_init_pop = ter_init_pop

        # retrieve compartments and parameters
        comp  = self.compartments
        param = self.parameters

        # init terrestrial compartments
        comp.S_ter = 1.0
        comp.E_ter = 0.0
        comp.I_ter = 0.0
        comp.R_ter = 0.0
        comp.Z_ter = 0.0

        # init riparian compartments
        comp.S_rip = 1.0
        comp.E_rip = 0.0
        comp.I_rip = 0.0
        comp.R_rip = 0.0
        comp.Z_rip = 0.0

        self.updateTotals()

        # set plotting parameters
        comp.setVarPlotArgs("S_rip",       do_plot=False, color='blue',        pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("S_rip_total", do_plot=plot_all, color='blue',        pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("S_ter",       do_plot=False, color='deepskyblue', pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("S_ter_total", do_plot=plot_all, color='deepskyblue', pattern='-', linewidth=1.0)

        comp.setVarPlotArgs("E_rip",       do_plot=False, color='red',    pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("E_rip_total", do_plot=plot_all, color='red',    pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("E_ter",       do_plot=False, color='orange', pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("E_ter_total", do_plot=plot_all, color='orange', pattern='-', linewidth=1.0)

        comp.setVarPlotArgs("I_rip",       do_plot=False, color='darkgreen', pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("I_rip_total", do_plot=plot_all, color='darkgreen', pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("I_ter",       do_plot=False, color='palegreen', pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("I_ter_total", do_plot=plot_all, color='palegreen', pattern='-', linewidth=1.0)

        comp.setVarPlotArgs("R_rip",       do_plot=False, color='fuchsia', pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("R_rip_total", do_plot=plot_all, color='fuchsia', pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("R_ter",       do_plot=False, color='plum',    pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("R_ter_total", do_plot=plot_all, color='plum',    pattern='-', linewidth=1.0)

        # disable plotting of the spores
        comp.setVarPlotArgs("Z_rip",       do_plot=False, color='black', pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("Z_rip_total", do_plot=False, color='black', pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("Z_ter",       do_plot=False, color='black', pattern='-', linewidth=1.0)
        comp.setVarPlotArgs("Z_ter_total", do_plot=False, color='black', pattern='-', linewidth=1.0)

        comp.setVarPlotArgs("total_ter",       do_plot=True, color='silver', pattern='-', linewidth=3.0, label="Terrestrial Model")
        comp.setVarPlotArgs("total_rip",       do_plot=True, color='black',  pattern='-', linewidth=3.0, label="Riparian Model")

        # NOTE: the time unit of these parameters is year^-1

        # direct-contact contamination
        # 0.15/day => 54.75/year
        param.beta = 54.75

        # incubation period
        # 10-47 days => 10 days
        param.sigma = (1/10) * 365

        # mortality of infectious individuals
        # 24-220 days => 24 days
        param.alpha = (1/24) * 365

        # rate at which individuals are cured
        # 0.071/day => 25.915/yr
        param.gamma = 25.915

        # absolute grownth of the terrestrial population
        # OLD: 0.6951803989203453
        param.lamb_ter = 0.5760521042084168

        # absolute grownth of the riparian population
        # OLD: 1.2242897805796555
        param.lamb_rip = 0.9759519038076151

        # number of spores that die each year
        # 1-365/yr => 10/year
        param.mu = 10

        # number of spores that are spread by an individual per year
        # 26.9 - 51.4 / day => 51.4/day => 18761 / year
        param.phi = 18761

        # minimum number of spores to have 50% chance of infection.
        param.k = 10000

        # environmental transmission rate
        param.eps = 0.036512426

    def getTerrestrialTotal(self):
        return np.asarray(self.compartments.getVarHistory('total_ter'))

    def getRiparianTotal(self):
        return np.asarray(self.compartments.getVarHistory('total_rip'))

    def updateTotals(self):
        comp = self.compartments

        comp.S_rip_total = comp.S_rip * self.rip_init_pop
        comp.E_rip_total = comp.E_rip * self.rip_init_pop
        comp.I_rip_total = comp.I_rip * self.rip_init_pop
        comp.R_rip_total = comp.R_rip * self.rip_init_pop
        comp.Z_rip_total = comp.Z_rip * self.rip_init_pop

        comp.total_rip = (comp.S_rip + comp.E_rip + comp.I_rip + comp.R_rip) * self.rip_init_pop

        comp.S_ter_total = comp.S_ter * self.ter_init_pop
        comp.E_ter_total = comp.E_ter * self.ter_init_pop
        comp.I_ter_total = comp.I_ter * self.ter_init_pop
        comp.R_ter_total = comp.R_ter * self.ter_init_pop
        comp.Z_ter_total = comp.Z_ter * self.ter_init_pop

        comp.total_ter = (comp.S_ter + comp.E_ter + comp.I_ter + comp.R_ter) * self.ter_init_pop

    def update(self):
        super(BdModelTotalSEIRZ, self).update(1.0 / 365.0)

        # time unit is 1/365th of a year = 1 day
        dt = 1.0 / 365.0

        # retrieve compartments and parameters
        old = self.compartments.copyVars() # make copy so we can reference values from previous timestep
        new = self.compartments
        param = self.parameters

        def infl_z(Z, k):
            """
                Function that models the influence of the Z compartment.
            """
            return (Z / (Z + k))

        # on day zero: infect 5% of the population
        if len(self.time_ticks) == self.day0:
            new.S_ter = old.S_ter - 0.05 * old.S_ter
            new.E_ter = old.E_ter
            new.I_ter = old.I_ter + 0.05 * old.S_ter
            new.Z_ter = old.Z_ter
            new.R_ter = old.R_ter

            new.S_rip = old.S_rip - 0.05 * old.S_rip
            new.E_rip = old.E_rip
            new.I_rip = old.I_rip + 0.05 * old.S_rip
            new.Z_rip = old.Z_rip
            new.R_rip = old.R_rip

        # on other days we run the normal formulas
        else:

            ## terrestrial ##

            new.S_ter = old.S_ter - (param.beta * old.S_ter * old.I_ter) * dt + param.lamb_ter * dt 
            new.E_ter = old.E_ter + (param.beta * old.S_ter * old.I_ter) * dt - (param.sigma * infl_z(old.Z_ter, param.k) * old.E_ter) * dt
            new.I_ter = old.I_ter + (param.sigma * infl_z(old.Z_ter, param.k) * old.E_ter) * dt - (param.gamma * infl_z(old.Z_ter, param.k) * old.I_ter) * dt - (param.alpha * infl_z(old.Z_ter, param.k) * old.I_ter) * dt
            new.Z_ter = old.Z_ter + (param.phi * old.I_rip) * dt - (param.mu * old.Z_ter) * dt
            new.R_ter = old.R_ter + (param.gamma * infl_z(old.Z_ter, param.k) * old.I_ter) * dt

            ## riparian ##
            new.S_rip = old.S_rip - (param.beta * old.S_rip * old.I_rip) * dt + param.lamb_rip * dt - (param.eps * infl_z(old.Z_rip, param.k)) * dt
            new.E_rip = old.E_rip + (param.beta * old.S_rip * old.I_rip) * dt - (param.sigma * infl_z(old.Z_rip, param.k) * old.E_rip) * dt + (param.eps * infl_z(old.Z_rip, param.k)) * dt
            new.I_rip = old.I_rip + (param.sigma * infl_z(old.Z_rip, param.k) * old.E_rip) * dt - (param.gamma * infl_z(old.Z_rip, param.k) * old.I_rip) * dt - (param.alpha * infl_z(old.Z_rip, param.k) * old.I_rip) * dt
            new.Z_rip = old.Z_rip + (param.phi * old.I_rip) * dt - (param.mu * old.Z_rip) * dt
            new.R_rip = old.R_rip + (param.gamma * infl_z(old.Z_rip, param.k) * old.I_rip) * dt

        # update totals
        self.updateTotals()

    def run(self):
        """
            Run the model.
        """

        for _ in range(0, self.simulation_length):
            self.update()


def main():
    """
        Runs the script.
    """

    # whether or not to plot the S,E,I,R compartments. Set to False to only plot the total
    PLOT_ALL_COMPARTMENTS = True

    # whether or not to plot the original data points
    PLOT_DATA = True

    # whether or not to plot the fitting linears
    PLOT_FITTING_LINEAR = True


    # retrieve datasets
    total_range = np.linspace(1998, 2005, num=100)

    ter_data = process_dataset('terrestrial')
    ter_init_pop_abs = ter_data['start_pop_1998']

    rip_data = process_dataset('riparian')
    rip_init_pop_abs = rip_data['start_pop_1998']

    # run model
    model = BdModelTotalSEIRZ(rip_init_pop = rip_init_pop_abs, ter_init_pop = ter_init_pop_abs, plot_all = PLOT_ALL_COMPARTMENTS)
    model.run()

    # plot data
    if PLOT_DATA:
        plt.plot(ter_data['timesteps'],   ter_data['population'], 'v', label="Terrestrial data", markersize=4.0)
        plt.plot(rip_data['timesteps'],   rip_data['population'], 'o', label="Riparian data", markersize=3.0)

    # plot fitting linear
    if PLOT_FITTING_LINEAR:
        plt.plot(total_range,  ter_data['fit_total'](total_range), linewidth=5, color='brown', label="Terrestrial Fit")
        plt.plot(total_range,  rip_data['fit_total'](total_range), linewidth=5, color='purple', label="Riparian Fit")

    # plot model
    model.plot(show=False)

    # finish plot
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
