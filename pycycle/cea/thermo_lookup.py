import openmdao.api as om

# from pycycle.cea.species_data import Thermo
from pycycle.cea.explicit_isentropic import ExplicitIsentropic
import pycycle.cea.properties as properties


class ThermoLookup(om.Group):

    def initialize(self):
        # self.options.declare('thermo_data', desc='thermodynamic data set', recordable=False)
        self.options.declare('Cp', default=False, values=(True, False), desc='switch to tell whether to look up Cp')
        self.options.declare('h', default=False, values=(True, False), desc='switch to tell whether to look up Cp')
        self.options.declare('S', default=False, values=(True, False), desc='switch to tell whether to look up Cp')
        self.options.declare('Cp_data', default=None, desc='thermodynamic property data')
        self.options.declare('h_data', default=None, desc='thermodynamic property data')
        self.options.declare('S_data', default=None, desc='thermodynamic property data')

    def setup(self):

        Cp = self.options['Cp']
        h = self.options['h']
        S = self.options['S']

        Cp_data = self.options['Cp_data']
        h_data = self.options['h_data']
        S_data = self.options['S_data']

        if Cp is True:
            if Cp_data == None:
                raise ValueError('You have requested a Cp value from ThermoLookup but you have not provided Cp_data, which is required')

            self.add_subsystem('Cp_table', properties.PropertyMap(
                    map_data=Cp_data), promotes_inputs=('T',), promotes_outputs=('Cp',))

        if h is True:
            if h_data == None:
                raise ValueError('You have requested an h value from ThermoLookup but you have not provided h_data, which is required')
                
            self.add_subsystem('h_table', properties.PropertyMap(
                    map_data=h_data), promotes_inputs=('T',), promotes_outputs=('h',))

        if S is True:
            if S_data == None:
                raise ValueError('You have requested a S value from ThermoLookup but you have not provided S_data, which is required')
                
            self.add_subsystem('S_table', properties.PropertyMap(
                    map_data=S_data), promotes_inputs=('T', 'P'), promotes_outputs=('S',))

if __name__ == "__main__":

    import scipy

    from pycycle.cea import species_data
    from pycycle import constants
    import numpy as np

    prob = om.Problem()
    prob.model = ThermoLookup(Cp=True, h=True, S=True, Cp_data=properties.AIR_MIX_Cp, h_data=properties.AIR_MIX_enthalpy, S_data=properties.AIR_MIX_entropy)
    prob.model.set_input_defaults('T', 800, units='degK')
    prob.model.set_input_defaults('P', 1, units='bar')
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)
    print(prob['T'])
    print(prob['P'])
    print(prob['S'])
    print(prob['h'])
    print(prob['Cp'])