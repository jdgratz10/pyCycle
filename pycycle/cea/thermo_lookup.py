import openmdao.api as om
import numpy as np

# from pycycle.cea.species_data import Thermo
from pycycle.cea.explicit_isentropic import ExplicitIsentropic
import pycycle.cea.properties as properties

class EnthalpyFromTemp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('h_base', default=0.0, desc='enthalpy at base temperature (units are cal/g)')
        self.options.declare('T_base', default=302.4629819, desc='base temperature (units are degK)')

    def setup(self):
        self.add_input('Cp', units='cal/(g*degK)', desc='specific heat (assumed constant)')
        self.add_input('T', units='degK', desc='temperature at which to find enthalpy')

        self.add_output('h', units='cal/g', desc='enthalpy at input temperature assuming constant specific heat')

        self.declare_partials('h', ('Cp', 'T'))

    def compute(self, inputs, outputs):

        Cp = inputs['Cp']
        T = inputs['T']
        h_base = self.options['h_base']
        T_base = self.options['T_base']

        outputs['h'] = h_base + Cp*(T - T_base)

    def compute_partials(self, inputs, J):

        Cp = inputs['Cp']
        T = inputs['T']
        T_base = self.options['T_base']

        J['h', 'Cp'] = T - T_base
        J['h', 'T'] = Cp





class EntropyFromTemp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('T_base', default=302.4629819, desc='base temperature (units are degK)')
        self.options.declare('P_base', default=1.013, desc='base pressure (units are bar)')
        self.options.declare('S_base', default=1.64322352, desc='base entropy (units are cal/(g*degK)')

    def setup(self):
        self.add_input('Cp', units='cal/(g*degK)', desc='specific heat (assumed constant)')
        self.add_input('T', units='degK', desc='entropy at input pressure and output temperature')
        self.add_input('R', units='cal/(g*degK)', desc='specific gas constant')
        self.add_input('P', units='bar', desc='pressure to find temperature at')

        self.add_output('S', units='cal/(g*degK)', desc='temperature at specified pressure and entropy')

        self.declare_partials('S', ('Cp', 'T', 'R', 'P'))

    def compute(self, inputs, outputs):

        T_base = self.options['T_base']
        P_base = self.options['P_base']
        S_base = self.options['S_base']

        Cp = inputs['Cp']
        T = inputs['T']
        R = inputs['R']
        P = inputs['P']

        outputs['S'] = S_base + Cp*np.log(T/T_base) - R*np.log(P/P_base)

        print(self.pathname)
        print(T)
        print(P)
        print(outputs['S'])

    def compute_partials(self, inputs, J):

        T_base = self.options['T_base']
        P_base = self.options['P_base']
        S_base = self.options['S_base']

        Cp = inputs['Cp']
        T = inputs['T']
        R = inputs['R']
        P = inputs['P']

        J['S', 'Cp'] = np.log(T/T_base)
        J['S', 'T'] = Cp/T
        J['S', 'R'] = -np.log(P/P_base)
        J['S', 'P'] = -R/P


class ThermoLookup(om.Group):

    def initialize(self):
        # self.options.declare('thermo_data', desc='thermodynamic data set', recordable=False)
        self.options.declare('h', default=False, values=(True, False), desc='switch to tell whether to look up Cp')
        self.options.declare('S', default=False, values=(True, False), desc='switch to tell whether to look up Cp')
        self.options.declare('h_base', default=-78.65840276, desc='enthalpy at base temperature (units are cal/g)')
        self.options.declare('T_base', default=0, desc='base temperature (units are degK)')
        self.options.declare('P_base', default=1.013, desc='base pressure (units are bar)')
        self.options.declare('S_base', default=1.64322352, desc='base entropy (units are cal/(g*degK)')

    def setup(self):

        h = self.options['h']
        S = self.options['S']

        T_base = self.options['T_base']

        if h is True:

            h_base = self.options['h_base']

            self.add_subsystem('h_table', EnthalpyFromTemp(h_base=h_base, T_base=T_base),
                promotes_inputs=('Cp', 'T'), promotes_outputs=('h',))

        if S is True:

            P_base = self.options['P_base']
            S_base = self.options['S_base']

            self.add_subsystem('S_table', EntropyFromTemp(T_base=T_base, P_base=P_base, S_base=S_base), 
                promotes_inputs=('T', 'P', 'Cp', 'R'), promotes_outputs=('S',))

if __name__ == "__main__":

    import scipy

    from pycycle.cea import species_data
    from pycycle import constants
    import numpy as np

    # prob = om.Problem()
    # prob.model = ThermoLookup(Cp=True, h=True, S=True, Cp_data=properties.AIR_MIX_Cp, h_data=properties.AIR_MIX_enthalpy, S_data=properties.AIR_MIX_entropy)
    # prob.model.set_input_defaults('T', 800, units='degK')
    # prob.model.set_input_defaults('P', 1, units='bar')
    # prob.setup(force_alloc_complex=True)
    # prob.run_model()
    # prob.check_partials(method='cs', compact_print=True)
    # print(prob['T'])
    # print(prob['P'])
    # print(prob['S'])
    # print(prob['h'])
    # print(prob['Cp'])

    p = om.Problem()
    p.model = om.Group()
    p.model.add_subsystem('enth', EntropyFromTemp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.run_model()
    p.check_partials(method='cs', compact_print=True)