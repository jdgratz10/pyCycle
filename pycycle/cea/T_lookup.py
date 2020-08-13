import openmdao.api as om
import numpy as np

# from pycycle.cea.species_data import Thermo
from pycycle.cea.explicit_isentropic import ExplicitIsentropic
import pycycle.cea.properties as properties
from pycycle.cea.thermo_lookup import ThermoLookup
from pycycle.cea import properties

class TempFromEnthalpy(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('h_base', default=0.0, desc='enthalpy at base temperature (units are cal/g)')
        self.options.declare('T_base', default=302.4629819, desc='base temperature (units are degK)')

    def setup(self):
        self.add_input('Cp', units='cal/(g*degK)', desc='specific heat (assumed constant)')
        self.add_input('h', units='cal/g', desc='enthalpy at input temperature assuming constant specific heat')

        self.add_output('T', units='degK', desc='temperature at which to find enthalpy')
        
        self.declare_partials('T', ('Cp', 'h'))

    def compute(self, inputs, outputs):

        Cp = inputs['Cp']
        h = inputs['h']
        h_base = self.options['h_base']
        T_base = self.options['T_base']

        outputs['T'] = 1/Cp * (h - h_base) + T_base

    def compute_partials(self, inputs, J):

        Cp = inputs['Cp']
        h = inputs['h']
        h_base = self.options['h_base']

        J['T', 'Cp'] = -1/Cp**2 * (h - h_base)
        J['T', 'h'] = 1/Cp



class TempFromEntropy(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('T_base', default=302.4629819, desc='base temperature (units are degK)')
        self.options.declare('P_base', default=1.013, desc='base pressure (units are bar)')
        self.options.declare('S_base', default=1.64322352, desc='base entropy (units are cal/(g*degK)')

    def setup(self):
        self.add_input('Cp', units='cal/(g*degK)', desc='specific heat (assumed constant)')
        self.add_input('S', units='cal/(g*degK)', desc='entropy at input pressure and output temperature')
        self.add_input('R', units='cal/(g*degK)', desc='specific gas constant')
        self.add_input('P', units='bar', desc='pressure to find temperature at')

        self.add_output('T', units='degK', desc='temperature at specified pressure and entropy')

        self.declare_partials('T', ('Cp', 'S', 'R', 'P'))

    def compute(self, inputs, outputs):

        T_base = self.options['T_base']
        P_base = self.options['P_base']
        S_base = self.options['S_base']

        Cp = inputs['Cp']
        S = inputs['S']
        R = inputs['R']
        P = inputs['P']

        outputs['T'] = T_base*np.exp((1/Cp)*(S - S_base + R*np.log(P/P_base)))

    def compute_partials(self, inputs, J):

        T_base = self.options['T_base']
        P_base = self.options['P_base']
        S_base = self.options['S_base']

        Cp = inputs['Cp']
        S = inputs['S']
        R = inputs['R']
        P = inputs['P']

        T = T_base*np.exp((1/Cp)*(S - S_base + R*np.log(P/P_base)))

        J['T', 'Cp'] = T*(-1/Cp**2)*(S - S_base + R*np.log(P/P_base))
        J['T', 'S'] = T/Cp
        J['T', 'R'] = T*np.log(P/P_base)
        J['T', 'P'] = T*R/P



class TLookup(om.Group):

    def initialize(self):
        self.options.declare('mode', values=('h', 'S'), desc='switch to tell whether to look up Cp')
        self.options.declare('h_base', default=0.0, desc='enthalpy at base temperature (units are cal/g)')
        self.options.declare('T_base', default=302.4629819, desc='base temperature (units are degK)')
        self.options.declare('P_base', default=1.013, desc='base pressure (units are bar)')
        self.options.declare('S_base', default=1.64322352, desc='base entropy (units are cal/(g*degK)')

    def setup(self):

        comp_mode = self.options['mode']
        T_base = self.options['T_base']

        if comp_mode == 'h':

            h_base=self.options['h_base']

            self.add_subsystem('h_table', TempFromEnthalpy(h_base=h_base, T_base=T_base),
                promotes_inputs=('h', 'Cp'), promotes_outputs=('T',))

        else:

            P_base = self.options['P_base']
            S_base = self.options['S_base']

            # self.add_subsystem('S_table', properties.PropertyMap(map_data=S_data), promotes_inputs=('P', 'T'), promotes_outputs=(('S', 'S_calculated'),))

            # self.add_subsystem('entropy_matching', om.BalanceComp('T', units='degK', eq_units='cal/(g*degK)'), promotes_outputs=('T', ), promotes_inputs=(('lhs:T', 'S'),))
            # self.connect('S_calculated', 'entropy_matching.rhs:T')

            self.add_subsystem('S_table', TempFromEntropy(T_base=T_base, P_base=P_base, S_base=S_base), 
                promotes_inputs=('P', 'S', 'Cp', 'R'), promotes_outputs=('T',))

    def configure(self):

        mode = self.options['mode']

        if mode == 'S':
            newton = self.nonlinear_solver = om.NewtonSolver()
            newton.options['atol'] = 1e-10
            newton.options['rtol'] = 1e-10
            newton.options['maxiter'] = 50
            newton.options['iprint'] = 2
            newton.options['solve_subsystems'] = True
            newton.options['max_sub_solves'] = 50
            newton.options['reraise_child_analysiserror'] = False


            newton.options['debug_print'] = True



            self.options['assembled_jac_type'] = 'dense'
            # newton.linear_solver = om.DirectSolver(assemble_jac=True)
            newton.linear_solver = om.ScipyKrylov()

            ln_bt = newton.linesearch = om.ArmijoGoldsteinLS()
            ln_bt.options['bound_enforcement'] = 'scalar'
            ln_bt.options['iprint'] = -1

if __name__ == "__main__":

    import scipy

    from pycycle.cea import species_data
    from pycycle import constants
    import numpy as np

    S_data = properties.AIR_MIX_entropy

    prob = om.Problem()


    prob.model = TLookup(mode='S', S_data=S_data)
    # prob.model.set_input_defaults('h', -0.59153318, units='cal/g')

    # prob.model = TLookup(mode='S', data=S_data)
    # prob.model.set_input_defaults('P', 1.2, units='bar')
    # prob.model.set_input_defaults('S', 2.5, units='cal/(g*degK)')


    prob.set_solver_print(level=2)




    prob.setup(force_alloc_complex=True)
    prob.set_val('T', 500, units='degK')







    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)
    # print(prob['P'])
    # print(prob['S_calculated'])
    # print(prob['S'])

    # prob.model.list_inputs(units=True, prom_name=True)
    # prob.model.list_outputs(units=True, prom_name=True)

    p = om.Problem()
    p.model = om.Group()
    p.model.add_subsystem('temp', TempFromEntropy(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.run_model()
    p.check_partials(method='cs', compact_print=True)
