import openmdao.api as om

# from pycycle.cea.species_data import Thermo
from pycycle.isentropic.explicit_isentropic import ExplicitIsentropic
import pycycle.isentropic.properties as properties
from pycycle.isentropic import properties

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
        

class TempFromGamma(om.ExplicitComponent):

    def setup(self):

        self.add_input('Tt', units='degK', desc='total temperature')
        self.add_input('gamma', units=None, desc='ratio of specific heats')
        self.add_input('MN', units=None, desc='mach number')

        self.add_output('T', units='degK', desc='static temperature')

        self.declare_partials('T', ('Tt', 'gamma', 'MN'))

    def compute(self, inputs, outputs):

        Tt = inputs['Tt']
        gamma = inputs['gamma']
        MN = inputs['MN']

        outputs['T'] = Tt*(1 + (gamma - 1)/2 * MN**2)**(-1)

    def  compute_partials(self, inputs, J):

        Tt = inputs['Tt']
        gamma = inputs['gamma']
        MN = inputs['MN']

        J['T', 'Tt'] = (1 + (gamma - 1)/2 * MN**2)**(-1)
        J['T', 'gamma'] = -Tt/(1 + (gamma - 1)/2 * MN**2)**2 * MN**2 / 2
        J['T', 'MN'] = -Tt/(1 + (gamma - 1)/2 * MN**2)**2 * (gamma - 1)*MN


class TempFromSP(om.Group):

    def initialize(self):
        self.options.declare('S_data', default=None, desc='thermodynamic property data')

    def setup(self):
        S_data = self.options['S_data']

        if S_data is None:
            raise ValueError('You have not provided data to TLookup and it is required')

        self.add_subsystem('S_table', properties.PropertyMap(map_data=S_data), promotes_inputs=('P', 'T'), promotes_outputs=(('S', 'S_calculated'),))

        self.add_subsystem('entropy_matching', om.BalanceComp('T', units='degK', eq_units='cal/(g*degK)'), promotes_outputs=('T', ), promotes_inputs=(('lhs:T', 'S'),))
        self.connect('S_calculated', 'entropy_matching.rhs:T')

    def configure(self):

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
        newton.linear_solver = om.DirectSolver(assemble_jac=True)

        ln_bt = newton.linesearch = om.ArmijoGoldsteinLS()
        ln_bt.options['bound_enforcement'] = 'scalar'
        ln_bt.options['iprint'] = -1


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

if __name__ == "__main__":

    import scipy

    from pycycle.cea import species_data
    from pycycle import constants
    import numpy as np

    S_data = properties.AIR_MIX_entropy
    # h_data = properties.AIR_MIX_enthalpy

    # prob = om.Problem()


    # prob.model = TLookup(mode='h', data=h_data)
    # prob.model.set_input_defaults('h', -0.59153318, units='cal/g')

    # # prob.model = TLookup(mode='S', data=S_data)
    # # prob.model.set_input_defaults('P', 1.2, units='bar')
    # # prob.model.set_input_defaults('S', 2.5, units='cal/(g*degK)')


    # prob.set_solver_print(level=2)




    # prob.setup(force_alloc_complex=True)
    # prob.set_val('T', 500, units='degK')







    # prob.run_model()
    # prob.check_partials(method='cs', compact_print=True)
    # print(prob['h'])
    # print(prob['T'])
    # # print(prob['P'])
    # # print(prob['S_calculated'])
    # # print(prob['S'])

    # # prob.model.list_inputs(units=True, prom_name=True)
    # # prob.model.list_outputs(units=True, prom_name=True)

    # p = om.Problem()
    # p.model = om.Group()
    # p.model.add_subsystem('temp', TempFromEnthalpy(), promotes=['*'])
    # p.setup(force_alloc_complex=True)
    # p.run_model()
    # p.check_partials(method='cs', compact_print=True)

    prob = om.Problem()
    prob.model = om.Group()
    prob.model.add_subsystem('temp', TempFromGamma())
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)
