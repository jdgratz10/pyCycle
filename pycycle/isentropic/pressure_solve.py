import openmdao.api as om

# from pycycle.cea.species_data import Thermo
from pycycle.isentropic.explicit_isentropic import ExplicitIsentropic
import pycycle.isentropic.properties as properties
from pycycle.isentropic import properties

import numpy as np

class IsentropicPressure(om.ExplicitComponent):

    def setup(self):

        self.add_input('Pt', val=2, units='bar', desc='total pressure')
        self.add_input('gamma', val=3, units=None, desc='ratio of specific heats')
        self.add_input('MN', val=4, units=None, desc='mach number')

        self.add_output('P', val=7, units='bar', desc='static pressure')

        self.declare_partials('P', ('Pt', 'gamma', 'MN'))

    def compute(self, inputs, outputs):

        Pt = inputs['Pt']
        gamma = inputs['gamma']
        MN = inputs['MN']

        outputs['P'] = Pt*(1 + (gamma - 1)/2 * MN**2)**(-gamma/(gamma - 1))

    def compute_partials(self, inputs, J):

        Pt = inputs['Pt']
        gamma = inputs['gamma']
        MN = inputs['MN']

        J['P', 'Pt'] = (1 + (gamma - 1)/2 * MN**2)**(-gamma/(gamma - 1))
        J['P', 'gamma'] = Pt*((gamma - 1)/2 * MN**2 + 1)**(-gamma/(gamma - 1)) * ((gamma/(gamma - 1)**2 - 1/(gamma - 1))*np.log((gamma - 1)/2 * MN**2 + 1) - gamma*MN**2 / (2*(gamma - 1)*((gamma - 1)/2*MN**2 + 1)))
        J['P', 'MN'] = Pt*(-gamma/(gamma - 1))*(1 + (gamma - 1)/2*MN**2)**((-2*gamma + 1)/(gamma - 1)) * (gamma - 1)*MN


class PressureSolve(om.Group):

    def initialize(self):
        self.options.declare('mode', default='T', values=('T', 'h', 'S'), desc='mode to calculate in')
        self.options.declare('S_data', default=None, desc='thermodynamic property data')

    def setup(self):

        mode = self.options['mode']
        S_data = self.options['S_data']

        if mode == 'T' or mode == 'h':
            self.add_subsystem('isentropic_pressure', IsentropicPressure(),
                promotes_inputs=('Pt', 'gamma', 'MN'), promotes_outputs=('P',))

            self.add_subsystem('S_table', properties.PropertyMap(map_data=S_data), promotes_inputs=('P', 'T'), promotes_outputs=('S',))
            self.add_subsystem('St_table', properties.PropertyMap(map_data=S_data), promotes_inputs=(('P', 'Pt'), ('T', 'Tt')), promotes_outputs=(('S', 'St'),))
            

            self.add_subsystem('entropy_matching', om.BalanceComp('Pt', units='bar', eq_units='cal/(g*degK)'), promotes_outputs=('Pt',))
            self.connect('S', 'entropy_matching.lhs:Pt')
            self.connect('St', 'entropy_matching.rhs:Pt')

        else:
            self.add_subsystem('entropy_matching', om.BalanceComp('P', units='bar', eq_units='cal/(g*degK)'), promotes_outputs=('P',), promotes_inputs=(('lhs:P', 'S_desired'),))
            self.connect('S_actual', 'entropy_matching.rhs:P')

            self.add_subsystem('S_table', properties.PropertyMap(map_data=S_data), promotes_inputs=('P', 'T'), promotes_outputs=(('S', 'S_actual'),))


    def configure(self):

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-10
        newton.options['rtol'] = 1e-10
        newton.options['maxiter'] = 50
        newton.options['iprint'] = 2
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 50
        newton.options['reraise_child_analysiserror'] = False

        self.options['assembled_jac_type'] = 'dense'
        newton.linear_solver = om.DirectSolver(assemble_jac=True)

        ln_bt = newton.linesearch = om.BoundsEnforceLS()
        ln_bt.options['bound_enforcement'] = 'scalar'
        ln_bt.options['iprint'] = -1

if __name__ == "__main__":

    S_data = properties.AIR_MIX_entropy

    prob = om.Problem()
    prob.model = PressureSolve(mode='T', S_data=S_data)
    prob.model.set_input_defaults('gamma', 1.4, units=None)
    prob.model.set_input_defaults('MN', 1.1, units=None)
    prob.model.set_input_defaults('Pt', 1.2, units='bar')
    prob.model.set_input_defaults('T', 300, units='degK')
    prob.model.set_input_defaults('Tt', 352, units='degK')

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)