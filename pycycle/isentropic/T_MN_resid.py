import numpy as np

import openmdao.api as om

from pycycle.constants import P_REF, R_UNIVERSAL_ENG, MIN_VALID_CONCENTRATION
from pycycle.isentropic.thermo_lookup import EnthalpyFromTemp

class TMN(om.ImplicitComponent):

    def setup(self):

        self.add_input('h', val=23.51179312, units='cal/g', desc='static enthalpy')
        self.add_input('ht', val=47.91243885, units='cal/g', desc='total enthalpy')
        self.add_input('gamma', val=1.4, units=None, desc='ratio of specific heats')
        self.add_input('R', val=4400.0, units='J/(kg*K)', desc='specific ideal gas constant')
        self.add_input('Tt', val=500, units='degK', desc='total temperature')

        self.add_output('MN', val=.8, units=None, desc='mach number')
        self.add_output('T', val=400, units='degK', desc='static temperature')

        self.declare_partials('MN', ('ht', 'MN', 'gamma', 'R', 'h', 'T'))
        self.declare_partials('T', ('Tt', 'gamma', 'MN', 'T'))
 
    def apply_nonlinear(self, inputs, outputs, resids):

        h = inputs['h']
        ht = inputs['ht']
        gamma = inputs['gamma']
        R = inputs['R']
        Tt = inputs['Tt']

        MN = outputs['MN']
        T = outputs['T']

        dh = MN**2 * gamma*R*T/2
        T_eq = Tt*(1 + (gamma - 1)*MN**2 / 2)**(-1)

        resids['MN'] = h - ht + dh    
        resids['T'] = T - T_eq   

    def linearize(self, inputs, outputs, J):

        h = inputs['h']
        ht = inputs['ht']
        gamma = inputs['gamma']
        R = inputs['R']
        Tt = inputs['Tt']

        MN = outputs['MN']
        T = outputs['T']

        J['MN', 'h'] = 1
        J['MN', 'ht'] = -1
        J['MN', 'MN'] = MN*gamma*R*T
        J['MN', 'gamma'] = MN**2 * R*T/2
        J['MN', 'R'] = MN**2 * gamma * T/2
        J['MN', 'T'] = MN**2 * gamma*R/2

        J['T', 'T'] = 1
        J['T', 'Tt'] = -(1 + (gamma - 1)*MN**2 / 2)**(-1)
        J['T', 'gamma'] = Tt*(1 + (gamma - 1)*MN**2 / 2)**(-2) * (MN**2 / 2)
        J['T', 'MN'] = Tt*(1 + (gamma - 1)*MN**2 / 2)**(-2) * MN*(gamma - 1) 


class TmnResid(om.Group):

    def initialize(self):
        self.options.declare('h_base', default=-78.65840276, desc='enthalpy at base temperature (units are cal/g)')
        self.options.declare('T_base', default=0, desc='base temperature (units are degK)')

    def setup(self):

        h_base = self.options['h_base']
        T_base = self.options['T_base']

        self.add_subsystem('h_table', EnthalpyFromTemp(h_base=h_base, T_base=T_base),
                promotes_inputs=('Cp', 'T'), promotes_outputs=('h',))
        self.add_subsystem('T_MN', TMN(), promotes_inputs=('h', 'ht', 'gamma', 'R', 'Tt'), promotes_outputs=('T',))

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

        ln_bt = newton.linesearch = om.ArmijoGoldsteinLS()
        ln_bt.options['bound_enforcement'] = 'scalar'
        ln_bt.options['iprint'] = -1


if __name__ == "__main__":

    prob = om.Problem()
    prob.model = TmnResid()

    prob.setup(force_alloc_complex=True)


    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)