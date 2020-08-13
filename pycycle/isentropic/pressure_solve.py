import openmdao.api as om

# from pycycle.cea.species_data import Thermo
from pycycle.cea.explicit_isentropic import ExplicitIsentropic
from pycycle.cea import properties
from pycycle.cea.thermo_lookup import EntropyFromTemp


class PressureSolve(om.Group):

    def initialize(self):
        self.options.declare('mode', default='T', values=('T', 'h', 'S'), desc='mode to calculate in')
        self.options.declare('S_data', default=None, desc='thermodynamic property data')
        self.options.declare('T_base', default=302.4629819, desc='base temperature (units are degK)')
        self.options.declare('P_base', default=1.013, desc='base pressure (units are bar)')
        self.options.declare('S_base', default=1.64322352, desc='base entropy (units are cal/(g*degK)')

    def setup(self):

        mode = self.options['mode']
        S_data = self.options['S_data']
        T_base = self.options['T_base']
        P_base = self.options['P_base']
        S_base = self.options['S_base']

        if mode == 'T' or mode == 'h':
            self.add_subsystem('isentropic_pressure', om.ExecComp('P=Pt*(1 + (gamma - 1)/2 * MN**2)**(-gamma/(gamma - 1))',
                P={'units':'bar', 'lower':'.1'}, Pt={'units':'bar'}, MN={'units':None}, gamma={'units':None}),
                promotes_inputs=('Pt', 'gamma', 'MN'), promotes_outputs=('P',))

            self.add_subsystem('S_table', properties.PropertyMap(map_data=S_data), promotes_inputs=('P', 'T'), promotes_outputs=('S',))
            self.add_subsystem('St_table', properties.PropertyMap(map_data=S_data), promotes_inputs=(('P', 'Pt'), ('T', 'Tt')), promotes_outputs=(('S', 'St'),))
            ## the following impelementation was written to replace the maps, but is ill-defined and needs reworking
            # self.add_subsystem('S_table', EntropyFromTemp(T_base=T_base, P_base=P_base, S_base=S_base
            #     ), promotes_inputs=('P', 'T'), promotes_outputs=('S',))
            # self.add_subsystem('St_table', EntropyFromTemp(T_base=T_base, P_base=P_base, S_base=S_base
            #     ), promotes_inputs=(('P', 'Pt'), ('T', 'Tt')), promotes_outputs=(('S', 'St'),))
            

            self.add_subsystem('entropy_matching', om.BalanceComp('Pt', units='bar', eq_units='cal/(g*degK)', lower=1e-2, res_ref=.001), promotes_outputs=('Pt',))
            self.connect('S', 'entropy_matching.lhs:Pt')
            self.connect('St', 'entropy_matching.rhs:Pt')

        else: #this can be replaced with a pressure lookup using constant Cp
            self.add_subsystem('entropy_matching', om.BalanceComp('P', units='bar', eq_units='cal/(g*degK)', lower=1e-5), promotes_outputs=('P',), promotes_inputs=(('lhs:P', 'S_desired'),))
            self.connect('S_actual', 'entropy_matching.rhs:P')

            # self.add_subsystem('S_table', properties.PropertyMap(map_data=S_data), promotes_inputs=('P', 'T'), promotes_outputs=(('S', 'S_actual'),))
            self.add_subsystem('S_table', EntropyFromTemp(T_base=T_base, P_base=P_base, S_base=S_base
                ), promotes_inputs=('P', 'T'), promotes_outputs=(('S', 'S_actual'),))

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
        if self.options['mode'] != 'S':
            newton.linear_solver = om.LinearRunOnce()
            # newton.linear_solver.options['maxiter'] = 10

        ln_bt = newton.linesearch = om.BoundsEnforceLS()
        ln_bt.options['bound_enforcement'] = 'scalar'
        ln_bt.options['iprint'] = -1

if __name__ == "__main__":

    import scipy

    from pycycle.cea import species_data
    from pycycle import constants
    import numpy as np

    S_data = properties.AIR_MIX_entropy

    prob = om.Problem()
    prob.model = PressureSolve(mode='T', S_data=S_data)
    prob.model.linear_solver = om.DirectSolver()
    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100)
    prob.model.set_input_defaults('gamma', 1.4, units=None)
    prob.model.set_input_defaults('MN', 1.1, units=None)
    prob.model.set_input_defaults('Pt', 1.2, units='bar')
    prob.model.set_input_defaults('T', 300, units='degK')
    prob.model.set_input_defaults('Tt', 352, units='degK')


    # prob.model = PressureSolve(mode='S', S_data=S_data)
    # prob.model.set_input_defaults('T', 300, units='degK')
    # prob.model.set_input_defaults('S_desired', 1.71137505, units='cal/(g*degK)')

    prob.set_solver_print(level=2)




    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)
    print(prob['T'])
    print(prob['P'])
    print(prob['Tt'])
    print(prob['Pt'])
    print(prob['S'])
    print(prob['St'])
    print(prob['gamma'])
    print(prob['MN'])

    # prob.model.list_inputs(units=True)
    # prob.model.list_outputs(units=True)