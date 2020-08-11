import openmdao.api as om

# from pycycle.cea.species_data import Thermo
from pycycle.cea.explicit_isentropic import ExplicitIsentropic
import pycycle.cea.properties as properties
from pycycle.cea.thermo_lookup import ThermoLookup 
from pycycle.cea.pressure_solve import PressureSolve
from pycycle.cea.T_lookup import TLookup
from pycycle.cea.T_MN_resid import TmnCalc
from pycycle.cea.set_output_data import SetOutputData, Hackery


class SetTotal(om.Group):

    def initialize(self):
        self.options.declare('thermo_data', desc='thermodynamic data set', recordable=False) #not used, just here as a hack to make calls happy
        self.options.declare('init_reacts') #not used, just here as a hack to make calls happy
        self.options.declare('gamma', default=1.4, desc='ratio of specific heats')
        self.options.declare('for_statics', default=False, values=(False, 'Ps', 'MN', 'area'), desc='type of static calculation to perform')
        self.options.declare('mode', default='T', values=('T', 'h', 'S'), desc='mode to calculate in')
        self.options.declare('fl_name', default="flow", desc='flowstation name of the output flow variables')
        self.options.declare('MW', default=28.2, desc='molecular weight of gas in units of g/mol')


    def setup(self):

        # thermo_data = self.options['thermo_data']
        fl_name = self.options['fl_name']
        mode = self.options['mode']
        for_statics = self.options['for_statics']
        gamma = self.options['gamma']
        MW = self.options['MW']

        ### Get specific gas constant value ###

        self.set_input_defaults('MW.MW', MW, units='g/mol')

        if for_statics:
            self.add_subsystem('MW', om.ExecComp('R = R_bar/MW', MW={'units':'g/mol'}, R={'units':'J/(kg*degK)'}, R_bar={'units':'J/(mol*degK)', 'value':8314.4598}), 
                        promotes_outputs=('R',))

        else:
            self.add_subsystem('MW', om.ExecComp('R = R_bar/MW', MW={'units':'g/mol'}, R={'units':'J/(kg*degK)'}, R_bar={'units':'J/(mol*degK)', 'value':8314.4598}), 
                        promotes_outputs=('R',))

        ### Get temperature values when not in mode T ###

        if for_statics is False:
            if mode == 'h':
                self.add_subsystem('T_val', TLookup(mode=mode, data=properties.AIR_MIX_enthalpy), 
                    promotes_inputs=('h',), promotes_outputs=('T',))

            elif mode == 'S':
                self.add_subsystem('T_val', TLookup(mode=mode, data=properties.AIR_MIX_entropy),
                    promotes_inputs=('S', 'P'), promotes_outputs=('T',))

        elif for_statics == 'Ps':
            self.add_subsystem('Tt_val', TLookup(mode='h', data=properties.AIR_MIX_enthalpy), 
                promotes_inputs=(('h', 'ht'),), promotes_outputs=(('T', 'Tt'),))

            if mode == 'h':
                self.add_subsystem('T_val', TLookup(mode=mode, data=properties.AIR_MIX_enthalpy), 
                    promotes_inputs=('h',), promotes_outputs=('T',))

            elif mode == 'S':
                self.add_subsystem('T_val', TLookup(mode=mode, data=properties.AIR_MIX_entropy),
                    promotes_inputs=('S', 'P'), promotes_outputs=('T',))

        elif for_statics == 'MN':
            self.add_subsystem('Tt_val', TLookup(mode='h', data=properties.AIR_MIX_enthalpy), 
                promotes_inputs=(('h', 'ht'),), promotes_outputs=(('T', 'Tt'),))

            ### Temporary gamma hack, fix this ###
            self.set_input_defaults('gamma', gamma, units=None)

            if mode == 'h':
                self.add_subsystem('T_val', TLookup(mode=mode, data=properties.AIR_MIX_enthalpy), 
                    promotes_inputs=('h',), promotes_outputs=('T',))

            elif mode == 'S':
                ### temporary hack, fix this ###
                self.add_subsystem('T_val', om.ExecComp('T = Tt*(1 + (gamma - 1)/2 * MN**2)**(-1)', T={'units':'degK'}, Tt={'units':'degK'}, gamma={'units':None}, MN={'units':None}), 
                    promotes_inputs=('Tt', 'gamma', 'MN'), promotes_outputs=('T',))

        else:
            self.add_subsystem('Tt_val', TLookup(mode='h', data=properties.AIR_MIX_enthalpy), 
                promotes_inputs=(('h', 'ht'),), promotes_outputs=(('T', 'Tt'),))
            if mode == 'h':
                self.add_subsystem('T_val', TLookup(mode=mode, data=properties.AIR_MIX_enthalpy), 
                    promotes_inputs=('h',), promotes_outputs=('T',))

            elif mode == 'S':
                self.add_subsystem('T_MN_val', TmnCalc(h_data=properties.AIR_MIX_enthalpy), 
                    promotes_inputs=('ht', 'gamma', 'R', 'Tt'), promotes_outputs=('T',))


        ### Add implicit pressure solves ###

        if for_statics == 'MN':
            if mode == 'T' or mode == 'h':
                
                inputs = ('T', 'Tt', 'gamma', 'MN')
                outputs = ('P', 'Pt', 'S')

            else:
                inputs = (('S_desired', 'S'), 'T')
                outputs = ('P',)

            self.add_subsystem('MN_pressure', PressureSolve(mode=mode, S_data=properties.AIR_MIX_entropy), 
                promotes_inputs=inputs, promotes_outputs=outputs)

        elif for_statics == 'area':
            if mode == 'S':
                self.add_subsystem('area_pressure', PressureSolve(mode=mode, S_data=properties.AIR_MIX_entropy),
                    promotes_inputs=('T', ('S_desired', 'S')), promotes_outputs=('P',))

        ### Add table lookups ###

        if mode == 'T':
            self.add_subsystem('lookup_data', ThermoLookup(Cp=True, h=True, S=True,
                Cp_data=properties.AIR_MIX_Cp, h_data=properties.AIR_MIX_enthalpy, S_data=properties.AIR_MIX_entropy),
                promotes_inputs=('P', 'T'), promotes_outputs=('Cp', 'h', 'S'))
            
        elif mode == 'h':
            self.add_subsystem('lookup_data', ThermoLookup(Cp=True, S=True,
                Cp_data=properties.AIR_MIX_Cp, S_data=properties.AIR_MIX_entropy),
                promotes_inputs=('P', 'T'), promotes_outputs=('Cp', 'S'))

        else:
            self.add_subsystem('lookup_data', ThermoLookup(Cp=True, h=True,
                Cp_data=properties.AIR_MIX_Cp, h_data=properties.AIR_MIX_enthalpy),
                promotes_inputs=('T',), promotes_outputs=('Cp', 'h'))
            
            
        ### Set up variables for explicit calculations ###

        if for_statics:  
            in_vars = ('T', 'R', 'Cp')
            out_vars = ('Cv', 'rho')

            if for_statics == 'Ps':
                in_vars += ('P', 'W', 'Tt')
                out_vars += ('Vsonic', 'V', 'area', 'MN')
                
            elif for_statics == 'MN':
                in_vars += ('P', 'W', 'MN')
                out_vars += ('Vsonic', 'V', 'area')
                
            else:
                if mode == 'T' or mode == 'h':
                    in_vars += ('W', 'Tt', 'area')
                    out_vars += ('Vsonic', 'V', 'MN', 'P')

                else:
                    in_vars += ('P', 'W', 'area')
                    out_vars += ('Vsonic', 'V', 'MN')
                
        else:
            in_vars = ('P', 'T', 'R', 'Cp')
            out_vars = ('Cv', 'rho')

        ### Add explicit calculations ###

        self.add_subsystem('explicit', ExplicitIsentropic(gamma=gamma, for_statics=for_statics, mode=mode, fl_name=fl_name),
                           promotes_inputs=in_vars,
                           promotes_outputs=out_vars)

        if for_statics == 'area':
            if mode == 'T' or mode == 'h':

                self.add_subsystem('entropy_lookup', ThermoLookup(S=True,
                    S_data=properties.AIR_MIX_entropy), promotes_inputs=('P', 'T'), promotes_outputs=('S',))

        if not for_statics:
            ### temporary hack, fix this ###
            self.set_input_defaults('gamma', gamma, units=None)
            self.set_input_defaults('b0', -1, units=None)
            self.set_input_defaults('n', -1, units=None)

            self.add_subsystem('flow', SetOutputData(fl_name=fl_name),
                               promotes_inputs=('T', 'P', 'h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R', 'b0', 'n'),
                               promotes_outputs=('{}:*'.format(fl_name),))

        else:
            ### temporary gamma hack, fix this ###
            self.set_input_defaults('temp_hack.alpha', gamma, units=None)
            self.add_subsystem('temp_hack', Hackery(),
                                promotes_inputs=('b0', 'P'), promotes_outputs=('n', 'n_moles', 'Ps', 'gamma'))


    def configure(self):

        for_statics = self.options['for_statics']
        if for_statics and for_statics != 'Ps':

            # statics need an newton solver to converge the outer loop with Ps
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

    import scipy

    from pycycle.cea import species_data
    from pycycle import constants
    import numpy as np

    prob = om.Problem()
    prob.model = SetTotal(for_statics='Ps', mode='T', MW=28.9651784)


    prob.model.set_input_defaults('P', 1.013, units="bar")
    # prob.model.set_input_defaults('h', 7, units='cal/g')
    # prob.model.set_input_defaults('S', 1.65, units='cal/(g*degK)')

    # prob.model.set_input_defaults('MN', .6, units=None)

    prob.model.set_input_defaults('T', 330, units='degK')
    # prob.model.set_input_defaults('ht', 10, units='cal/g')
    # prob.model.set_input_defaults('W', 15, units='lbm/s')
    # prob.model.set_input_defaults('area', .5, units='m**2')

    prob.setup(force_alloc_complex=True)
    # prob.set_val('flow:h', -0.59153318, units='cal/g')
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)
    # print(prob['T'])
    # print(prob['P'])
    # print(prob['S'])

    # prob.model.list_inputs(units=True)
    # prob.model.list_outputs(units=True)

    # print(prob.get_val('flow:T', units='degK'))
    # print(prob.get_val('flow:P', units='bar'))
    # print(prob.get_val('flow:h', units='cal/g'))
    # print(prob.get_val('S', units='cal/(g*degK)'))
    # print(prob.get_val('flow:gamma'))
    # print(prob.get_val('Cp', units='cal/(g*degK)'))
    # print(prob.get_val('Cv', units='cal/(g*degK)'))
    # print(prob.get_val('flow:rho', units='lbm/ft**3'))
    # print(prob.get_val('R', units='cal/(g*degK)'))
    # print(prob.get_val('V', units='ft/s'))
    # print(prob.get_val('Vsonic', units='ft/s'))
    # print(prob.get_val('area', units='m**2'))
    # print(prob.get_val('MN', units=None))
    # print()
    # print(prob.get_val('W', units='lbm/s'))
    # print(prob.get_val('T', units='degK'))
    # print(prob.get_val('P', units='bar'))
    # print(prob.get_val('ht', units='cal/g'))
    # print(prob.get_val('S', units='cal/(g*degK)'))
    # print(prob.get_val('MN', units=None))
    # print(prob.get_val('area', units='m**2'))
    # print(prob.get_val('h', units='cal/g'))
    # print(prob.get_val('ht', units='cal/g'))

    prob.model.list_inputs(prom_name=True, hierarchical=False)
    prob.model.list_outputs(prom_name=True, hierarchical=False)