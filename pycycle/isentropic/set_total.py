import openmdao.api as om

# from pycycle.cea.species_data import Thermo
from pycycle.isentropic.explicit_isentropic import ExplicitIsentropic
from pycycle.isentropic.properties import PropertyMap, AIR_MIX_entropy
from pycycle.isentropic.pressure_solve import PressureSolve
from pycycle.isentropic.thermo_lookup import TempFromSP, TempFromGamma, TempFromEnthalpy, EnthalpyFromTemp
from pycycle.isentropic.T_MN_resid import TmnResid
from pycycle.isentropic.set_output_data import SetOutputData, IOMatching


class SetTotal(om.Group):

    def initialize(self):
        self.options.declare('thermo_data', desc='thermodynamic data set', recordable=False) #not used, just here to make calls happy
        self.options.declare('init_reacts') #not used, just here to make calls happy
        self.options.declare('gamma', default=1.4, desc='ratio of specific heats')
        self.options.declare('for_statics', default=False, values=(False, 'Ps', 'MN', 'area'), desc='type of static calculation to perform')
        self.options.declare('mode', default='T', values=('T', 'h', 'S'), desc='mode to calculate in')
        self.options.declare('fl_name', default="flow", desc='flowstation name of the output flow variables')
        self.options.declare('MW', default=28.2, desc='molecular weight of gas in units of g/mol')
        self.options.declare('h_base', default=0, desc='enthalpy at base temperature (units are cal/g)')
        self.options.declare('T_base', default=302.4629819, desc='base temperature (units are degK)')
        self.options.declare('Cp', default=0.24015494, desc='constant specific heat that is assumed (units are cal/(g*degK)')
        self.options.declare('S_data', default=AIR_MIX_entropy, desc='entropy property data')


    def setup(self):

        # thermo_data = self.options['thermo_data']
        for_statics = self.options['for_statics']
        fl_name = self.options['fl_name']
        S_data = self.options['S_data']
        h_base = self.options['h_base']
        T_base = self.options['T_base']
        gamma = self.options['gamma']
        mode = self.options['mode']
        MW = self.options['MW']
        Cp = self.options['Cp']

        ### Make Cp an output ###

        self.add_subsystem('option_vars', om.IndepVarComp(), promotes_outputs=['*'])
        self.option_vars.add_output('Cp', val=Cp, units='cal/(g*degK)')

        self.option_vars.add_output('gamma', val=gamma, units=None)

        ### Get specific gas constant value ###

        self.set_input_defaults('MW.MW', MW, units='g/mol')

        self.add_subsystem('MW', om.ExecComp('R = R_bar/MW', MW={'units':'g/mol'}, R={'units':'J/(kg*degK)'}, R_bar={'units':'J/(mol*degK)', 'value':8314.4598}), 
                        promotes_outputs=('R',))

        ### Get temperature values when not in mode T ###

        if for_statics is False:
            if mode == 'h':
                self.add_subsystem('T_val', TempFromEnthalpy(h_base=h_base, T_base=T_base), 
                    promotes_inputs=('h', 'Cp'), promotes_outputs=('T',))

            elif mode == 'S':
                self.add_subsystem('T_val', TempFromSP(S_data=S_data),
                    promotes_inputs=('S', 'P'), promotes_outputs=('T',))

        elif for_statics == 'Ps':
            self.add_subsystem('Tt_val', TempFromEnthalpy(h_base=h_base, T_base=T_base), 
                promotes_inputs=(('h', 'ht'), 'Cp'), promotes_outputs=(('T', 'Tt'),))

            if mode == 'h':
                self.add_subsystem('T_val', TempFromEnthalpy(h_base=h_base, T_base=T_base), 
                    promotes_inputs=('h', 'Cp'), promotes_outputs=('T',))

            elif mode == 'S':
                self.add_subsystem('T_val', TempFromSP(S_data=S_data),
                    promotes_inputs=('S', 'P'), promotes_outputs=('T',))

        elif for_statics == 'MN':
            self.add_subsystem('Tt_val', TempFromEnthalpy(h_base=h_base, T_base=T_base), 
                promotes_inputs=(('h', 'ht'), 'Cp'), promotes_outputs=(('T', 'Tt'),))

            if mode == 'h':
                self.add_subsystem('T_val', TempFromEnthalpy(h_base=h_base, T_base=T_base), 
                    promotes_inputs=('h', 'Cp'), promotes_outputs=('T',))

            elif mode == 'S':
                self.add_subsystem('T_val', TempFromGamma(),
                    promotes_inputs=('Tt', 'gamma', 'MN'), promotes_outputs=('T',))

        else:
            self.add_subsystem('Tt_val', TempFromEnthalpy(h_base=h_base, T_base=T_base), 
                promotes_inputs=(('h', 'ht'), 'Cp'), promotes_outputs=(('T', 'Tt'),))
            if mode == 'h':
                self.add_subsystem('T_val', TempFromEnthalpy(h_base=h_base, T_base=T_base), 
                    promotes_inputs=('h', 'Cp'), promotes_outputs=('T',))

            elif mode == 'S':
                self.add_subsystem('T_MN_val', TmnResid(h_base=h_base, T_base=T_base), 
                    promotes_inputs=('ht', 'gamma', 'R', 'Tt', 'Cp'), promotes_outputs=('T',))


        ### Add implicit pressure solves ###

        if for_statics == 'MN':
            if mode == 'T' or mode == 'h':
                
                inputs = ('T', 'Tt', 'gamma', 'MN')
                outputs = ('P', 'Pt', 'S')

            else:
                inputs = (('S_desired', 'S'), 'T')
                outputs = ('P',)

            self.add_subsystem('MN_pressure', PressureSolve(mode=mode, S_data=S_data), 
                promotes_inputs=inputs, promotes_outputs=outputs)

        elif for_statics == 'area':
            if mode == 'S':

                inputs = ('T', ('S_desired', 'S'))
                outputs = ('P',)

                self.add_subsystem('area_pressure', PressureSolve(mode=mode, S_data=S_data),
                    promotes_inputs=inputs, promotes_outputs=outputs)

        ### Add property lookups ###

        if mode == 'T':
            if for_statics == 'MN' or for_statics == 'area':
                self.add_subsystem('enthalpy_calc', EnthalpyFromTemp(h_base=h_base, T_base=T_base),
                    promotes_inputs=('T', 'Cp'), promotes_outputs=('h',))
            
            elif for_statics == 'Ps' or for_statics is False:
                self.add_subsystem('enthalpy_calc', EnthalpyFromTemp(h_base=h_base, T_base=T_base),
                    promotes_inputs=('T', 'Cp'), promotes_outputs=('h',))
                self.add_subsystem('entropy_lookup', PropertyMap(map_data=S_data),
                    promotes_inputs=('P', 'T'), promotes_outputs=('S',))
            
        elif mode == 'h':
            if for_statics == 'Ps' or for_statics is False:
                self.add_subsystem('entropy_lookup', PropertyMap(map_data=S_data),
                    promotes_inputs=('P', 'T'), promotes_outputs=('S',))
        else:
            self.add_subsystem('enthalpy_calc', EnthalpyFromTemp(h_base=h_base, T_base=T_base),
                promotes_inputs=('T', 'Cp'), promotes_outputs=('h',))
            
            
        ### Add explicit calculations ###

        if for_statics:  
            in_vars = ('T', 'R', 'Cp', 'gamma')
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
            in_vars = ('P', 'T', 'R', 'Cp', 'gamma')
            out_vars = ('Cv', 'rho')

        self.add_subsystem('explicit', ExplicitIsentropic(for_statics=for_statics, mode=mode, fl_name=fl_name),
                           promotes_inputs=in_vars,
                           promotes_outputs=out_vars)

        ### Find entropy value when pressure is output from explicit calcs ###

        if for_statics == 'area':
            if mode == 'T' or mode == 'h':

                self.add_subsystem('entropy_lookup', PropertyMap(map_data=S_data), 
                    promotes_inputs=('P', 'T'), promotes_outputs=('S',))

        ### Set up dummy variables and components to make I/O switching happy ###

        if for_statics is False:
            self.set_input_defaults('b0', -1, units=None)
            self.set_input_defaults('n', -1, units=None)

            self.add_subsystem('flow', SetOutputData(fl_name=fl_name),
                               promotes_inputs=('T', 'P', 'h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R', 'b0', 'n'),
                               promotes_outputs=('{}:*'.format(fl_name),))

        else: 
            self.add_subsystem('IO_matching', IOMatching(),
                                promotes_inputs=('b0', 'P'), promotes_outputs=('n', 'n_moles', 'Ps'))


if __name__ == "__main__":

    prob = om.Problem()
    prob.model = SetTotal(for_statics='area', mode='S', MW=28.9651784)


    # prob.model.set_input_defaults('P', 1.013, units="bar")
    # prob.model.set_input_defaults('h', 7, units='cal/g')
    # prob.model.set_input_defaults('S', 1.65, units='cal/(g*degK)')
    # prob.model.set_input_defaults('MN', .6, units=None)
    # prob.model.set_input_defaults('T', 330, units='degK')
    prob.model.set_input_defaults('ht', 10, units='cal/g')

    prob.set_solver_print(level=2, depth=2)

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)

    # print(prob.get_val('T', units='degK'))
    # print(prob.get_val('Tt', units='degK'))
    # print(prob.get_val('P', units='bar'))
    # print(prob.get_val('MN'))
    # # print(prob.get_val('flow:h', units='cal/g'))
    # print(prob.get_val('S', units='cal/(g*degK)'))
    # # print(prob.get_val('flow:gamma'))
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