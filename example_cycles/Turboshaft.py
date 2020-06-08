import sys
import numpy as np

import openmdao.api as om

import pycycle.api as pyc

class Turboshaft(pyc.Cycle):

    def initialize(self):
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')
        
    def setup(self):

        thermo_spec = pyc.species_data.janaf
        design = self.options['design']

        self.add_subsystem('fc', pyc.FlightConditions(thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('duct1', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('lpc', pyc.Compressor(map_data=pyc.LPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX),
                           promotes_inputs=[('Nmech','IP_Nmech')])
        self.add_subsystem('icduct', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('hpc_axi', pyc.Compressor(map_data=pyc.HPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('bld25', pyc.BleedOut(design=design, bleed_names=['cool1','cool2']))
        self.add_subsystem('hpc_centri', pyc.Compressor(map_data=pyc.HPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('bld3', pyc.BleedOut(design=design, bleed_names=['cool3','cool4']))
        self.add_subsystem('duct6', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('burner', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                                   inflow_elements=pyc.AIR_MIX,
                                                   air_fuel_elements=pyc.AIR_FUEL_MIX,
                                                   fuel_type='Jet-A(g)'))
        self.add_subsystem('hpt', pyc.Turbine(map_data=pyc.HPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                                              bleed_names=['cool3','cool4']),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('duct43', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('lpt', pyc.Turbine(map_data=pyc.LPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                                              bleed_names=['cool1','cool2']),
                           promotes_inputs=[('Nmech','IP_Nmech')])
        self.add_subsystem('itduct', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('pt', pyc.Turbine(map_data=pyc.LPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX),
                           promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('duct12', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('nozzle', pyc.Nozzle(nozzType='CV', lossCoef='Cv', thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))

        self.add_subsystem('lp_shaft', pyc.Shaft(num_ports=1),promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('ip_shaft', pyc.Shaft(num_ports=2),promotes_inputs=[('Nmech','IP_Nmech')])
        self.add_subsystem('hp_shaft', pyc.Shaft(num_ports=3),promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        self.connect('duct1.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc_centri.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozzle.Fg', 'perf.Fg_0')
        self.connect('lp_shaft.pwr_in', 'perf.power')

        self.connect('pt.trq', 'lp_shaft.trq_0')
        self.connect('lpc.trq', 'ip_shaft.trq_0')
        self.connect('lpt.trq', 'ip_shaft.trq_1')
        self.connect('hpc_axi.trq', 'hp_shaft.trq_0')
        self.connect('hpc_centri.trq', 'hp_shaft.trq_1')
        self.connect('hpt.trq', 'hp_shaft.trq_2')
        self.connect('fc.Fl_O:stat:P', 'nozzle.Ps_exhaust')

        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:

            balance.add_balance('W', units='lbm/s', eq_units=None, rhs_name='nozz_PR_target')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozzle.PR', 'balance.lhs:W')

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017, rhs_name='T4_target')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('lpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('ip_shaft.pwr_net', 'balance.lhs:lpt_PR')

            balance.add_balance('hpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_net', 'balance.lhs:hpt_PR')

            balance.add_balance('pt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.pt_PR', 'pt.PR')
            self.connect('lp_shaft.pwr_net', 'balance.lhs:pt_PR')


        else:
            # Need to check all these balances once power turbine map is updated
            balance.add_balance('FAR', eq_units='lbf', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('perf.Fn', 'balance.lhs:FAR')

            balance.add_balance('W', units='lbm/s', eq_units=None)
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozzle.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('IP_Nmech', val=12000.0, units='rpm', lower=1.001, eq_units='hp', rhs_val=0.)
            self.connect('balance.IP_Nmech', 'IP_Nmech')
            self.connect('ip_shaft.pwr_net', 'balance.lhs:IP_Nmech')

            balance.add_balance('HP_Nmech', val=14800.0, units='rpm', lower=1.001, eq_units='hp', rhs_val=0.)
            self.connect('balance.HP_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_net', 'balance.lhs:HP_Nmech')

            balance.add_balance('LP_Nmech', val=1800.0, units='rpm', lower=1.001, eq_units='hp', rhs_val=0.)
            self.connect('balance.LP_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_net', 'balance.lhs:LP_Nmech')

        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'duct1.Fl_I')
        self.pyc_connect_flow('duct1.Fl_O', 'lpc.Fl_I')
        self.pyc_connect_flow('lpc.Fl_O', 'icduct.Fl_I')
        self.pyc_connect_flow('icduct.Fl_O', 'hpc_axi.Fl_I')
        self.pyc_connect_flow('hpc_axi.Fl_O', 'bld25.Fl_I')
        self.pyc_connect_flow('bld25.Fl_O', 'hpc_centri.Fl_I')
        self.pyc_connect_flow('hpc_centri.Fl_O', 'bld3.Fl_I')
        self.pyc_connect_flow('bld3.Fl_O', 'duct6.Fl_I')
        self.pyc_connect_flow('duct6.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'hpt.Fl_I')
        self.pyc_connect_flow('hpt.Fl_O', 'duct43.Fl_I')
        self.pyc_connect_flow('duct43.Fl_O', 'lpt.Fl_I')
        self.pyc_connect_flow('lpt.Fl_O', 'itduct.Fl_I')
        self.pyc_connect_flow('itduct.Fl_O', 'pt.Fl_I')
        self.pyc_connect_flow('pt.Fl_O', 'duct12.Fl_I')
        self.pyc_connect_flow('duct12.Fl_O', 'nozzle.Fl_I')

        self.pyc_connect_flow('bld25.cool1', 'lpt.cool1', connect_stat=False)
        self.pyc_connect_flow('bld25.cool2', 'lpt.cool2', connect_stat=False)
        self.pyc_connect_flow('bld3.cool3', 'hpt.cool3', connect_stat=False)
        self.pyc_connect_flow('bld3.cool4', 'hpt.cool4', connect_stat=False)

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch = om.BoundsEnforceLS()
        # newton.linesearch = ArmijoGoldsteinLS()
        # newton.linesearch.options['c'] = .0001
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1

        self.linear_solver = om.DirectSolver()

def viewer(prob, pt, file=sys.stdout):
    """
    print a report of all the relevant cycle properties
    """

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     PSFC ")
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f" \
                %(prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.alt'],prob[pt+'.inlet.Fl_O:stat:W'], \
                prob[pt+'.perf.Fn'],prob[pt+'.perf.Fg'],prob[pt+'.inlet.F_ram'],prob[pt+'.perf.OPR'],prob[pt+'.perf.PSFC']))


    fs_names = ['fc.Fl_O','inlet.Fl_O','duct1.Fl_O','lpc.Fl_O',
                'icduct.Fl_O','hpc_axi.Fl_O','bld25.Fl_O',
                'hpc_centri.Fl_O','bld3.Fl_O','duct6.Fl_O',
                'burner.Fl_O','hpt.Fl_O','duct43.Fl_O','lpt.Fl_O',
                'itduct.Fl_O','pt.Fl_O','duct12.Fl_O','nozzle.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ['lpc','hpc_axi','hpc_centri']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    pyc.print_burner(prob, [f'{pt}.burner'])

    turb_names = ['hpt','lpt']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ['nozzle']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ['hp_shaft','lp_shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)

    bleed_names = ['bld25', 'bld3']
    bleed_full_names = [f'{pt}.{b}' for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)


if __name__ == "__main__":

    import time
    from openmdao.api import Problem
    from openmdao.utils.units import convert_units as cu

    prob = om.Problem()

    prob.model = pyc.MPCycle()



    # DESIGN CASE
    prob.model.pyc_add_des_pnt('DESIGN', Turboshaft(design=True))


    # OFF DESIGN 1
    # des_vars.add_output('OD1_MN', 0.5),
    # des_vars.add_output('OD1_alt', 28000.0, units='ft'),
    # # des_vars.add_output('OD1_Fn_target', 5497.0, units='lbf'),
    # des_vars.add_output('OD1_P_target', 7500.0, units='hp')

    # OFF DESIGN CASES
    pts = [] # 'OD1','OD2','OD3','OD4','OD5','OD6','OD7','OD8']

    for pt in pts:
        ODpt = prob.model.pyc_add_od_pnt(pt, Turboshaft(design=False))

    #     prob.model.connect(pt+'_alt', pt+'.fc.alt')
    #     prob.model.connect(pt+'_MN', pt+'.fc.MN')
    #     # prob.model.connect(pt+'_Fn_target', pt+'.thrust_balance.rhs')

    #     prob.model.connect('inlet:ram_recovery', pt+'.inlet.ram_recovery')
    #     prob.model.connect('duct4:dPqP', pt+'.duct4.dPqP')
    #     prob.model.connect('duct6:dPqP', pt+'.duct6.dPqP')
    #     prob.model.connect('burner:dPqP', pt+'.burner.dPqP')
    #     prob.model.connect('duct11:dPqP', pt+'.duct11.dPqP')
    #     prob.model.connect('duct13:dPqP', pt+'.duct13.dPqP')
    #     prob.model.connect('nozzle:Cv', pt+'.nozzle.Cv')
    #     prob.model.connect('duct15:dPqP', pt+'.duct15.dPqP')
    #     prob.model.connect('hp_shaft:HPX', pt+'.hp_shaft.HPX')

    #     prob.model.connect('hpc:cool1:frac_W', pt+'.hpc.cool1:frac_W')
    #     prob.model.connect('hpc:cool1:frac_P', pt+'.hpc.cool1:frac_P')
    #     prob.model.connect('hpc:cool1:frac_work', pt+'.hpc.cool1:frac_work')
    #     prob.model.connect('hpc:cool2:frac_W', pt+'.hpc.cool2:frac_W')
    #     prob.model.connect('hpc:cool2:frac_P', pt+'.hpc.cool2:frac_P')
    #     prob.model.connect('hpc:cool2:frac_work', pt+'.hpc.cool2:frac_work')
    #     prob.model.connect('bld3:cool3:frac_W', pt+'.bld3.cool3:frac_W')
    #     prob.model.connect('bld3:cool4:frac_W', pt+'.bld3.cool4:frac_W')
    #     prob.model.connect('hpc:cust:frac_W', pt+'.hpc.cust:frac_W')
    #     prob.model.connect('hpc:cust:frac_P', pt+'.hpc.cust:frac_P')
    #     prob.model.connect('hpc:cust:frac_work', pt+'.hpc.cust:frac_work')
    #     prob.model.connect('hpt:cool3:frac_P', pt+'.hpt.cool3:frac_P')
    #     prob.model.connect('hpt:cool4:frac_P', pt+'.hpt.cool4:frac_P')
    #     prob.model.connect('lpt:cool1:frac_P', pt+'.lpt.cool1:frac_P')
    #     prob.model.connect('lpt:cool2:frac_P', pt+'.lpt.cool2:frac_P')
    #     prob.model.connect('bypBld:frac_W', pt+'.byp_bld.bypBld:frac_W')

    #     prob.model.connect('DESIGN.fan.s_PRdes', pt+'.fan.s_PRdes')
    #     prob.model.connect('DESIGN.fan.s_WcDes', pt+'.fan.s_WcDes')
    #     prob.model.connect('DESIGN.fan.s_effDes', pt+'.fan.s_effDes')
    #     prob.model.connect('DESIGN.fan.s_NcDes', pt+'.fan.s_NcDes')
    #     prob.model.connect('DESIGN.lpc.s_PRdes', pt+'.lpc.s_PRdes')
    #     prob.model.connect('DESIGN.lpc.s_WcDes', pt+'.lpc.s_WcDes')
    #     prob.model.connect('DESIGN.lpc.s_effDes', pt+'.lpc.s_effDes')
    #     prob.model.connect('DESIGN.lpc.s_NcDes', pt+'.lpc.s_NcDes')
    #     prob.model.connect('DESIGN.hpc.s_PRdes', pt+'.hpc.s_PRdes')
    #     prob.model.connect('DESIGN.hpc.s_WcDes', pt+'.hpc.s_WcDes')
    #     prob.model.connect('DESIGN.hpc.s_effDes', pt+'.hpc.s_effDes')
    #     prob.model.connect('DESIGN.hpc.s_NcDes', pt+'.hpc.s_NcDes')
    #     prob.model.connect('DESIGN.hpt.s_PRdes', pt+'.hpt.s_PRdes')
    #     prob.model.connect('DESIGN.hpt.s_WpDes', pt+'.hpt.s_WpDes')
    #     prob.model.connect('DESIGN.hpt.s_effDes', pt+'.hpt.s_effDes')
    #     prob.model.connect('DESIGN.hpt.s_NpDes', pt+'.hpt.s_NpDes')
    #     prob.model.connect('DESIGN.lpt.s_PRdes', pt+'.lpt.s_PRdes')
    #     prob.model.connect('DESIGN.lpt.s_WpDes', pt+'.lpt.s_WpDes')
    #     prob.model.connect('DESIGN.lpt.s_effDes', pt+'.lpt.s_effDes')
    #     prob.model.connect('DESIGN.lpt.s_NpDes', pt+'.lpt.s_NpDes')

    #     prob.model.connect('DESIGN.nozzle.Throat:stat:area',pt+'.core_flow_balance.rhs')

    #     prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
    #     prob.model.connect('DESIGN.fan.Fl_O:stat:area', pt+'.fan.area')
    #     prob.model.connect('DESIGN.splitter.Fl_O1:stat:area', pt+'.splitter.area1')
    #     prob.model.connect('DESIGN.splitter.Fl_O2:stat:area', pt+'.splitter.area2')
    #     prob.model.connect('DESIGN.duct4.Fl_O:stat:area', pt+'.duct4.area')
    #     prob.model.connect('DESIGN.lpc.Fl_O:stat:area', pt+'.lpc.area')
    #     prob.model.connect('DESIGN.duct6.Fl_O:stat:area', pt+'.duct6.area')
    #     prob.model.connect('DESIGN.hpc.Fl_O:stat:area', pt+'.hpc.area')
    #     prob.model.connect('DESIGN.bld3.Fl_O:stat:area', pt+'.bld3.area')
    #     prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
    #     prob.model.connect('DESIGN.hpt.Fl_O:stat:area', pt+'.hpt.area')
    #     prob.model.connect('DESIGN.duct11.Fl_O:stat:area', pt+'.duct11.area')
    #     prob.model.connect('DESIGN.lpt.Fl_O:stat:area', pt+'.lpt.area')
    #     prob.model.connect('DESIGN.duct13.Fl_O:stat:area', pt+'.duct13.area')
    #     prob.model.connect('DESIGN.byp_bld.Fl_O:stat:area', pt+'.byp_bld.area')
    #     prob.model.connect('DESIGN.duct15.Fl_O:stat:area', pt+'.duct15.area')


    # cycle parameters are shared values across all points
    prob.model.pyc_set_cycle_param('inlet.ram_recovery', 1.0)

    prob.model.pyc_set_cycle_param('bld25.cool1:frac_W', 0.024)
    prob.model.pyc_set_cycle_param('bld25.cool2:frac_W', 0.0146)

    prob.model.pyc_set_cycle_param('bld3.cool3:frac_W', 0.1705)
    prob.model.pyc_set_cycle_param('bld3.cool4:frac_W', 0.1209)

    prob.model.pyc_set_cycle_param('hpt.cool3:frac_P', 1.0)
    prob.model.pyc_set_cycle_param('hpt.cool4:frac_P', 0.0)

    prob.model.pyc_set_cycle_param('lpt.cool1:frac_P', 1.0)
    prob.model.pyc_set_cycle_param('lpt.cool2:frac_P', 0.0)

    prob.model.pyc_set_cycle_param('duct1.dPqP', 0.0)
    prob.model.pyc_set_cycle_param('duct12.dPqP', 0.00)
    prob.model.pyc_set_cycle_param('icduct.dPqP', 0.002)
    prob.model.pyc_set_cycle_param('burner.dPqP', 0.50)
    prob.model.pyc_set_cycle_param('duct43.dPqP', 0.0051)
    prob.model.pyc_set_cycle_param('duct6.dPqP', 0.00)
    prob.model.pyc_set_cycle_param('itduct.dPqP', 0.00)

    prob.model.pyc_set_cycle_param('lp_shaft.HPX', 1800.0, units='hp')


    prob.setup(check=False)


    # Parameters for the DESIGN case
    prob.set_val('DESIGN.fc.alt', 28000, units='ft')
    prob.set_val('DESIGN.fc.MN', 0.5)
    prob.set_val('DESIGN.balance.T4_target', 2740.0)
    prob.set_val('DESIGN.balance.nozz_PR_target', 1.1)


    prob.set_val('DESIGN.inlet.MN', 0.4)

    prob.set_val('DESIGN.duct1.MN', 0.4)

    prob.set_val('DESIGN.lpc.PR', 5.0)
    prob.set_val('DESIGN.lpc.eff', 0.89)
    prob.set_val('DESIGN.lpc.MN', 0.3)

    prob.set_val('DESIGN.icduct.MN', 0.3)

    prob.set_val('DESIGN.hpc_axi.PR', 3.0)
    prob.set_val('DESIGN.hpc_axi.eff', 2.7)
    prob.set_val('DESIGN.hpc_axi.MN', 0.25)

    prob.set_val('DESIGN.bld25.MN', 0.3)

    prob.set_val('DESIGN.hpc_centri.PR', 2.7)
    prob.set_val('DESIGN.hpc_centri.eff', 0.88)
    prob.set_val('DESIGN.hpc_centri.MN', 0.20)

    prob.set_val('DESIGN.bld3.MN', 0.2)

    prob.set_val('DESIGN.duct6.MN', 0.20)

    prob.set_val('DESIGN.burner.MN', 0.15)

    prob.set_val('DESIGN.hpt.eff', 0.89)
    prob.set_val('DESIGN.hpt.MN', 0.30)

    prob.set_val('DESIGN.duct43.MN', 0.30)
    
    prob.set_val('DESIGN.lpt.eff', 0.9)
    prob.set_val('DESIGN.lpt.MN', 0.4)
    
    prob.set_val('DESIGN.itduct.MN', 0.4)
    
    prob.set_val('DESIGN.pt.eff', 0.85)
    prob.set_val('DESIGN.pt.MN', 0.4)
    
    prob.set_val('DESIGN.duct12.MN', 0.4)
    
    prob.set_val('DESIGN.nozzle.Cv', 0.99)
    
    prob.set_val('DESIGN.LP_Nmech', 12750, units='rpm')
    
    prob.set_val('DESIGN.IP_Nmech', 12000., units='rpm')
    
    prob.set_val('DESIGN.HP_Nmech', 14800, units='rpm')


    # initial guesses
    prob['DESIGN.balance.FAR'] = 0.02261
    prob['DESIGN.balance.W'] = 10.76
    prob['DESIGN.balance.pt_PR'] = 4.939
    prob['DESIGN.balance.lpt_PR'] = 1.979
    prob['DESIGN.balance.hpt_PR'] = 4.236
    prob['DESIGN.fc.balance.Pt'] = 5.666
    prob['DESIGN.fc.balance.Tt'] = 440.0

   
    st = time.time()


    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()
    # prob.final_setup()

    # prob.model.DESIGN.burner.list_inputs(prom_name=True)
    print(prob.model.DESIGN.burner._get_val('dPqP'))

    exit()

    for pt in ['DESIGN']+pts:
        viewer(prob, pt)

    print()
    print("time", time.time() - st)

