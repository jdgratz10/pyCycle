import sys

import openmdao.api as om

import pycycle.api as pyc

class Turboshaft(om.Group):

    def initialize(self):
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')

    def setup(self):

        thermo_spec = pyc.species_data.janaf
        design = self.options['design']

        # Add engine elements
        self.add_subsystem('fc', pyc.FlightConditions(thermo_data=thermo_spec,
                                    elements=pyc.AIR_MIX))
        self.add_subsystem('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec,
                                    elements=pyc.AIR_MIX))
        self.add_subsystem('duct1', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('lpc', pyc.Compressor(map_data=pyc.LPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX),
                           promotes_inputs=[('Nmech','IP_Nmech')])
        self.add_subsystem('icduct', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('hpc_axi', pyc.Compressor(map_data=pyc.AXI5, design=design,
                                    thermo_data=thermo_spec, elements=pyc.AIR_MIX, map_extrap=True),
                                    promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('bld25', pyc.BleedOut(design=design, bleed_names=['cool1','cool2']))
        self.add_subsystem('hpc_centri', pyc.Compressor(map_data=pyc.HPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('bld3', pyc.BleedOut(design=design, bleed_names=['cool3','cool4']))
        self.add_subsystem('duct6', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('burner', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                    inflow_elements=pyc.AIR_MIX,
                                    air_fuel_elements=pyc.AIR_FUEL_MIX,
                                    fuel_type='JP-7'))
        self.add_subsystem('hpt', pyc.Turbine(map_data=pyc.LPT2269, design=design,
                                    thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX, bleed_names=['cool3','cool4']),
                                    promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('lpt', pyc.Turbine(map_data=pyc.LPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                            bleed_names=['cool1','cool2']),
                           promotes_inputs=[('Nmech','IP_Nmech')])
        self.add_subsystem('pt', pyc.Turbine(map_data=pyc.LPT2269, design=design,
                                    thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX, map_extrap=True),
                                    promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv',
                                    thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('LP_shaft', pyc.Shaft(num_ports=1),promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('ip_shaft', pyc.Shaft(num_ports=2),promotes_inputs=[('Nmech','IP_Nmech')])
        self.add_subsystem('HP_shaft', pyc.Shaft(num_ports=3),promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        # Connect outputs to perfomance element
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc_axi.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')
        self.connect('LP_shaft.pwr_in', 'perf.power')

        # Connect turbomachinery elements to shaft
        self.connect('hpc_axi.trq', 'HP_shaft.trq_0')
        self.connect('hpt.trq', 'HP_shaft.trq_1')
        self.connect('hpc_centri.trq', 'HP_shaft.trq_2')
        self.connect('pt.trq', 'LP_shaft.trq_0')
        self.connect('lpc.trq', 'ip_shaft.trq_0')
        self.connect('lpt.trq', 'ip_shaft.trq_1')

        # Connnect nozzle exhaust to freestream static conditions
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')

        # Add balances for design and off-design
        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:

            balance.add_balance('W', val=27.0, units='lbm/s', eq_units=None)
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozz.PR', 'balance.lhs:W')

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('lpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('ip_shaft.pwr_net', 'balance.lhs:lpt_PR')

            balance.add_balance('hpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('HP_shaft.pwr_net', 'balance.lhs:hpt_PR')

            balance.add_balance('pt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.pt_PR', 'pt.PR')
            self.connect('LP_shaft.pwr_net', 'balance.lhs:pt_PR')


        else:

            balance.add_balance('FAR', eq_units='hp', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('LP_shaft.pwr_net', 'balance.lhs:FAR')

            balance.add_balance('W', units='lbm/s', eq_units='inch**2')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('IP_Nmech', val=12000.0, units='rpm', lower=1.001, eq_units='hp', rhs_val=0.)
            self.connect('balance.IP_Nmech', 'IP_Nmech')
            self.connect('ip_shaft.pwr_net', 'balance.lhs:IP_Nmech')

            balance.add_balance('HP_Nmech', val=14800.0, units='rpm', lower=1.001, eq_units='hp', rhs_val=0.)
            self.connect('balance.HP_Nmech', 'HP_Nmech')
            self.connect('HP_shaft.pwr_net', 'balance.lhs:HP_Nmech')

        # Connect flow stations
        pyc.connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        pyc.connect_flow(self, 'inlet.Fl_O', 'duct1.Fl_I')
        pyc.connect_flow(self, 'duct1.Fl_O', 'lpc.Fl_I')
        pyc.connect_flow(self, 'lpc.Fl_O', 'icduct.Fl_I')
        pyc.connect_flow(self, 'icduct.Fl_O', 'hpc_axi.Fl_I')
        pyc.connect_flow(self, 'hpc_axi.Fl_O', 'bld25.Fl_I')
        pyc.connect_flow(self, 'bld25.Fl_O', 'hpc_centri.Fl_I')
        pyc.connect_flow(self, 'hpc_centri.Fl_O', 'bld3.Fl_I')
        pyc.connect_flow(self, 'bld3.Fl_O', 'duct6.Fl_I')
        pyc.connect_flow(self, 'duct6.Fl_O', 'burner.Fl_I')
        pyc.connect_flow(self, 'burner.Fl_O', 'hpt.Fl_I')
        pyc.connect_flow(self, 'hpt.Fl_O', 'lpt.Fl_I')
        pyc.connect_flow(self, 'lpt.Fl_O', 'pt.Fl_I')
        pyc.connect_flow(self, 'pt.Fl_O', 'nozz.Fl_I')

        pyc.connect_flow(self, 'bld25.cool1', 'lpt.cool1', connect_stat=False)
        pyc.connect_flow(self, 'bld25.cool2', 'lpt.cool2', connect_stat=False)
        pyc.connect_flow(self, 'bld3.cool3', 'hpt.cool3', connect_stat=False)
        pyc.connect_flow(self, 'bld3.cool4', 'hpt.cool4', connect_stat=False)

        # Setup solver to converge engine
        # self.set_order(['balance', 'fc', 'inlet', 'lpc', 'hpc_axi', 'hpc_centri', 'burner', 'hpt', 'lpt', 'pt', 'nozz', 'HP_shaft', 'ip_shaft', 'LP_shaft', 'perf'])

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 70
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False

        newton.linesearch = om.BoundsEnforceLS()
        # newton.linesearch = om.ArmijoGoldsteinLS()
        # newton.linesearch.options['c'] = .0001
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1

        self.linear_solver = om.DirectSolver(assemble_jac=True)

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


    fs_names = ['fc.Fl_O','inlet.Fl_O','duct1.Fl_O', 'lpc.Fl_O', 'icduct.Fl_O', 
                'hpc_axi.Fl_O', 'bld25.Fl_O',
                'hpc_centri.Fl_O', 'bld3.Fl_O', 'duct6.Fl_O'
                'burner.Fl_O','hpt.Fl_O','lpt.Fl_O', 'pt.Fl_O',
                'nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ['lpc', 'hpc_axi', 'hpc_centri']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    pyc.print_burner(prob, [f'{pt}.burner'])

    turb_names = ['hpt', 'lpt', 'pt']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ['nozz']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ['HP_shaft','ip_shaft', 'LP_shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)

    bleed_names = ['bld25', 'bld3']
    bleed_full_names = [f'{pt}.{b}' for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)


if __name__ == "__main__":

    import time

    prob = om.Problem()

    des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])

    des_vars.add_output('DESIGNfc_alt', 28000., units='ft')
    des_vars.add_output('DESIGNfc_MN', 0.5)
    des_vars.add_output('DESIGNbalance_rhs_FAR', 2740.0, units='degR')
    des_vars.add_output('DESIGNLP_shaft_HPX', 1800, units='hp')##################
    des_vars.add_output('DESIGNbalance_rhs_W', 1.1)
    des_vars.add_output('DESIGNinlet_ram_recovery', 1.0)

    des_vars.add_output('DESIGNduct1_dPqP', 0.0),
    des_vars.add_output('DESIGNduct1_MN_out', 0.4),

    des_vars.add_output('DESIGNlpc_PR', 5)
    des_vars.add_output('DESIGNlpc_eff', .89)
    des_vars.add_output('DESIGNlpc_MN', .3)
    des_vars.add_output('DESIGNicduct_dPqP', 0.002),
    des_vars.add_output('DESIGNicduct_MN_out', 0.3),
    des_vars.add_output('DESIGNhpc_axi_PR', 3)
    des_vars.add_output('DESIGNhpc_axi_eff', 0.89)
    des_vars.add_output('DESIGNbld25_cool1_frac_W', 0.024),
    des_vars.add_output('DESIGNbld25_cool2_frac_W', 0.0146),
    des_vars.add_output('DESIGNbld25_MN_out', 0.3000),
    des_vars.add_output('DESIGNhpc_centri_PR', 2.7)
    des_vars.add_output('DESIGNhpc_centri_eff', .88)
    des_vars.add_output('DESIGNhpc_centri_MN', .2)
    des_vars.add_output('DESIGNbld3_cool3_frac_W', 0.1705),
    des_vars.add_output('DESIGNbld3_cool4_frac_W', 0.1209),
    des_vars.add_output('DESIGNbld3_MN_out', 0.2000),
    des_vars.add_output('DESIGNduct6_dPqP', 0.00),
    des_vars.add_output('DESIGNduct6_MN_out', 0.2000),
    des_vars.add_output('DESIGNburner_dPqP', 0.05)
    des_vars.add_output('DESIGNhpt_eff', 0.89)
    des_vars.add_output('DESIGNhpt_cool3_frac_P', 1.0),
    des_vars.add_output('DESIGNhpt_cool4_frac_P', 0.0),
    des_vars.add_output('DESIGNpt_eff', 0.9)
    des_vars.add_output('DESIGNlpt_eff', .9)
    des_vars.add_output('DESIGNlpt_cool1_frac_P', 1.0),
    des_vars.add_output('DESIGNlpt_cool2_frac_P', 0.0),
    des_vars.add_output('DESIGNnozz_Cv', 0.99)
    des_vars.add_output('DESIGNHP_Nmech', 14800, units='rpm')
    des_vars.add_output('DESIGNIP_Nmech', 12000, units='rpm')
    des_vars.add_output('DESIGNLP_Nmech', 12750, units='rpm')

    des_vars.add_output('DESIGNinlet_MN', 0.40)
    des_vars.add_output('DESIGNhpc_axi_MN', 0.30)
    des_vars.add_output('DESIGNburner_MN', 0.15)
    des_vars.add_output('DESIGNhpt_MN', 0.3)
    des_vars.add_output('DESIGNlpt_MN', .4)
    des_vars.add_output('DESIGNpt_MN', .4)

    des_vars.add_output('OD1_fc_alt', 28000, units='ft')
    des_vars.add_output('OD1_fc_MN', .5)
    des_vars.add_output('OD1_LP_Nmech', 12750.0, units='rpm')
    des_vars.add_output('OD1_balance_rhs_FAR', 1600.0, units='hp')

    # Create design instance of model


    prob.model.connect('OD1_fc_alt', 'OD1.fc.alt')
    prob.model.connect('OD1_fc_MN', 'OD1.fc.MN')
    prob.model.connect('OD1_LP_Nmech', 'OD1.LP_Nmech')
    prob.model.connect('OD1_balance_rhs_FAR', 'OD1.balance.rhs:FAR')




    prob.model.add_subsystem('DESIGN', Turboshaft())


    prob.model.connect('DESIGNfc_alt', 'DESIGN.fc.alt')
    prob.model.connect('DESIGNfc_MN', 'DESIGN.fc.MN')
    prob.model.connect('DESIGNbalance_rhs_FAR', 'DESIGN.balance.rhs:FAR')
    prob.model.connect('DESIGNLP_shaft_HPX', 'DESIGN.LP_shaft.HPX')
    prob.model.connect('DESIGNbalance_rhs_W', 'DESIGN.balance.rhs:W')
    prob.model.connect('DESIGNinlet_ram_recovery', 'DESIGN.inlet.ram_recovery')

    prob.model.connect('DESIGNduct1_dPqP', 'DESIGN.duct1.dPqP')
    prob.model.connect('DESIGNduct1_MN_out', 'DESIGN.duct1.MN')

    prob.model.connect('DESIGNbld25_cool1_frac_W', 'OD1.bld25.cool1:frac_W')
    prob.model.connect('DESIGNbld25_cool2_frac_W', 'OD1.bld25.cool2:frac_W')
    prob.model.connect('DESIGNbld3_cool3_frac_W', 'OD1.bld3.cool3:frac_W')
    prob.model.connect('DESIGNbld3_cool4_frac_W', 'OD1.bld3.cool4:frac_W')
    prob.model.connect('DESIGNlpt_cool1_frac_P', 'DESIGN.lpt.cool1:frac_P')
    prob.model.connect('DESIGNlpt_cool2_frac_P', 'DESIGN.lpt.cool2:frac_P')
    prob.model.connect('DESIGNlpt_cool1_frac_P', 'OD1.lpt.cool1:frac_P')
    prob.model.connect('DESIGNlpt_cool2_frac_P', 'OD1.lpt.cool2:frac_P')
    prob.model.connect('DESIGNhpt_cool3_frac_P', 'OD1.hpt.cool3:frac_P')
    prob.model.connect('DESIGNhpt_cool4_frac_P', 'OD1.hpt.cool4:frac_P')

    prob.model.connect('DESIGNlpc_PR', 'DESIGN.lpc.PR')
    prob.model.connect('DESIGNlpc_eff', 'DESIGN.lpc.eff')
    prob.model.connect('DESIGNlpc_MN', 'DESIGN.lpc.MN')
    prob.model.connect('DESIGNicduct_dPqP', 'DESIGN.icduct.dPqP')
    prob.model.connect('DESIGNicduct_MN_out', 'DESIGN.icduct.MN')
    prob.model.connect('DESIGNhpc_axi_PR', 'DESIGN.hpc_axi.PR')
    prob.model.connect('DESIGNhpc_axi_eff', 'DESIGN.hpc_axi.eff')
    prob.model.connect('DESIGNbld25_cool1_frac_W', 'DESIGN.bld25.cool1:frac_W')
    prob.model.connect('DESIGNbld25_cool2_frac_W', 'DESIGN.bld25.cool2:frac_W')
    prob.model.connect('DESIGNbld25_MN_out', 'DESIGN.bld25.MN')
    prob.model.connect('DESIGNhpc_centri_PR', 'DESIGN.hpc_centri.PR')
    prob.model.connect('DESIGNhpc_centri_eff', 'DESIGN.hpc_centri.eff')
    prob.model.connect('DESIGNhpc_centri_MN', 'DESIGN.hpc_centri.MN')
    prob.model.connect('DESIGNbld3_cool3_frac_W', 'DESIGN.bld3.cool3:frac_W')
    prob.model.connect('DESIGNbld3_cool4_frac_W', 'DESIGN.bld3.cool4:frac_W')
    prob.model.connect('DESIGNbld3_MN_out', 'DESIGN.bld3.MN')
    prob.model.connect('DESIGNduct6_dPqP', 'DESIGN.duct6.dPqP')
    prob.model.connect('DESIGNduct6_MN_out', 'DESIGN.duct6.MN')
    prob.model.connect('DESIGNburner_dPqP', 'DESIGN.burner.dPqP')
    prob.model.connect('DESIGNhpt_eff', 'DESIGN.hpt.eff')
    prob.model.connect('DESIGNhpt_cool3_frac_P', 'DESIGN.hpt.cool3:frac_P')
    prob.model.connect('DESIGNhpt_cool4_frac_P', 'DESIGN.hpt.cool4:frac_P')
    prob.model.connect('DESIGNpt_eff', 'DESIGN.pt.eff')
    prob.model.connect('DESIGNlpt_eff', 'DESIGN.lpt.eff')
    prob.model.connect('DESIGNnozz_Cv', 'DESIGN.nozz.Cv')
    prob.model.connect('DESIGNHP_Nmech', 'DESIGN.HP_Nmech')
    prob.model.connect('DESIGNIP_Nmech', 'DESIGN.IP_Nmech')
    prob.model.connect('DESIGNLP_Nmech', 'DESIGN.LP_Nmech')

    prob.model.connect('DESIGNinlet_MN', 'DESIGN.inlet.MN')
    prob.model.connect('DESIGNhpc_axi_MN', 'DESIGN.hpc_axi.MN')
    prob.model.connect('DESIGNburner_MN', 'DESIGN.burner.MN')
    prob.model.connect('DESIGNhpt_MN', 'DESIGN.hpt.MN')
    prob.model.connect('DESIGNlpt_MN', 'DESIGN.lpt.MN')
    prob.model.connect('DESIGNpt_MN', 'DESIGN.pt.MN')

    # Connect off-design and required design inputs to model
    od_pts = ['OD1']
    od_MNs = [.5,]
    od_alts =[28000,]
    od_pwrs =[1600,]
    od_nmechs =[12750,]

    for pt in od_pts:
        prob.model.add_subsystem(pt, Turboshaft(design=False))

    prob.model.connect('DESIGNduct1_dPqP', 'OD1.duct1.dPqP')

    prob.model.connect('DESIGN.lpc.s_PR', 'OD1.lpc.s_PR')
    prob.model.connect('DESIGN.lpc.s_Wc', 'OD1.lpc.s_Wc')
    prob.model.connect('DESIGN.lpc.s_eff', 'OD1.lpc.s_eff')
    prob.model.connect('DESIGN.lpc.s_Nc', 'OD1.lpc.s_Nc')

    prob.model.connect('DESIGN.hpc_axi.s_PR', 'OD1.hpc_axi.s_PR')
    prob.model.connect('DESIGN.hpc_axi.s_Wc', 'OD1.hpc_axi.s_Wc')
    prob.model.connect('DESIGN.hpc_axi.s_eff', 'OD1.hpc_axi.s_eff')
    prob.model.connect('DESIGN.hpc_axi.s_Nc', 'OD1.hpc_axi.s_Nc')

    prob.model.connect('DESIGN.hpc_centri.s_PR', 'OD1.hpc_centri.s_PR')
    prob.model.connect('DESIGN.hpc_centri.s_Wc', 'OD1.hpc_centri.s_Wc')
    prob.model.connect('DESIGN.hpc_centri.s_eff', 'OD1.hpc_centri.s_eff')
    prob.model.connect('DESIGN.hpc_centri.s_Nc', 'OD1.hpc_centri.s_Nc')

    prob.model.connect('DESIGN.hpt.s_PR', 'OD1.hpt.s_PR')
    prob.model.connect('DESIGN.hpt.s_Wp', 'OD1.hpt.s_Wp')
    prob.model.connect('DESIGN.hpt.s_eff', 'OD1.hpt.s_eff')
    prob.model.connect('DESIGN.hpt.s_Np', 'OD1.hpt.s_Np')

    prob.model.connect('DESIGNicduct_dPqP', 'OD1.icduct.dPqP')
    prob.model.connect('DESIGNduct6_dPqP', 'OD1.duct6.dPqP')

    prob.model.connect('DESIGN.pt.s_PR', 'OD1.pt.s_PR')
    prob.model.connect('DESIGN.pt.s_Wp', 'OD1.pt.s_Wp')
    prob.model.connect('DESIGN.pt.s_eff', 'OD1.pt.s_eff')
    prob.model.connect('DESIGN.pt.s_Np', 'OD1.pt.s_Np')

    prob.model.connect('DESIGN.lpt.s_PR', 'OD1.lpt.s_PR')
    prob.model.connect('DESIGN.lpt.s_Wp', 'OD1.lpt.s_Wp')
    prob.model.connect('DESIGN.lpt.s_eff', 'OD1.lpt.s_eff')
    prob.model.connect('DESIGN.lpt.s_Np', 'OD1.lpt.s_Np')

    prob.model.connect('DESIGN.inlet.Fl_O:stat:area', 'OD1.inlet.area')
    prob.model.connect('DESIGN.duct1.Fl_O:stat:area', 'OD1.duct1.area')
    prob.model.connect('DESIGN.lpc.Fl_O:stat:area', 'OD1.lpc.area')
    prob.model.connect('DESIGN.icduct.Fl_O:stat:area', 'OD1.icduct.area')
    prob.model.connect('DESIGN.hpc_axi.Fl_O:stat:area', 'OD1.hpc_axi.area')
    prob.model.connect('DESIGN.bld25.Fl_O:stat:area', 'OD1.bld25.area')
    prob.model.connect('DESIGN.hpc_centri.Fl_O:stat:area', 'OD1.hpc_centri.area')
    prob.model.connect('DESIGN.bld3.Fl_O:stat:area', 'OD1.bld3.area')
    prob.model.connect('DESIGN.burner.Fl_O:stat:area', 'OD1.burner.area')
    prob.model.connect('DESIGN.hpt.Fl_O:stat:area', 'OD1.hpt.area')
    prob.model.connect('DESIGN.pt.Fl_O:stat:area', 'OD1.pt.area')
    prob.model.connect('DESIGN.lpt.Fl_O:stat:area', 'OD1.lpt.area')

    # prob.model.pyc_connect_des_od(p, 'balance.rhs:FAR')
    # prob.model.pyc_connect_des_od(p', 'balance.rhs:FAR')
    prob.model.connect('DESIGN.nozz.Throat:stat:area', 'OD1.balance.rhs:W')


    prob.setup(check=False)



    # Connect design point inputs to model
    


    # Set initial guesses for balances
    # prob['DESIGN.balance.FAR'] = 0.02261
    # prob['DESIGN.balance.W'] = 6.363
    # prob['DESIGN.balance.hpt_PR'] = 4
    # prob['DESIGN.balance.lpt_PR'] = 3
    # prob['DESIGN.balance.pt_PR'] = 4
    # prob['DESIGN.fc.balance.Pt'] = 5.7
    # prob['DESIGN.fc.balance.Tt'] = 440

    prob['DESIGN.balance.FAR'] = 0.02261
    prob['DESIGN.balance.W'] = 10.76
    prob['DESIGN.balance.hpt_PR'] = 4.233
    prob['DESIGN.balance.lpt_PR'] = 1.979
    prob['DESIGN.balance.pt_PR'] = 4.919
    prob['DESIGN.fc.balance.Pt'] = 5.666
    prob['DESIGN.fc.balance.Tt'] = 440.0

    for i,pt in enumerate(od_pts):

        # prob[pt+'.burner.dPqP'] = 0.03
        # prob[pt+'.nozz.Cv'] = 0.99

        # prob.set_val(pt+'.fc.alt', od_alts[i], units='ft')
        # prob.set_val(pt+'.fc.MN', od_MNs[i])
        # prob.set_val(pt+'.LP_Nmech', od_nmechs[i], units='rpm')
        # prob.set_val(pt+'.balance.rhs:FAR', od_pwrs[i], units='hp')

        # prob[pt+'.balance.W'] = 6.035
        # prob[pt+'.balance.FAR'] = 0.02135
        # prob[pt+'.balance.HP_Nmech'] = 14800
        # prob[pt+'.fc.balance.Pt'] = 5.7
        # prob[pt+'.fc.balance.Tt'] = 440
        # prob[pt+'.hpt.PR'] = 4
        # prob[pt+'.lpt.PR'] = 1.979
        # prob[pt+'.pt.PR'] = 5
        prob[pt+'.balance.IP_Nmech'] = 12000.000
        prob[pt+'.nozz.PR'] = 1.1
        prob[pt+'.balance.W'] = 10.775
        prob[pt+'.balance.FAR'] = 0.02135
        prob[pt+'.balance.HP_Nmech'] = 14800.
        prob[pt+'.fc.balance.Pt'] = 5.666
        prob[pt+'.fc.balance.Tt'] = 440.
        prob[pt+'.hpt.PR'] = 4.233
        prob[pt+'.lpt.PR'] = 1.979
        prob[pt+'.pt.PR'] = 4.919

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    for pt in ['DESIGN']+od_pts:
        viewer(prob, pt)

    print()
    print("time", time.time() - st)