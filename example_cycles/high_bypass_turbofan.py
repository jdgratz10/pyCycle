import sys

import numpy as np

import openmdao.api as om

import pycycle.api as pyc


class HBTF(om.Group):

    def initialize(self):
        # Initialize the model here by setting option variables such as a switch for design vs off-des cases
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')

    def setup(self):
        #Setup the problem by including all the relavant components here - comp, burner, turbine etc
        
        #Create any relavent short hands here:
        thermo_spec = pyc.species_data.janaf #Thermodynamic data specification 
        design = self.options['design']
        
        #Add subsystems to build the engine deck:
        self.add_subsystem('fc', pyc.FlightConditions(thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        
        self.add_subsystem('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        
        # Note variable promotion for the fan -- 
        # the LP spool speed and the fan speed are INPUTS that are promoted:
        # Note here that promotion aliases are used. Here Nmech is being aliased to LP_Nmech
        # in fact for a multi-spool engine you HAVE(?) to alias if you want to promote_inputs
        # check out: http://openmdao.org/twodocs/versions/latest/features/core_features/grouping_components/add_subsystem.html?highlight=alias
        self.add_subsystem('fan', pyc.Compressor(map_data=pyc.FanMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX,
                                        bleed_names=[], map_extrap=True), promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('splitter', pyc.Splitter(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('duct4', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('lpc', pyc.Compressor(map_data=pyc.LPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX,
                                        map_extrap=True),promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('duct6', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('hpc', pyc.Compressor(map_data=pyc.HPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX,
                                        bleed_names=['cool1','cool2','cust'], map_extrap=True),promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('bld3', pyc.BleedOut(design=design, bleed_names=['cool3','cool4']))
        self.add_subsystem('burner', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                        inflow_elements=pyc.AIR_MIX,
                                        air_fuel_elements=pyc.AIR_FUEL_MIX,
                                        fuel_type='Jet-A(g)'))
        self.add_subsystem('hpt', pyc.Turbine(map_data=pyc.HPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                                        bleed_names=['cool3','cool4'], map_extrap=True),promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('duct11', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('lpt', pyc.Turbine(map_data=pyc.LPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                                        bleed_names=['cool1','cool2'], map_extrap=True),promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('duct13', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('core_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv', thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))

        self.add_subsystem('byp_bld', pyc.BleedOut(design=design, bleed_names=['bypBld']))
        self.add_subsystem('duct15', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('byp_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv', thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        
        #Create shaft instances. Note that LP shaft has 3 ports! => no gearbox
        self.add_subsystem('lp_shaft', pyc.Shaft(num_ports=3),promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('hp_shaft', pyc.Shaft(num_ports=2),promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=2, num_burners=1))
    
        # Now use the explicit connect method to make connections -- connect(<from>, <to>)
        
        #Connect the inputs to perf group
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('core_nozz.Fg', 'perf.Fg_0')
        self.connect('byp_nozz.Fg', 'perf.Fg_1')
        
        #LP-shaft connections
        self.connect('fan.trq', 'lp_shaft.trq_0')
        self.connect('lpc.trq', 'lp_shaft.trq_1')
        self.connect('lpt.trq', 'lp_shaft.trq_2')
        #HP-shaft connections
        self.connect('hpc.trq', 'hp_shaft.trq_0')
        self.connect('hpt.trq', 'hp_shaft.trq_1')
        #Ideally expanding flow by conneting flight condition static pressure to nozzle exhaust pressure
        self.connect('fc.Fl_O:stat:P', 'core_nozz.Ps_exhaust')
        self.connect('fc.Fl_O:stat:P', 'byp_nozz.Ps_exhaust')
        
        #Create a balance component
        # Balances can be a bit confusing, here's some explanation -
        #   State Variables:
        #           (W)        Inlet mass flow rate to implictly balance thrust
        #                      LHS: perf.Fn  == RHS: Thrust requirement (set when TF is instantiated)
        #
        #           (FAR)      Fuel-air ratio to balance Tt4
        #                      LHS: burner.Fl_O:tot:T  == RHS: Tt4 target (set when TF is instantiated)
        #
        #           (lpt_PR)   LPT press ratio to balance shaft power on the low spool
        #           (hpt_PR)   HPT press ratio to balance shaft power on the high spool
        # Ref: look at the XDSM diagrams in the pyCycle paper and this:
        # http://openmdao.org/twodocs/versions/latest/features/building_blocks/components/balance_comp.html

        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            balance.add_balance('W', units='lbm/s', eq_units='lbf')
            #Here balance.W is implicit state variable that is the OUTPUT of balance object
            self.connect('balance.W', 'inlet.Fl_I:stat:W') #Connect the output of balance to the relevant input
            self.connect('perf.Fn', 'balance.lhs:W')       #This statement makes perf.Fn the LHS of the balance eqn.

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')
            
            # Note that for the following two balances the mult val is set to -1 so that the NET torque is zero
            balance.add_balance('lpt_PR', val=1.5, lower=1.001, upper=8,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('lp_shaft.pwr_in_real', 'balance.lhs:lpt_PR')
            self.connect('lp_shaft.pwr_out_real', 'balance.rhs:lpt_PR')

            balance.add_balance('hpt_PR', val=1.5, lower=1.001, upper=8,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_in_real', 'balance.lhs:hpt_PR')
            self.connect('hp_shaft.pwr_out_real', 'balance.rhs:hpt_PR')

        else:
            
            #In OFF-DESIGN mode we need to redefine the balances:
            #   State Variables:
            #           (W)        Inlet mass flow rate to balance core flow area
            #                      LHS: core_nozz.Throat:stat:area == Area from DESIGN calculation 
            #
            #           (FAR)      Fuel-air ratio to balance Thrust req.
            #                      LHS: perf.Fn  == RHS: Thrust requirement (set when TF is instantiated)
            #
            #           (BPR)      Bypass ratio to balance byp. noz. area
            #                      LHS: byp_nozz.Throat:stat:area == Area from DESIGN calculation
            #
            #           (lp_Nmech)   LP spool speed to balance shaft power on the low spool
            #           (hp_Nmech)   HP spool speed to balance shaft power on the high spool
            balance.add_balance('FAR', val=0.017, lower=1e-4, eq_units='lbf')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('perf.Fn', 'balance.lhs:FAR')

            balance.add_balance('W', units='lbm/s', lower=10., upper=1000., eq_units='inch**2')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('core_nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('BPR', lower=2., upper=10., eq_units='inch**2')
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('byp_nozz.Throat:stat:area', 'balance.lhs:BPR')

            # Again for the following two balances the mult val is set to -1 so that the NET torque is zero
            balance.add_balance('lp_Nmech', val=1.5, units='rpm', lower=500., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.lp_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_in_real', 'balance.lhs:lp_Nmech')
            self.connect('lp_shaft.pwr_out_real', 'balance.rhs:lp_Nmech')

            balance.add_balance('hp_Nmech', val=1.5, units='rpm', lower=500., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.hp_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_in_real', 'balance.lhs:hp_Nmech')
            self.connect('hp_shaft.pwr_out_real', 'balance.rhs:hp_Nmech')
            
            # Specify the order in which the subsystems are executed:
            
            self.set_order(['balance', 'fc', 'inlet', 'fan', 'splitter', 'duct4', 'lpc', 'duct6', 'hpc', 'bld3', 'burner', 'hpt', 'duct11',
                            'lpt', 'duct13', 'core_nozz', 'byp_bld', 'duct15', 'byp_nozz', 'lp_shaft', 'hp_shaft', 'perf'])
        
        # Set up all the flow connections:
        pyc.connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        pyc.connect_flow(self, 'inlet.Fl_O', 'fan.Fl_I')
        pyc.connect_flow(self, 'fan.Fl_O', 'splitter.Fl_I')
        pyc.connect_flow(self, 'splitter.Fl_O1', 'duct4.Fl_I')
        pyc.connect_flow(self, 'duct4.Fl_O', 'lpc.Fl_I')
        pyc.connect_flow(self, 'lpc.Fl_O', 'duct6.Fl_I')
        pyc.connect_flow(self, 'duct6.Fl_O', 'hpc.Fl_I')
        pyc.connect_flow(self, 'hpc.Fl_O', 'bld3.Fl_I')
        pyc.connect_flow(self, 'bld3.Fl_O', 'burner.Fl_I')
        pyc.connect_flow(self, 'burner.Fl_O', 'hpt.Fl_I')
        pyc.connect_flow(self, 'hpt.Fl_O', 'duct11.Fl_I')
        pyc.connect_flow(self, 'duct11.Fl_O', 'lpt.Fl_I')
        pyc.connect_flow(self, 'lpt.Fl_O', 'duct13.Fl_I')
        pyc.connect_flow(self, 'duct13.Fl_O','core_nozz.Fl_I')
        pyc.connect_flow(self, 'splitter.Fl_O2', 'byp_bld.Fl_I')
        pyc.connect_flow(self, 'byp_bld.Fl_O', 'duct15.Fl_I')
        pyc.connect_flow(self, 'duct15.Fl_O', 'byp_nozz.Fl_I')

        #Bleed flows:
        pyc.connect_flow(self, 'hpc.cool1', 'lpt.cool1', connect_stat=False)
        pyc.connect_flow(self, 'hpc.cool2', 'lpt.cool2', connect_stat=False)
        pyc.connect_flow(self, 'bld3.cool3', 'hpt.cool3', connect_stat=False)
        pyc.connect_flow(self, 'bld3.cool4', 'hpt.cool4', connect_stat=False)
        
        #Specify solver settings:
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-8
        newton.options['rtol'] = 1e-8
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 50
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        # ls = newton.linesearch = BoundsEnforceLS()
        ls = newton.linesearch = om.ArmijoGoldsteinLS()
        ls.options['maxiter'] = 3
        ls.options['bound_enforcement'] = 'scalar'
        # ls.options['print_bound_enforce'] = True

        self.linear_solver = om.DirectSolver(assemble_jac=True)


def viewer(prob, pt, file=sys.stdout):
    """
    print a report of all the relevant cycle properties
    """

    if pt == 'DESIGN':
        MN = prob['DESIGN.fc.Fl_O:stat:MN']
        LPT_PR = prob['DESIGN.balance.lpt_PR']
        HPT_PR = prob['DESIGN.balance.hpt_PR']
        FAR = prob['DESIGN.balance.FAR']
    else:
        MN = prob[pt+'.fc.Fl_O:stat:MN']
        LPT_PR = prob[pt+'.lpt.PR']
        HPT_PR = prob[pt+'.hpt.PR']
        FAR = prob[pt+'.balance.FAR']

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC      BPR ", file=file, flush=True)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f  %7.3f" %(MN, prob[pt+'.fc.alt'],prob[pt+'.inlet.Fl_O:stat:W'],prob[pt+'.perf.Fn'],prob[pt+'.perf.Fg'],prob[pt+'.inlet.F_ram'],prob[pt+'.perf.OPR'],prob[pt+'.perf.TSFC'], prob[pt+'.splitter.BPR']), file=file, flush=True)


    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'fan.Fl_O', 'splitter.Fl_O1', 'splitter.Fl_O2',
                'duct4.Fl_O', 'lpc.Fl_O', 'duct6.Fl_O', 'hpc.Fl_O', 'bld3.Fl_O', 'burner.Fl_O',
                'hpt.Fl_O', 'duct11.Fl_O', 'lpt.Fl_O', 'duct13.Fl_O', 'core_nozz.Fl_O', 'byp_bld.Fl_O',
                'duct15.Fl_O', 'byp_nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ['fan', 'lpc', 'hpc']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    pyc.print_burner(prob, [f'{pt}.burner'])

    turb_names = ['hpt', 'lpt']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ['core_nozz', 'byp_nozz']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ['hp_shaft', 'lp_shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)

    bleed_names = ['hpc', 'bld3', 'byp_bld']
    bleed_full_names = [f'{pt}.{b}' for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)

if __name__ == "__main__":

    import time

    prob = om.Problem()

    # DESIGN CASE  
    prob.model.add_subsystem('DESIGN', HBTF(), promotes=['hp_shaft.HPX']) # Create an instace of the High Bypass ratio Turbofan
    #Note that we promote hp_shaft.HPX because otherwise it's absolute name would be DESIGN.hp_shaft.HPX, which would cause a promotion mask error
    #and we would not be allowed to promote hp_shaft.HPX from the off-design cases to the name DESIGN.hp_shaft.HPX

    # OFF DESIGN CASES
    pts = ['OD1'] #,'OD2','OD3','OD4']

    for i_OD, pt in enumerate(pts):
        ODpt = prob.model.add_subsystem(pt, HBTF(design=False), promotes=[('inlet.ram_recovery', 'DESIGN.inlet.ram_recovery'), 
            ('duct4.dPqP', 'DESIGN.duct4.dPqP'), ('duct6.dPqP', 'DESIGN.duct6.dPqP'), ('burner.dPqP', 'DESIGN.burner.dPqP'),
            ('duct11.dPqP', 'DESIGN.duct11.dPqP'), ('duct13.dPqP', 'DESIGN.duct13.dPqP'), ('duct15.dPqP', 'DESIGN.duct15.dPqP'),
            ('core_nozz.Cv', 'DESIGN.core_nozz.Cv'), ('byp_bld.bypBld:frac_W', 'DESIGN.byp_bld.bypBld:frac_W'),
            ('byp_nozz.Cv', 'DESIGN.byp_nozz.Cv'), ('hpc.cool1:frac_W', 'DESIGN.hpc.cool1:frac_W'), ('hpc.cool1:frac_P', 'DESIGN.hpc.cool1:frac_P'),
            ('hpc.cool1:frac_work', 'DESIGN.hpc.cool1:frac_work'), ('hpc.cool2:frac_W', 'DESIGN.hpc.cool2:frac_W'), ('hpc.cool2:frac_P', 'DESIGN.hpc.cool2:frac_P'),
            ('hpc.cool2:frac_work', 'DESIGN.hpc.cool2:frac_work'), ('bld3.cool3:frac_W', 'DESIGN.bld3.cool3:frac_W'), ('bld3.cool4:frac_W', 'DESIGN.bld3.cool4:frac_W'),
            ('hpc.cust:frac_P', 'DESIGN.hpc.cust:frac_P'), ('hpc.cust:frac_work', 'DESIGN.hpc.cust:frac_work'), ('hpt.cool3:frac_P', 'DESIGN.hpt.cool3:frac_P'),
            ('hpt.cool4:frac_P', 'DESIGN.hpt.cool4:frac_P'), ('lpt.cool1:frac_P', 'DESIGN.lpt.cool1:frac_P'), ('lpt.cool2:frac_P', 'DESIGN.lpt.cool2:frac_P'), 'hp_shaft.HPX'])

        #Connect all DESIGN map scalars to the off design cases
        prob.model.connect('DESIGN.fan.s_PR', pt+'.fan.s_PR')
        prob.model.connect('DESIGN.fan.s_Wc', pt+'.fan.s_Wc')
        prob.model.connect('DESIGN.fan.s_eff', pt+'.fan.s_eff')
        prob.model.connect('DESIGN.fan.s_Nc', pt+'.fan.s_Nc')
        prob.model.connect('DESIGN.lpc.s_PR', pt+'.lpc.s_PR')
        prob.model.connect('DESIGN.lpc.s_Wc', pt+'.lpc.s_Wc')
        prob.model.connect('DESIGN.lpc.s_eff', pt+'.lpc.s_eff')
        prob.model.connect('DESIGN.lpc.s_Nc', pt+'.lpc.s_Nc')
        prob.model.connect('DESIGN.hpc.s_PR', pt+'.hpc.s_PR')
        prob.model.connect('DESIGN.hpc.s_Wc', pt+'.hpc.s_Wc')
        prob.model.connect('DESIGN.hpc.s_eff', pt+'.hpc.s_eff')
        prob.model.connect('DESIGN.hpc.s_Nc', pt+'.hpc.s_Nc')
        prob.model.connect('DESIGN.hpt.s_PR', pt+'.hpt.s_PR')
        prob.model.connect('DESIGN.hpt.s_Wp', pt+'.hpt.s_Wp')
        prob.model.connect('DESIGN.hpt.s_eff', pt+'.hpt.s_eff')
        prob.model.connect('DESIGN.hpt.s_Np', pt+'.hpt.s_Np')
        prob.model.connect('DESIGN.lpt.s_PR', pt+'.lpt.s_PR')
        prob.model.connect('DESIGN.lpt.s_Wp', pt+'.lpt.s_Wp')
        prob.model.connect('DESIGN.lpt.s_eff', pt+'.lpt.s_eff')
        prob.model.connect('DESIGN.lpt.s_Np', pt+'.lpt.s_Np')
        
        #Set up the RHS of the balances!
        prob.model.connect('DESIGN.core_nozz.Throat:stat:area',pt+'.balance.rhs:W')
        prob.model.connect('DESIGN.byp_nozz.Throat:stat:area',pt+'.balance.rhs:BPR')


        prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
        prob.model.connect('DESIGN.fan.Fl_O:stat:area', pt+'.fan.area')
        prob.model.connect('DESIGN.splitter.Fl_O1:stat:area', pt+'.splitter.area1')
        prob.model.connect('DESIGN.splitter.Fl_O2:stat:area', pt+'.splitter.area2')
        prob.model.connect('DESIGN.duct4.Fl_O:stat:area', pt+'.duct4.area')
        prob.model.connect('DESIGN.lpc.Fl_O:stat:area', pt+'.lpc.area')
        prob.model.connect('DESIGN.duct6.Fl_O:stat:area', pt+'.duct6.area')
        prob.model.connect('DESIGN.hpc.Fl_O:stat:area', pt+'.hpc.area')
        prob.model.connect('DESIGN.bld3.Fl_O:stat:area', pt+'.bld3.area')
        prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
        prob.model.connect('DESIGN.hpt.Fl_O:stat:area', pt+'.hpt.area')
        prob.model.connect('DESIGN.duct11.Fl_O:stat:area', pt+'.duct11.area')
        prob.model.connect('DESIGN.lpt.Fl_O:stat:area', pt+'.lpt.area')
        prob.model.connect('DESIGN.duct13.Fl_O:stat:area', pt+'.duct13.area')
        prob.model.connect('DESIGN.byp_bld.Fl_O:stat:area', pt+'.byp_bld.area')
        prob.model.connect('DESIGN.duct15.Fl_O:stat:area', pt+'.duct15.area')

    prob.setup(check=False)

    # FOR DESIGN
    # Note that here the values we are setting are actually DESIGN INPUTS/ FLIGHT CONDITIONS

    # ====== START DECLARING DESIGN VARIABLES ======
    #Flight conditions
    prob.set_val('DESIGN.fc.alt', 35000., units='ft')
    prob.set_val('DESIGN.fc.MN', 0.8)

    #Target Tt4 and Fn_design for the balances
    prob.set_val('DESIGN.balance.rhs:FAR', 2857, units='degR')
    prob.set_val('DESIGN.balance.rhs:W', 5500.0, units='lbf')  

    # Component level setup
    # --- INLET -----
    prob.set_val('DESIGN.inlet.ram_recovery', 0.9990)
    prob.set_val('DESIGN.inlet.MN', 0.751)

    # ---------------
    # ----- FAN -----
    prob.set_val('DESIGN.fan.PR', 1.685)
    prob.set_val('DESIGN.fan.eff', 0.8948)
    prob.set_val('DESIGN.fan.MN', 0.4578)

    # ---------------
    # --- SPLITTER ---
    prob.set_val('DESIGN.splitter.BPR', 5.105)
    prob.set_val('DESIGN.splitter.MN1', 0.3104)
    prob.set_val('DESIGN.splitter.MN2', 0.4518)

    # ---------------
    # --- DUCT 4 -----
    prob.set_val('DESIGN.duct4.dPqP', 0.0048)
    prob.set_val('DESIGN.duct4.MN', 0.3121)
    prob.set_val('DESIGN.lpc.PR', 1.935)

    # ---------------
    # --- LPC -----
    prob.set_val('DESIGN.lpc.eff', 0.9243)
    prob.set_val('DESIGN.lpc.MN', 0.3059)

    # ---------------
    # --- DUCT 6 -----
    prob.set_val('DESIGN.duct6.dPqP', 0.0101),
    prob.set_val('DESIGN.duct6.MN', 0.3563),

    # ---------------
    # ---  HPC -----
    prob.set_val('DESIGN.hpc.PR', 9.369),
    prob.set_val('DESIGN.hpc.eff', 0.8707),
    prob.set_val('DESIGN.hpc.MN', 0.2442),

    # ---------------
    # --- BLEED -----
    prob.set_val('DESIGN.bld3.MN', 0.3000)

    # ---------------
    # --- BURNER -----
    prob.set_val('DESIGN.burner.dPqP', 0.0540),
    prob.set_val('DESIGN.burner.MN', 0.1025),

    # ---------------
    # --- HPT -----
    prob.set_val('DESIGN.hpt.eff', 0.8888),
    prob.set_val('DESIGN.hpt.MN', 0.3650),

    # ---------------
    # --- DUCT -----
    prob.set_val('DESIGN.duct11.dPqP', 0.0051),
    prob.set_val('DESIGN.duct11.MN', 0.3063),

    # ---------------
    # --- LPT -----
    prob.set_val('DESIGN.lpt.eff', 0.8996),
    prob.set_val('DESIGN.lpt.MN', 0.4127),

    # ---------------
    # --- DUCT 13 -----
    prob.set_val('DESIGN.duct13.dPqP', 0.0107),
    prob.set_val('DESIGN.duct13.MN', 0.4463),

    # ---------------
    # --- CORE NOZZLE -----
    prob.set_val('DESIGN.core_nozz.Cv', 0.9933),

    # ---------------
    # --- BLEED -----
    prob.set_val('DESIGN.byp_bld.bypBld:frac_W', 0.005),
    prob.set_val('DESIGN.byp_bld.MN', 0.4489),

    # ---------------
    # --- DUCT 15 -----
    prob.set_val('DESIGN.duct15.dPqP', 0.0149),
    prob.set_val('DESIGN.duct15.MN', 0.4589),

    # ---------------
    # --- BYPASS NOZZ -----
    prob.set_val('DESIGN.byp_nozz.Cv', 0.9939),

    # ---------------
    # --- LP SHAFT -----
    prob.set_val('DESIGN.LP_Nmech', 4666.1, units='rpm'),

    # ---------------
    # --- HP SHAFT -----
    prob.set_val('DESIGN.HP_Nmech', 14705.7, units='rpm'),
    prob.set_val('DESIGN.hp_shaft.HPX', 250.0, units='hp'),

    # --- Set up bleed values -----
    prob.set_val('DESIGN.hpc.cool1:frac_W', 0.050708),
    prob.set_val('DESIGN.hpc.cool1:frac_P', 0.5),
    prob.set_val('DESIGN.hpc.cool1:frac_work', 0.5),
    prob.set_val('DESIGN.hpc.cool2:frac_W', 0.020274),
    prob.set_val('DESIGN.hpc.cool2:frac_P', 0.55),
    prob.set_val('DESIGN.hpc.cool2:frac_work', 0.5),
    prob.set_val('DESIGN.bld3.cool3:frac_W', 0.067214),
    prob.set_val('DESIGN.bld3.cool4:frac_W', 0.101256),
    prob.set_val('DESIGN.hpc.cust:frac_W', 0.0445),
    prob.set_val('DESIGN.hpc.cust:frac_P', 0.5),
    prob.set_val('DESIGN.hpc.cust:frac_work', 0.5),
    prob.set_val('DESIGN.hpt.cool3:frac_P', 1.0),
    prob.set_val('DESIGN.hpt.cool4:frac_P', 0.0),
    prob.set_val('DESIGN.lpt.cool1:frac_P', 1.0),
    prob.set_val('DESIGN.lpt.cool2:frac_P', 0.0),

    # OFF DESIGN
    # The arrays represent multiple flight conditions. 
    OD_MN = [0.8, 0.8, 0.25, 0.00001]
    OD_alt = [35000.0, 35000.0, 0.0, 0.0]
    OD_FAR = [5500.0, 5970.0, 22590.0, 27113.0]
    OD_dTs = [0.0, 0.0, 27.0, 27.0]
    OD_W = [0.0445, 0.0422, 0.0177, 0.0185]

    for i_OD, pt in enumerate(pts):
        prob.set_val(pt+'.fc.MN', OD_MN[i_OD]),
        prob.set_val(pt+'.fc.alt', OD_alt[i_OD], units='ft'),
        prob.set_val(pt+'.balance.rhs:FAR', OD_FAR[i_OD], units='lbf'), #8950.0
        prob.set_val(pt+'.fc.dTs', OD_dTs[i_OD], units='degR')
        prob.set_val(pt+'.hpc.cust:frac_W', OD_W[i_OD])

    # ====== END DECLARING DESIGN VARIABLES ======

    # initial guesses
    prob['DESIGN.balance.FAR'] = 0.025
    prob['DESIGN.balance.W'] = 100.
    prob['DESIGN.balance.lpt_PR'] = 4.0
    prob['DESIGN.balance.hpt_PR'] = 3.0
    prob['DESIGN.fc.balance.Pt'] = 5.2
    prob['DESIGN.fc.balance.Tt'] = 440.0

    W_guesses = [300, 300, 700, 700]
    for i, pt in enumerate(pts):
        # ADP and TOC guesses
        prob[pt+'.balance.FAR'] = 0.02467
        prob[pt+'.balance.W'] = W_guesses[i]
        prob[pt+'.balance.BPR'] = 5.105
        prob[pt+'.balance.lp_Nmech'] = 5000 # 4666.1
        prob[pt+'.balance.hp_Nmech'] = 15000 # 14705.7
        # prob[pt+'.fc.balance.Pt'] = 5.2
        # prob[pt+'.fc.balance.Tt'] = 440.0
        prob[pt+'.hpt.PR'] = 3.
        prob[pt+'.lpt.PR'] = 4.
        prob[pt+'.fan.map.RlineMap'] = 2.0
        prob[pt+'.lpc.map.RlineMap'] = 2.0
        prob[pt+'.hpc.map.RlineMap'] = 2.0

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()
    prob.model.DESIGN.list_outputs(residuals=True, residuals_tol=1e-2)

    for pt in ['DESIGN']+pts:
        viewer(prob, pt)

    print()
    print("Run time", time.time() - st)

