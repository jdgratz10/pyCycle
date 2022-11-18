import openmdao.api as om

import pycycle.api as pyc

import numpy as np


class Propulsor(pyc.Cycle):

    def initialize(self):
        self.options.declare('power_type', default='max', values=['max', 'part'], desc='determines what type of power is targeted, ignored in design mode.')

        super().initialize()

    def setup(self):

        design = self.options['design']
        power_type = self.options['power_type']

        USE_TABULAR = True
        if USE_TABULAR: 
            self.options['thermo_method'] = 'TABULAR'
            self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC
        else: 
            self.options['thermo_method'] = 'CEA'
            self.options['thermo_data'] = pyc.species_data.janaf
            FUEL_TYPE = 'JP-7'

        HP_to_FT_LBF_per_SEC = 550
        convert = 2. * np.pi / 60. / HP_to_FT_LBF_per_SEC


        self.add_subsystem('fc', pyc.FlightConditions())

        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('fan', pyc.Compressor(map_data=pyc.FanMap, map_extrap=True),
                            promotes_inputs=['Nmech'])

        if design:
            self.add_subsystem('fan_dia', om.ExecComp('FanDia = 2.0*(area/(pi*(1.0-hub_tip**2.0)))**0.5',
                                area={'val':7000.0, 'units':'inch**2'},
                                hub_tip={'val':0.25, 'units':None},
                                FanDia={'val':100.0, 'units':'inch'}))
            self.connect('inlet.Fl_O:stat:area', 'fan_dia.area')

            self.add_subsystem('design_Nmech', om.ExecComp('design_fan_Nmech=max_MN*speed_of_sound/(fan_diam/2)',
                               design_fan_Nmech={'val':1000, 'units':'rad/s'},
                               max_MN={'val':.8, 'units':None},
                               speed_of_sound={'val':1077.39, 'units':'ft/s'},  # 1116.45 at sea level static (1077.39 at 10 kft)
                               fan_diam={'val':5, 'units':'ft'}),
                               promotes_outputs=[('design_fan_Nmech', 'Nmech')])
            self.connect('fan_dia.FanDia', 'design_Nmech.fan_diam')


            self.add_subsystem('tip_speed', om.ExecComp('TipSpeed = pi*FanDia*fan_rpm/60',  # rev/sec   ####IMPORTANT!!! Check with Dustin on this, I believe we simply assumed this to be mach 8 in the design_Nmech component
                                fan_rpm={'val': 1000, 'units': 'rpm'},
                                TipSpeed={'val': 12992*0.85, 'units': 'inch/s'}),
                                promotes_inputs=[('fan_rpm', 'Nmech')])         # 12992 in/sec == 330 m/s == speed of sound at SLS
            self.connect('fan_dia.FanDia', 'tip_speed.FanDia')                    # Constrain at design


        self.add_subsystem('nozz', pyc.Nozzle())
        
        self.add_subsystem('shaft', pyc.Shaft(num_ports=2), promotes_inputs=['Nmech'], promotes_outputs=[('trq_net', 'shaft_net_trq')]) 
        
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=0))

        self.add_subsystem('motor_calc_pwr', om.ExecComp('motor_pwr_out=motor_pwr_in*motor_eff',
                           motor_pwr_out={'val':1000, 'units':'hp'},
                           motor_pwr_in={'val':1000, 'units':'hp'},
                           motor_eff={'val':1, 'units':None}),  # for now this is just 1, eventually it will come from the motor model
                           promotes_inputs=['motor_pwr_in', 'motor_eff'],
                           promotes_outputs=['motor_pwr_out'])

        self.add_subsystem('motor_calc_trq', om.ExecComp('motor_trq_out=motor_pwr_out/(Nmech*conversion)',
                           motor_trq_out={'val':1000, 'units':'ft*lbf'},
                           motor_pwr_out={'val':1000, 'units':'hp'},
                           Nmech={'val':200, 'units':'rpm'},
                           conversion={'val':convert, 'units':None}),
                           promotes_inputs=['motor_pwr_out', 'Nmech'],
                           promotes_outputs=['motor_trq_out'])

        self.connect('fan.trq', 'shaft.trq_0')
        self.connect('motor_trq_out', 'shaft.trq_1')


        balance = om.BalanceComp()
        # vary input power to motor until the shaft balance is resolved
        balance.add_balance('motor_input_pwr', units='hp', eq_units='ft*lbf', rhs_val=0, val=2000)

        if design:
            # vary mass flow until the target power is reached
            balance.add_balance('W', units='lbm/s', eq_units='hp', val=50., lower=1., upper=500.)
            self.add_subsystem('balance', balance,
                               promotes_inputs=[('rhs:W', 'pwr_target')])
            self.connect('fan.power', 'balance.lhs:W')

        elif power_type == 'max':
            # vary mass flow till the nozzle area matches the design values
            # balance.add_balance('W', units='lbm/s', eq_units='inch**2', val=50, lower=1., upper=500.)
            # self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('Nmech', val=1000., units='rpm', lower=0.1, upper=10_000, rhs_val=.99)
            self.connect('balance.Nmech', 'Nmech')
            self.connect('fan.map.NcMap', 'balance.lhs:Nmech')

            self.add_subsystem('balance', balance)
            self.set_input_defaults('Nmech', 800, units='rpm')

        else:
            # vary mass flow till the nozzle area matches the design values
            balance.add_balance('W', units='lbm/s', eq_units='inch**2', val=50, lower=1., upper=500.)
            self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('Nmech', val=1., units='rpm', lower=0.1, upper=10_000, eq_units='hp')
            self.connect('balance.Nmech', 'Nmech')
            self.connect('fan.power', 'balance.lhs:Nmech')

            self.add_subsystem('balance', balance)



        self.promotes('balance', inputs=[('lhs:motor_input_pwr', 'shaft_net_trq')], outputs=[('motor_input_pwr', 'motor_pwr_in')])

        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O', 'fan.Fl_I')
        self.pyc_connect_flow('fan.Fl_O', 'nozz.Fl_I')


        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('fan.Fl_O:tot:P', 'perf.Pt3')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

        if design:
            self.connect('balance.W', 'fc.W')

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-12
        newton.options['rtol'] = 1e-12
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        if not design:
            newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        newton.options['reraise_child_analysiserror'] = False
        #
        # newton.linesearch = om.ArmijoGoldsteinLS()
        # newton.linesearch.options['maxiter'] = 3
        newton.linesearch = om.BoundsEnforceLS()
        # newton.linesearch.options['print_bound_enforce'] = True
        # newton.linesearch.options['iprint'] = -1
        #
        self.linear_solver = om.DirectSolver()

        # base_class setup should be called as the last thing in your setup
        super().setup()

def viewer(prob, pt):
    """
    print a report of all the relevant cycle properties
    """

    summary_data = (
        prob[pt + ".fc.Fl_O:stat:MN"],
        prob.get_val(pt + ".fc.alt", units='ft'),
        prob.get_val(pt + ".fc.dTs", units='degR'),
        prob.get_val(pt + ".fc.W", units='lbm/s'),
        prob.get_val(pt + ".perf.Fn", units='lbf'),
        prob.get_val(pt + ".Nmech", units='rpm'),
        prob[pt + ".motor_eff"],
        prob.get_val(pt + ".motor_pwr_in", units='hp'),
        prob.get_val(pt + ".motor_pwr_out", units='hp'),
        prob.get_val(pt + ".fan.power", units='hp')
    )

    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(
        "----------------------------------------------------------------------------",
        flush=True,
    )
    print("                              POINT:", pt, flush=True)
    print(
        "----------------------------------------------------------------------------",
        flush=True,
    )
    print("                       PERFORMANCE CHARACTERISTICS", flush=True)
    print(
        "     Mach         Alt              dTamb       W             Fn            fan_Nmech      motor_eff    motor_pwr_in  motor_pwr_out fan_pwr",
        flush=True,
    )
    print(
        " %9.3f  %13.1f %13.2f %13.2f %13.1f %13.1f %13.3f %13.1f %13.1f  %13.1f"
        % summary_data,
        flush=True,
    )

    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'fan.Fl_O', 'nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names)

    pyc.print_compressor(prob, [f'{pt}.fan'])

    pyc.print_nozzle(prob, [f'{pt}.nozz'])

def map_plots(prob, pt):
    comp_names = ['fan']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.plot_compressor_maps(prob, comp_full_names)



class MPpropulsor(pyc.MPCycle):

    def setup(self):

        design = self.pyc_add_pnt('design', Propulsor(design=True, thermo_method='CEA'))
        self.set_input_defaults('design.Nmech', 1000, units='rpm')
        self.set_input_defaults('design.fc.MN', .8)
        self.set_input_defaults('design.fc.alt', 10000, units='ft')

        self.pyc_add_pnt('OD_max_pwr', Propulsor(design=False, thermo_method='CEA', power_type='max'))
        self.set_input_defaults('OD_max_pwr.fc.MN', .8)
        self.set_input_defaults('OD_max_pwr.fc.alt', 10000, units='ft')

        # self.add_subsystem('calc_part_pwr', om.ExecComp('part_pwr=max_pwr*throttle_percentage',
        #                    part_pwr={'val':1000, 'units':'hp'},
        #                    max_pwr={'val':1000, 'units':'hp'},
        #                    throttle_percentage={'val':1, 'units':None}),
        #                    promotes_inputs=['throttle_percentage'],
        #                    promotes_outputs=['part_pwr'])

        # self.pyc_add_pnt('OD_prt_pwr', Propulsor(design=False, thermo_method='CEA', power_type='part'))
        # self.set_input_defaults('OD_prt_pwr.fc.MN', .8)
        # self.set_input_defaults('OD_prt_pwr.fc.alt', 10000, units='ft')

        # self.connect('part_pwr', 'OD_prt_pwr.balance.rhs:Nmech')
        # self.connect('OD_max_pwr.fan.power', 'calc_part_pwr.max_pwr')

    

        self.pyc_use_default_des_od_conns()

        # self.pyc_connect_des_od('nozz.Throat:stat:area', 'balance.rhs:W')

        super().setup()
        


if __name__ == "__main__":
    import time

    import numpy as np

    prob = om.Problem()

    prob.model = mp_propulsor = MPpropulsor()


    prob.setup()

    #Define the design point
    prob.set_val('design.fc.alt', 10000, units='ft')
    prob.set_val('design.fc.MN', 0.8)
    prob.set_val('design.inlet.MN', 0.6)
    prob.set_val('design.fan.PR', 1.3)
    prob.set_val('design.pwr_target', -2100.041, units='hp')
    prob.set_val('design.fan.eff', 0.96)


    # Set initial guesses for balances
    prob['design.balance.W'] = 200.
    
    # for i, pt in enumerate(['OD_max_pwr', 'OD_prt_pwr']):
    for i, pt in enumerate(['OD_max_pwr',]):
    
        # initial guesses
        prob[pt+'.fan.PR'] = 1.3
        # prob[pt+'.balance.W'] = 62.44
        # prob[pt+'.balance.Nmech'] = 852.3
        prob[pt+'.fc.W'] = 62.44
        prob.set_val(pt+'.balance.Nmech', 852.3, units='rad/s')

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=2)
    prob.model.design.nonlinear_solver.options['atol'] = 1e-6
    prob.model.design.nonlinear_solver.options['rtol'] = 1e-6

    prob.model.OD_max_pwr.nonlinear_solver.options['atol'] = 1e-6
    prob.model.OD_max_pwr.nonlinear_solver.options['rtol'] = 1e-6
    # prob.model.OD_prt_pwr.nonlinear_solver.options['atol'] = 1e-6
    # prob.model.OD_prt_pwr.nonlinear_solver.options['rtol'] = 1e-6


    prob.run_model()
    run_time = time.time() - st

    # for pt in ['design', 'OD_max_pwr', 'OD_prt_pwr']:
    for pt in ['design', 'OD_max_pwr',]:
        # print('\n\n\n'+'#'*100,)
        # print(pt)
        # print('#'*100+'\n')
        viewer(prob, pt)

    map_plots(prob,'design')


    print("Run time", run_time)
    # prob.model.list_outputs(implicit=True, prom_name=True, explicit=False, includes='*OD*', residuals=True, print_max=True, print_min=True)
    # print(prob.get_val('design.fan.Nc', units=None))
    # print(prob.get_val('OD_max_pwr.fan.Nc', units=None))
    # print(prob.get_val('design.Nmech', units='rpm'))
    # print(prob.get_val('OD_max_pwr.Nmech', units='rpm'))
    # 'design_Nmech', om.ExecComp('design_fan_Nmech=max_MN*speed_of_sound/(fan_diam/2)'
    # print(prob.get_val('design.design_Nmech.design_fan_Nmech', units='rad/s'))
    # print(prob.get_val('design.design_Nmech.max_MN', units=None))
    # print(prob.get_val('design.design_Nmech.speed_of_sound', units='ft/s'))
    # print(prob.get_val('design.design_Nmech.fan_diam', units='ft'))
    # print(prob.get_val('design.inlet.Fl_O:stat:area', units='ft**2'))