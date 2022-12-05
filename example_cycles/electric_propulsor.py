import openmdao.api as om

import pycycle.api as pyc

import numpy as np
import sys

from pycycle.maps.Motor_map import MotorMap

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
                            area={'val':1000.0, 'units':'inch**2'},
                            hub_tip={'val':0.25, 'units':None},
                            FanDia={'val':100.0, 'units':'inch'}))
            self.connect('inlet.Fl_O:stat:area', 'fan_dia.area')
            

            self.add_subsystem('design_Nmech', om.ExecComp('design_fan_Nmech=4*pi*(max_MN*speed_of_sound)/(fan_diam)',
                               design_fan_Nmech={'val':1000, 'units':'rpm'},
                               max_MN={'val':1, 'units':None},
                               speed_of_sound={'val':1116., 'units':'ft/s'},  # 1116.45 at sea level static (1077.39 at 10 kft)
                               fan_diam={'val':5, 'units':'ft'}),
                               promotes_outputs=[('design_fan_Nmech', 'Nmech')])
            self.connect('fan_dia.FanDia', 'design_Nmech.fan_diam')



            self.add_subsystem('tip_speed', om.ExecComp('TipSpeed = pi*FanDia*fan_rpm/60',
                            fan_rpm={'val': 1200, 'units': 'rpm'},
                            TipSpeed={'val': 12992*0.85, 'units': 'inch/s'},
                            FanDia={'val': 100, 'units': 'inch'}),
                            promotes_inputs=[('fan_rpm', 'Nmech')])                      # 12992 in/sec == 330 m/s == speed of sound at SLS
            self.connect('fan_dia.FanDia', 'tip_speed.FanDia') 


        
        self.add_subsystem('motor', MotorMap(), promotes_inputs=[('motor_rpm', 'Nmech'), 'motor_power_in', ('fan_power', 'fan.power')], 
                                                promotes_outputs=['motor_efficiency', 'motor_torque_out', 'motor_power_out'])

        

        self.add_subsystem('nozz', pyc.Nozzle())

        self.add_subsystem('shaft', pyc.Shaft(num_ports=2), promotes_inputs=['Nmech'], promotes_outputs=[('trq_net', 'shaft_net_trq')])

        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=0), promotes_outputs=['Fn'])

        
        self.connect('fan.trq', 'shaft.trq_0')
        self.connect('motor_torque_out', 'shaft.trq_1')



        balance = om.BalanceComp()
        balance.add_balance('motor_input_power', units='hp', eq_units='ft*lbf', rhs_val=0, val=15000)
        
        if design:
            # vary mass flow until the target power is reached
            balance.add_balance('W', units='lbm/s', eq_units='hp', val=50., lower=1., upper=50000.)
            self.add_subsystem('balance', balance,
                               promotes_inputs=[('rhs:W', 'pwr_target')])
            self.connect('fan.power', 'balance.lhs:W')

            

        elif power_type == 'max':
            # vary mass flow till the nozzle area matches the design values
            balance.add_balance('W', units='lbm/s', eq_units='inch**2', val=50, lower=1., upper=50000.)
            self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('Nmech', val=1000., units='rpm', lower=0.1, upper=10000, rhs_val=.99)
            self.connect('balance.Nmech', 'Nmech')
            self.connect('fan.map.NcMap', 'balance.lhs:Nmech')

            self.add_subsystem('balance', balance)
            self.set_input_defaults('Nmech', 800, units='rpm')

        else:
            # vary mass flow till the nozzle area matches the design values
            balance.add_balance('W', units='lbm/s', eq_units='inch**2', val=50, lower=1., upper=50000.)
            self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('Nmech', val=1., units='rpm', lower=0.1, upper=10000, eq_units='hp')
            self.connect('balance.Nmech', 'Nmech')
            self.connect('fan.power', 'balance.lhs:Nmech')

            self.add_subsystem('balance', balance)

        # balance the shaft and motor torques
        self.promotes('balance', inputs=[('lhs:motor_input_power', 'shaft_net_trq')], outputs=[('motor_input_power', 'motor_power_in')])


        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O', 'fan.Fl_I')
        self.pyc_connect_flow('fan.Fl_O', 'nozz.Fl_I')


        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('fan.Fl_O:tot:P', 'perf.Pt3')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

        self.connect('balance.W', 'fc.W')



        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-12
        newton.options['rtol'] = 1e-12
        newton.options['iprint'] = 0
        newton.options['maxiter'] = 20
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        newton.options['reraise_child_analysiserror'] = False
        newton.options['err_on_non_converge'] = True
        #
        # newton.linesearch = om.ArmijoGoldsteinLS()
        # newton.linesearch.options['maxiter'] = 3
        newton.linesearch = om.BoundsEnforceLS()
        # newton.linesearch.options['print_bound_enforce'] = True
        newton.linesearch.options['iprint'] = -1
        #
        self.linear_solver = om.DirectSolver()

        # base_class setup should be called as the last thing in your setup
        super().setup()

def viewer(prob, pt, file=sys.stdout):
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
        prob[pt + ".motor_efficiency"],
        prob.get_val(pt + ".motor_power_in", units='hp'),
        prob.get_val(pt + ".motor.motor_current", units='A'),
        prob.get_val(pt + ".fan.power", units='hp')
        # prob.get_val(pt + ".fan.trq", units='ft*lbf')
    )

    print(flush=True, file=file)
    print(flush=True, file=file)
    print(flush=True, file=file)
    print(
        "----------------------------------------------------------------------------",
        flush=True, file=file
    )
    print("                              POINT:", pt, flush=True, file=file)
    print(
        "----------------------------------------------------------------------------",
        flush=True, file=file
    )
    print("                       PERFORMANCE CHARACTERISTICS", flush=True, file=file)
    print(
        "     Mach         Alt              dTamb       W             Fn            fan_Nmech      motor_eff    motor_pwr_in  motor_current fan_pwr",
        flush=True, file=file
    )
    print(
        " %9.3f  %13.1f %13.2f %13.2f %13.1f %13.1f %13.3f %13.1f %13.1f  %13.1f"
        % summary_data,
        flush=True, file=file
    )

    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'fan.Fl_O', 'nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    pyc.print_compressor(prob, [f'{pt}.fan'], file=file)

    pyc.print_nozzle(prob, [f'{pt}.nozz'], file=file)




def map_plots(prob, pt):
    comp_names = ['fan']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.plot_compressor_maps(prob, comp_full_names)



class MPpropulsor(pyc.MPCycle):

    def setup(self):

        design = self.pyc_add_pnt('design', Propulsor(design=True, thermo_method='CEA'))
        self.set_input_defaults('design.Nmech', 1000, units='rpm')
        self.set_input_defaults('design.fc.MN', .25)
        self.set_input_defaults('design.fc.alt', 0, units='ft')

        self.pyc_add_pnt('OD_max_pwr', Propulsor(design=False, thermo_method='CEA', power_type='max'), promotes_inputs=[('fc.MN', 'OD_MN'), ('fc.alt', 'OD_alt')])

        self.add_subsystem('calc_part_pwr', om.ExecComp('part_pwr=max_pwr*throttle_percentage',
                           part_pwr={'val':15000, 'units':'hp'},
                           max_pwr={'val':15000, 'units':'hp'},
                           throttle_percentage={'val':1, 'units':None}),
                           promotes_inputs=['throttle_percentage'],
                           promotes_outputs=['part_pwr'])

        self.pyc_add_pnt('OD_prt_pwr', Propulsor(design=False, thermo_method='CEA', power_type='part'), promotes_inputs=[('fc.MN', 'OD_MN'), ('fc.alt', 'OD_alt')])

        self.connect('part_pwr', 'OD_prt_pwr.balance.rhs:Nmech')
        self.connect('OD_max_pwr.fan.power', 'calc_part_pwr.max_pwr')

    

        self.pyc_use_default_des_od_conns()

        self.pyc_connect_des_od('nozz.Throat:stat:area', 'balance.rhs:W')


        super().setup()











if __name__ == "__main__":
    import time

    import numpy as np

    prob = om.Problem()

    prob.model = mp_propulsor = MPpropulsor()


    prob.setup()

    # prob.model.list_outputs(includes=['*CONV*'])
    # prob.model.list_inputs(includes=['*CONV*'])


    #Define the design point
    prob.set_val('design.fc.alt', 0.1, units='ft')
    prob.set_val('design.fc.MN', 0.25)
    prob.set_val('design.inlet.MN', 0.2)
    prob.set_val('design.fan.PR', 1.3)
    prob.set_val('design.pwr_target', -15625, units='hp')
    prob.set_val('design.fan.eff', 0.96)
    prob.set_val('design.motor.supp_voltage', 10000, units='V')

    ''''
    Static values 0 mn 0 alt numbers as well
    corrected air flow
    10kw/kg electric motor model 12 MW
    '''


    # Set initial guesses for balances
    prob['design.balance.W'] = 1000.
    
    for i, pt in enumerate(['OD_max_pwr', 'OD_prt_pwr']):
    
    #     # initial guesses
        prob[pt+'.fan.PR'] = 1.3
        prob.set_val(pt+'.balance.W', 1000, units='kg/s')
        prob.set_val(pt+'.balance.Nmech', 100, units='rad/s')

    st = time.time()

    # prob.set_solver_print(level=-1)
    # prob.set_solver_print(level=0, depth=2)
    prob.model.design.nonlinear_solver.options['atol'] = 1e-6
    prob.model.design.nonlinear_solver.options['rtol'] = 1e-6

    prob.model.OD_max_pwr.nonlinear_solver.options['atol'] = 1e-6
    prob.model.OD_max_pwr.nonlinear_solver.options['rtol'] = 1e-6
    prob.model.OD_prt_pwr.nonlinear_solver.options['atol'] = 1e-6
    prob.model.OD_prt_pwr.nonlinear_solver.options['rtol'] = 1e-6


    print()
    print()
    print('#####################################################################')
    print('###################### RUNNING AT DESIGN POINT ######################')
    print('#####################################################################')
    print()

    prob.set_val('OD_MN', .8)
    prob.set_val('OD_alt', 10000, units='ft')
    prob.set_val('throttle_percentage', 1)


    prob.run_model()



    print('Fan Diameter [in]:. . . .', prob.get_val('design.fan_dia.FanDia', units='inch'))
    print('Flow Area [in**2]:. . . .', prob.get_val('design.fan_dia.area', units='inch**2'))
    print('Fan Power [hp]:. . . . . ', prob.get_val('design.fan.power', units='hp'))
    print('Net Thrust:. . . . . . . ', prob.get_val('design.Fn'))
    print('Motor Power In [kW]:. . .', prob.get_val('design.motor_power_in', units='kW'))
    print('Motor Power In [hp]:. . .', prob.get_val('design.motor_power_in', units='hp'))
    print('Motor Power Out [hp]:. . ', prob.get_val('design.motor_power_out', units='hp'))
    print('Motor Torque [Nm]:. . . .', prob.get_val('design.motor_torque_out', units='N*m'))
    print('Motor Efficiency:. . . . ', prob.get_val('design.motor_efficiency'))
    print('Shaft Net Torque:. . . . ', prob.get_val('design.shaft_net_trq'))
    print('Motor Current [A]:. . . .', prob.get_val('design.motor.motor_current', units='A'))
    print('Motor RPM:. . . . . . . .', prob.get_val('design.Nmech', units='rpm'))







    view_file = open('envelope_out.txt', 'w')

    for pt in ['design', 'OD_max_pwr', 'OD_prt_pwr']:
        viewer(prob, pt, file=view_file)

    print()
    print()
    print('#####################################################################')
    print('##################### RUNNING OFF-DESIGN POINTS #####################')
    print('#####################################################################')
    print()

    MNs = [.8, .7, .6, .5]
    # alts = [10000, 90000, 80000]
    # percentages = [1, .9, .8, .7, .6]

    MNs_high = [.8, .7, .6, .5, .4]
    MNs_low = [.8, .7, .6, .5, .4, .3, .2, .1, 0.001]
    alts = [10000, 7000, 4000, 2000, 0.0, 1000, 3000, 6000, 9000, 11000, 13000, 15000, 17000, 19000, 20000, 25000, 27000, 29000, 30000, 35000, 37000, 39000, 40000, 43000]
    percentages = [.9, .8, .7, .6, .5, .4, .3]
    MNs_high = [.8,]
    MNs_low = [.8,]
    alts = [10000, 13000, 15000, 17000, 19000, 20000, 25000, 27000, 29000, 30000, 35000, 37000, 39000, 40000, 43000]
    percentages = [.9, .8, .7, .6, .5, .4, .3]

    prob.model.set_solver_print(level=-1, depth=100, type_='all')
    # prob.model.set_solver_print(level=-1, depth=100, type_='NL')

    # up through this point converges: MN = 0.8   alt = 29000   throttle = 30.0 %

    for alt in alts:
        if alt <= 5000:
            MNs = MNs_low
        else:
            MNs = MNs_high

        # if alt != 20000:
        #     percentages = [.9,]
        #     # prob.set_solver_print(level=-1)
        #     # prob.set_solver_print(level=2, depth=4)
        # else:
        #     percentages = [.9, .8, .7, .6, .5, .4, .3]
        #     # prob.set_solver_print(level=-1)
        #     # prob.set_solver_print(level=0, depth=2)

        for MN in MNs:
            prob.set_val('OD_MN', MN)
            prob.set_val('OD_alt', alt, units='ft')
            for i, percentage in enumerate(percentages):
                print('##########################################')
                print('MN =', MN, '  alt =', alt, '  throttle =', percentage*100, '%')
                print('##########################################')
                prob.set_val('throttle_percentage', percentage, units=None)
                prob.run_model()

                if i == 0:
                    viewer(prob, 'OD_max_pwr', file=view_file)
                viewer(prob, 'OD_prt_pwr', file=view_file)

            #run back up in percentage
            for percentage in [.4, .6, .8]:
                prob.set_val('throttle_percentage', percentage, units=None)
                prob.run_model()

        # run back up in mach
        for MN in [.6, .7, .8]:
            prob.set_val('OD_MN', MN)
            prob.run_model()