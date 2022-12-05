import numpy as np
import openmdao.api as om
import os
import pandas as pd



"""
PW127-E 
- Approximately 12,000 ft*lb of torque at design RPM. This is also in line with the PW127 data, see type cert data sheet
- Power = 1.5 MW or roughly 2000 hp
*** currently the fan torque is low roughly an order of magnitude.***
"""


class MotorMap(om.Group):

    def setup(self):
        #Speed(y)/Current(x)
        map_data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../maps'))
        df = pd.read_excel(os.path.join(map_data_path,  'pw127e_motor_eff_map.xlsx'))
        df = df.div(100)

        # There are 100 current values (columns)
        # And 141 rpm values (rows)
        motor_loss_data = df.to_numpy()
        motor_eff_interp = om.MetaModelStructuredComp(method='slinear', extrapolate=True)

        current_data = np.linspace(0, 3500, len(motor_loss_data[0]))
        rpm_data = np.linspace(0, 2000, len(motor_loss_data))



        motor_eff_interp.add_input('motor_rpm', val=1000, units='rpm', training_data=rpm_data)
        motor_eff_interp.add_input('motor_current', val=1000, units='A', training_data=current_data)

        motor_eff_interp.add_output('motor_efficiency', val=0.90, training_data=motor_loss_data)



        # this exec comp connects the motor and fan power, but converts the value to Amps which is needed for the lookup table.
        self.add_subsystem('motor_current_calc', om.ExecComp('motor_current = (-1)*fan_power*hp_to_W_conv/supp_voltage * (1+(1-motor_efficiency))',
                            motor_current={'val':1000, 'units':'A'},
                            fan_power={'val':2000, 'units':'hp'},
                            supp_voltage={'val':2500, 'units':'V'},
                            motor_efficiency={'val': 0.90, 'units': None},
                            hp_to_W_conv={'val':745.6999, 'units': 'W/hp'}),
                            promotes_inputs=['fan_power', 'supp_voltage', 'motor_efficiency'],
                            promotes_outputs=['motor_current'])


        self.add_subsystem('motor_eff', motor_eff_interp, 
                            promotes_inputs=['motor_current', 'motor_rpm'], 
                            promotes_outputs=['motor_efficiency'])


        self.add_subsystem('motor_pwr', om.ExecComp('motor_power_out = motor_power_in * motor_efficiency',
                           motor_power_out={'val':1000, 'units':'hp'},
                           motor_power_in={'val':1000, 'units':'hp'},
                           motor_efficiency={'val':1, 'units':None}),  
                           promotes_inputs=['motor_power_in', 'motor_efficiency'],
                           promotes_outputs=['motor_power_out'])



        self.add_subsystem('motor_trq', om.ExecComp('motor_torque_out = HP_per_RPM_to_FT_LBF*(motor_power_out)/(motor_rpm)',
                            motor_torque_out={'val': 1000, 'units': 'ft*lbf'},
                            motor_power_out={'val': 2000, 'units': 'hp'},
                            motor_rpm={'val': 1000, 'units': 'rpm'},
                            HP_per_RPM_to_FT_LBF={'val': 5252.11, 'units': '(rad/s)/rpm'}),
                            promotes_inputs=['motor_power_out', 'motor_rpm'],
                            promotes_outputs=['motor_torque_out'])        




if __name__ == "__main__":
    from openmdao.api import Problem

    prob = Problem()
    prob.model. add_subsystem('motor_eff', MotorMap(), promotes=['*'])
    prob.setup()
    
    prob.set_val('motor_current', 1)
    prob.set_val('motor_rpm', 1)

    prob.run_model()

    print('Efficiency: ', prob.get_val('motor_efficiency'))
    print('RPM: ', prob.get_val('motor_rpm'))
    print('Current: ', prob.get_val('motor_current'))