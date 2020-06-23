import numpy as np
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error

import pycycle.api as pyc

from example_cycles.simple_turbojet import Turbojet

class SimpleTurbojetTestCase(unittest.TestCase):

    
    def benchmark_case1(self):

        prob = om.Problem()

        prob.model = pyc.MPCycle()

        prob.model.pyc_add_pnt('DESIGN', Turbojet())

        prob.model.pyc_add_cycle_param('burner.dPqP', 0.03)
        prob.model.pyc_add_cycle_param('nozz.Cv', 0.99)

        prob.model.pyc_add_pnt('OD', Turbojet(design=False))

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        

        prob.model.pyc_connect_des_od('comp.s_PR', 'comp.s_PR')
        prob.model.pyc_connect_des_od('comp.s_Wc', 'comp.s_Wc')
        prob.model.pyc_connect_des_od('comp.s_eff', 'comp.s_eff')
        prob.model.pyc_connect_des_od('comp.s_Nc', 'comp.s_Nc')

        prob.model.pyc_connect_des_od('turb.s_PR', 'turb.s_PR')
        prob.model.pyc_connect_des_od('turb.s_Wp', 'turb.s_Wp')
        prob.model.pyc_connect_des_od('turb.s_eff', 'turb.s_eff')
        prob.model.pyc_connect_des_od('turb.s_Np', 'turb.s_Np')

        prob.model.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
        prob.model.pyc_connect_des_od('comp.Fl_O:stat:area', 'comp.area')
        prob.model.pyc_connect_des_od('burner.Fl_O:stat:area', 'burner.area')
        prob.model.pyc_connect_des_od('turb.Fl_O:stat:area', 'turb.area')

        prob.model.pyc_connect_des_od('nozz.Throat:stat:area', 'balance.rhs:W')

        prob.setup(check=False)


        prob.set_val('DESIGN.fc.alt', 0, units='ft')
        prob.set_val('DESIGN.fc.MN', 0.000001)
        prob.set_val('DESIGN.balance.Fn_target', 11800.0, units='lbf')
        prob.set_val('DESIGN.balance.T4_target', 2370.0, units='degR') 

        prob.set_val('DESIGN.comp.PR', 13.5) 
        prob.set_val('DESIGN.comp.eff', 0.83)
        prob.set_val('DESIGN.turb.eff', 0.86)
        prob.set_val('DESIGN.Nmech', 8070.0, units='rpm')

        prob.set_val('DESIGN.inlet.MN', 0.60)
        prob.set_val('DESIGN.comp.MN', 0.020)#.2
        prob.set_val('DESIGN.burner.MN', 0.020)#.2
        prob.set_val('DESIGN.turb.MN', 0.4)

        prob.set_val('OD.fc.alt', 0, units='ft')
        prob.set_val('OD.fc.MN', 0.000001)
        prob.set_val('OD.balance.Fn_target', 11000.0, units='lbf')

        # Set initial guesses for balances
        prob['DESIGN.balance.FAR'] = 0.0175506829934
        prob['DESIGN.balance.W'] = 168.453135137
        prob['DESIGN.balance.turb_PR'] = 4.46138725662
        prob['DESIGN.fc.balance.Pt'] = 14.6955113159
        prob['DESIGN.fc.balance.Tt'] = 518.665288153

        prob['OD.balance.W'] = 166.073
        prob['OD.balance.FAR'] = 0.01680
        prob['OD.balance.Nmech'] = 8197.38
        prob['OD.fc.balance.Pt'] = 15.703
        prob['OD.fc.balance.Tt'] = 558.31

        np.seterr(divide='raise')

        prob.run_model()
        tol = 1e-3
        print()

        reg_data = 147.55303531
        ans = prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        reg_data = 13.500
        ans = prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        reg_data = 0.01755078
        ans = prob['DESIGN.balance.FAR'][0]
        print('Main FAR:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        reg_data = 3.87681144
        ans = prob['DESIGN.balance.turb_PR'][0]
        print('HPT PR:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        reg_data = 11800.00455497
        ans = prob['DESIGN.perf.Fg'][0]
        print('Fg:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        reg_data = 0.79006909
        ans = prob['DESIGN.perf.TSFC'][0]
        print('TSFC:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        reg_data = 1190.17776485
        ans = prob['DESIGN.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        reg_data = 142.69375835
        ans = prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        reg_data = 12.84084877
        ans = prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        reg_data = 0.01665102
        ans = prob['OD.balance.FAR'][0]
        print('Main FAR:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        reg_data = 7936.36544633
        ans = prob['OD.balance.Nmech'][0]
        print('HP Nmech:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        reg_data = 11000.00488519
        ans = prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        reg_data = 0.77759879
        ans = prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        reg_data = 1169.51252193
        ans = prob['OD.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, ans)
        assert_rel_error(self, ans, reg_data, tol)

        print()


if __name__ == "__main__":
    unittest.main()

        