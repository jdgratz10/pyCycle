import numpy as np
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import pycycle.api as pyc

from example_cycles.simple_turboshaft import Turboshaft

class SimpleTurboshaftTestCase(unittest.TestCase):

    def zbenchmark_case1(self):

        prob = om.Problem()

        prob.model = pyc.MPCycle()

        # Create design instance of model
        prob.model.pyc_add_pnt('DESIGN', Turboshaft())
        prob.model.pyc_add_cycle_param('burner.dPqP', .03)
        prob.model.pyc_add_cycle_param('nozz.Cv', 0.99)
        prob.model.pyc_add_pnt('OD', Turboshaft(design=False))

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

        prob.model.pyc_connect_des_od('pt.s_PR', 'pt.s_PR')
        prob.model.pyc_connect_des_od('pt.s_Wp', 'pt.s_Wp')
        prob.model.pyc_connect_des_od('pt.s_eff', 'pt.s_eff')
        prob.model.pyc_connect_des_od('pt.s_Np', 'pt.s_Np')

        prob.model.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
        prob.model.pyc_connect_des_od('comp.Fl_O:stat:area', 'comp.area')
        prob.model.pyc_connect_des_od('burner.Fl_O:stat:area', 'burner.area')
        prob.model.pyc_connect_des_od('turb.Fl_O:stat:area', 'turb.area')
        prob.model.pyc_connect_des_od('pt.Fl_O:stat:area', 'pt.area')

        prob.model.pyc_connect_des_od('nozz.Throat:stat:area', 'balance.rhs:W')

        prob.setup(check=False)

        # Connect design point inputs to model
        prob.set_val('DESIGN.fc.alt', 0.0, units='ft')
        prob.set_val('DESIGN.fc.MN', 0.000001)
        prob.set_val('DESIGN.balance.T4_target', 2370.0, units='degR')
        prob.set_val('DESIGN.balance.pwr_target', 4000.0, units='hp')
        prob.set_val('DESIGN.balance.nozz_PR_target', 1.2)

        prob.set_val('DESIGN.comp.PR', 13.5)
        prob.set_val('DESIGN.comp.eff', 0.83)
        # prob.set_val('DESIGN.burner.dPqP', 0.03)
        prob.set_val('DESIGN.turb.eff', 0.86)
        prob.set_val('DESIGN.pt.eff', 0.9)
        # prob.set_val('DESIGN.nozz.Cv', 0.99)
        prob.set_val('DESIGN.HP_Nmech', 8070.0, units='rpm')
        prob.set_val('DESIGN.LP_Nmech', 5000.0, units='rpm')

        prob.set_val('DESIGN.inlet.MN', 0.60)
        prob.set_val('DESIGN.comp.MN', 0.20)
        prob.set_val('DESIGN.burner.MN', 0.20)
        prob.set_val('DESIGN.turb.MN', 0.4)

        prob.set_val('OD.fc.alt', 0.0, units='ft')
        prob.set_val('OD.fc.MN', .1)
        prob.set_val('OD.LP_Nmech', 5000., units='rpm')
        prob.set_val('OD.balance.pwr_target', 3500., units='hp')

        # Set initial guesses for balances
        prob['DESIGN.balance.FAR'] = 0.0175506829934
        prob['DESIGN.balance.W'] = 27.265
        prob['DESIGN.balance.turb_PR'] = 3.8768
        prob['DESIGN.balance.pt_PR'] = 2.8148
        prob['DESIGN.fc.balance.Pt'] = 14.6955113159
        prob['DESIGN.fc.balance.Tt'] = 518.665288153

        # prob['OD.burner.dPqP'] = 0.03
        # prob['OD.nozz.Cv'] = 0.99

        prob['OD.balance.W'] = 27.265
        prob['OD.balance.FAR'] = 0.0175506829934
        prob['OD.balance.HP_Nmech'] = 8070.0
        prob['OD.fc.balance.Pt'] = 15.703
        prob['OD.fc.balance.Tt'] = 558.31
        prob['OD.turb.PR'] = 3.8768
        prob['OD.pt.PR'] = 2.8148

        np.seterr(divide='raise')

        prob.run_model()
        tol = 1e-3
        print()

        reg_data = 27.265342457866705
        ans = prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 13.5
        ans = prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.01755077946196377
        ans = prob['DESIGN.balance.FAR'][0]
        print('Main FAR:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 3.876811443569159
        ans = prob['DESIGN.balance.turb_PR'][0]
        print('HPT PR:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 800.8503668285215
        ans = prob['DESIGN.perf.Fg'][0]
        print('Fg:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 2.151092078410839
        ans = prob['DESIGN.perf.TSFC'][0]
        print('TSFC:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1190.1777648503974
        ans = prob['DESIGN.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 25.897231212494944
        ans = prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 12.42972778706185
        ans = prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.016304387482120156
        ans = prob['OD.balance.FAR'][0]
        print('Main FAR:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 7853.753342243985
        ans = prob['OD.balance.HP_Nmech'][0]
        print('HP Nmech:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 696.618372248896
        ans = prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 2.5052545862974696
        ans = prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1158.5197002795887
        ans = prob['OD.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, ans)
        assert_near_equal(ans, reg_data, tol)

        print()


if __name__ == "__main__":
    unittest.main()