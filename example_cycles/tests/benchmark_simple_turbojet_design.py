import numpy as np
import unittest
import os

import openmdao.api as om
import pycycle.api as pyc
from openmdao.utils.assert_utils import assert_rel_error

from example_cycles.simple_turbojet import Turbojet

class DesignTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = om.Problem()

        self.prob.model = pyc.MPCycle()

        self.prob.model.pyc_add_pnt('DESIGN', Turbojet())

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.setup(check=False)
        self.prob.final_setup()

        self.prob.set_val('DESIGN.fc.alt', 0, units='ft')
        self.prob.set_val('DESIGN.fc.MN', 0.000001)
        self.prob.set_val('DESIGN.balance.Fn_target', 11800.0, units='lbf')
        self.prob.set_val('DESIGN.balance.T4_target', 2370.0, units='degR') 

        self.prob.set_val('DESIGN.comp.PR', 13.5) 
        self.prob.set_val('DESIGN.comp.eff', 0.83)
        self.prob.set_val('DESIGN.burner.dPqP', 0.03)
        self.prob.set_val('DESIGN.turb.eff', 0.86)
        self.prob.set_val('DESIGN.nozz.Cv', 0.99)
        self.prob.set_val('DESIGN.Nmech', 8070.0, units='rpm')

        self.prob.set_val('DESIGN.inlet.MN', 0.60)
        self.prob.set_val('DESIGN.comp.MN', 0.020)#.2
        self.prob.set_val('DESIGN.burner.MN', 0.020)#.2
        self.prob.set_val('DESIGN.turb.MN', 0.4)

        # Set initial guesses for balances
        self.prob['DESIGN.balance.FAR'] = 0.0175506829934
        self.prob['DESIGN.balance.W'] = 168.453135137
        self.prob['DESIGN.balance.turb_PR'] = 4.46138725662
        self.prob['DESIGN.fc.balance.Pt'] = 14.6955113159
        self.prob['DESIGN.fc.balance.Tt'] = 518.665288153

    def benchmark_case1(self):
        np.seterr(divide='raise')

        self.prob.run_model()
        tol = 1e-3
        print()

        reg_data = 147.55303531
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 13.500
        pyc = self.prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.01755078
        pyc = self.prob['DESIGN.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 3.87681144
        pyc = self.prob['DESIGN.balance.turb_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 11800.00455497
        pyc = self.prob['DESIGN.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.79006909
        pyc = self.prob['DESIGN.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1190.17776485
        pyc = self.prob['DESIGN.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()


if __name__ == "__main__":
    unittest.main()

        