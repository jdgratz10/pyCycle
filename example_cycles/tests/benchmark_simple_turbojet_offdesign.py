import numpy as np
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error

from example_cycles.simple_turbojet import Turbojet

class OffDesignTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = om.Problem()

        self.prob.model.add_subsystem('OD', Turbojet(design=False))

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.setup(check=False)
        self.prob.final_setup()

        self.prob.set_val('OD.fc.alt', 0, units='ft')
        self.prob.set_val('OD.fc.MN', 0.000001)
        self.prob.set_val('OD.balance.Fn_target', 11000.0, units='lbf')
        
        self.prob.set_val('OD.comp.s_PR', 2.97619048)
        self.prob.set_val('OD.comp.s_Wc', 4.91845158)
        self.prob.set_val('OD.comp.s_eff', 0.97532315)
        self.prob.set_val('OD.comp.s_Nc', 8070.)

        self.prob.set_val('OD.turb.s_PR', 0.57536229)
        self.prob.set_val('OD.turb.s_Wp', 0.25338499)
        self.prob.set_val('OD.turb.s_eff', 0.92712376)
        self.prob.set_val('OD.turb.s_Np', 1.6576749)

        self.prob.set_val('OD.inlet.area', 510.88189692, units='inch**2')
        self.prob.set_val('OD.comp.area', 1412.40792088, units='inch**2')
        self.prob.set_val('OD.burner.area', 2144.0215018, units='inch**2')
        self.prob.set_val('OD.turb.area', 394.42366199, units='inch**2')

        self.prob.set_val('OD.balance.rhs:W', 246.32042321, units='inch**2')
        self.prob.set_val('OD.burner.dPqP', 0.03)
        self.prob.set_val('OD.nozz.Cv', 0.99)
        self.prob.set_val('OD.balance.Nmech', 8197.38)

        # Set initial guesses for balances
        self.prob['OD.balance.FAR'] = 0.01680
        self.prob['OD.balance.W'] = 166.073
        self.prob['OD.fc.balance.Pt'] = 15.703
        self.prob['OD.fc.balance.Tt'] = 558.31

    def benchmark_case1(self):
        # ADP Point
        np.seterr(divide='raise')

        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 142.69375835
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 12.84084877
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.01665102
        pyc = self.prob['OD.balance.FAR'][0]
        print('Main FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 7936.36544633
        pyc = self.prob['OD.balance.Nmech'][0]
        print('HP Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 11000.00488519
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.77759879
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1169.51252193
        pyc = self.prob['OD.comp.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

if __name__ == "__main__":
    unittest.main()