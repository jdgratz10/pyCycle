import numpy as np
import unittest
import os


import openmdao.api as om
import pycycle.api as pyc
from openmdao.utils.assert_utils import assert_rel_error

from example_cycles.high_bypass_turbofan import HBTF


class CFM56DesignTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = om.Problem()

        self.prob.model = pyc.MPCycle()

        self.prob.model.pyc_add_pnt('DESIGN', HBTF())

        self.prob.setup(check=False)

        self.prob.set_val('DESIGN.fc.alt', 35000., units='ft')
        self.prob.set_val('DESIGN.fc.MN', 0.8)
        self.prob.set_val('DESIGN.balance.rhs:W', 5500.0, units='lbf')
        self.prob.set_val('DESIGN.balance.rhs:FAR', 2857.0, units='degR')
        self.prob.set_val('DESIGN.inlet.ram_recovery', 0.9990)
        self.prob.set_val('DESIGN.inlet.MN', 0.751)
        self.prob.set_val('DESIGN.fan.PR', 1.685)
        self.prob.set_val('DESIGN.fan.eff', 0.8948)
        self.prob.set_val('DESIGN.fan.MN', 0.4578)
        self.prob.set_val('DESIGN.splitter.BPR', 5.105)
        self.prob.set_val('DESIGN.splitter.MN1', 0.3104)
        self.prob.set_val('DESIGN.splitter.MN2', 0.4518)
        self.prob.set_val('DESIGN.duct4.dPqP', 0.0048)
        self.prob.set_val('DESIGN.duct4.MN', 0.3121)
        self.prob.set_val('DESIGN.lpc.PR', 1.935)
        self.prob.set_val('DESIGN.lpc.eff', 0.9243)
        self.prob.set_val('DESIGN.lpc.MN', 0.3059)
        self.prob.set_val('DESIGN.duct6.dPqP', 0.0101)
        self.prob.set_val('DESIGN.duct6.MN', 0.3563)
        self.prob.set_val('DESIGN.hpc.PR', 9.369)
        self.prob.set_val('DESIGN.hpc.eff', 0.8707)
        self.prob.set_val('DESIGN.hpc.MN', 0.2442)
        self.prob.set_val('DESIGN.bld3.MN', 0.3000)
        self.prob.set_val('DESIGN.burner.dPqP', 0.0540)
        self.prob.set_val('DESIGN.burner.MN', 0.1025)
        self.prob.set_val('DESIGN.hpt.eff', 0.8888)
        self.prob.set_val('DESIGN.hpt.MN', 0.3650)
        self.prob.set_val('DESIGN.duct11.dPqP', 0.0051)
        self.prob.set_val('DESIGN.duct11.MN', 0.3063)
        self.prob.set_val('DESIGN.lpt.eff', 0.8996)
        self.prob.set_val('DESIGN.lpt.MN', 0.4127)
        self.prob.set_val('DESIGN.duct13.dPqP', 0.0107)
        self.prob.set_val('DESIGN.duct13.MN', 0.4463)
        self.prob.set_val('DESIGN.core_nozz.Cv', 0.9933)
        self.prob.set_val('DESIGN.byp_bld.bypBld:frac_W', 0.005)
        self.prob.set_val('DESIGN.byp_bld.MN',  0.4489)
        self.prob.set_val('DESIGN.duct15.dPqP', 0.0149)
        self.prob.set_val('DESIGN.duct15.MN', 0.4589)
        self.prob.set_val('DESIGN.byp_nozz.Cv', 0.9939)
        self.prob.set_val('DESIGN.LP_Nmech', 4666.1, units='rpm')
        self.prob.set_val('DESIGN.HP_Nmech', 14705.7, units='rpm')
        self.prob.set_val('DESIGN.hp_shaft.HPX', 250.0, units='hp')
        self.prob.set_val('DESIGN.hpc.cool1:frac_W', 0.050708)
        self.prob.set_val('DESIGN.hpc.cool1:frac_P', 0.5)
        self.prob.set_val('DESIGN.hpc.cool1:frac_work', 0.5)
        self.prob.set_val('DESIGN.hpc.cool2:frac_W', 0.020274)
        self.prob.set_val('DESIGN.hpc.cool2:frac_P', 0.55)
        self.prob.set_val('DESIGN.hpc.cool2:frac_work', 0.5)
        self.prob.set_val('DESIGN.bld3.cool3:frac_W', 0.067214)
        self.prob.set_val('DESIGN.bld3.cool4:frac_W', 0.101256)
        self.prob.set_val('DESIGN.hpc.cust:frac_W', 0.0445)
        self.prob.set_val('DESIGN.hpc.cust:frac_P', 0.5)
        self.prob.set_val('DESIGN.hpc.cust:frac_work', 0.5)
        self.prob.set_val('DESIGN.hpt.cool3:frac_P', 1.0)
        self.prob.set_val('DESIGN.hpt.cool4:frac_P', 0.0 )
        self.prob.set_val('DESIGN.lpt.cool1:frac_P', 1.0)
        self.prob.set_val('DESIGN.lpt.cool2:frac_P', 0.0)

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)

        self.prob['DESIGN.balance.FAR'] = 0.025
        self.prob['DESIGN.balance.W'] = 316.0
        self.prob['DESIGN.balance.lpt_PR'] = 4.4
        self.prob['DESIGN.balance.hpt_PR'] = 3.6
        self.prob['DESIGN.fc.balance.Pt'] = 5.2
        self.prob['DESIGN.fc.balance.Tt'] = 440.0

        # from openmdao.api import view_model
        # view_model(self.prob)
        # exit()

    def zbenchmark_case1(self):
        np.seterr(divide='raise')

        self.prob.run_model()
        tol = 1e-3
        print()

        reg_data = 321.253
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 30.094
        pyc = self.prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.02491
        pyc = self.prob['DESIGN.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 3.6228
        pyc = self.prob['DESIGN.balance.hpt_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4.3687
        pyc = self.prob['DESIGN.balance.lpt_PR'][0]
        print('LPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 13274.4
        pyc = self.prob['DESIGN.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.63101
        pyc = self.prob['DESIGN.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1276.48
        pyc = self.prob['DESIGN.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

if __name__ == "__main__":
    unittest.main()
