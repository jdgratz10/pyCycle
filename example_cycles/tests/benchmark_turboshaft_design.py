import numpy as np
import unittest
import os

from openmdao.api import Problem, IndepVarComp
import pycycle.api as pyc
from openmdao.utils.units import convert_units as cu
from openmdao.utils.assert_utils import assert_rel_error

from example_cycles.Turboshaft import Turboshaft


class TurboshaftDesignTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()

        self.prob.model = pyc.MPCycle()

        self.prob.model.pyc_add_pnt('DESIGN', Turboshaft())

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.setup(check=False)
        self.prob.final_setup()

        self.prob.set_val('DESIGN.fc.alt', 28000., units='ft'),
        self.prob.set_val('DESIGN.fc.MN', 0.5),
        self.prob.set_val('DESIGN.balance.rhs:FAR', 2740.0, units='degR'),
        self.prob.set_val('DESIGN.balance.rhs:W', 1.1)
        self.prob.set_val('DESIGN.inlet.ram_recovery', 1.0),

        self.prob.set_val('DESIGN.inlet.MN', 0.4),
        self.prob.set_val('DESIGN.duct1.dPqP', 0.0),
        self.prob.set_val('DESIGN.duct1.MN', 0.4),
        self.prob.set_val('DESIGN.lpc.PR', 5.000),
        self.prob.set_val('DESIGN.lpc.eff', 0.8900),

        self.prob.set_val('DESIGN.lpc.MN', 0.3),
        self.prob.set_val('DESIGN.icduct.dPqP', 0.002),
        self.prob.set_val('DESIGN.icduct.MN', 0.3),
        self.prob.set_val('DESIGN.hpc_axi.PR', 3.0),
        self.prob.set_val('DESIGN.hpc_axi.eff', 0.8900),

        self.prob.set_val('DESIGN.hpc_axi.MN', 0.25),
        self.prob.set_val('DESIGN.bld25.cool1:frac_W', 0.024),
        self.prob.set_val('DESIGN.bld25.cool2:frac_W', 0.0146),
        self.prob.set_val('DESIGN.bld25.MN', 0.3000),
        self.prob.set_val('DESIGN.hpc_centri.PR', 2.7),

        self.prob.set_val('DESIGN.hpc_centri.eff', 0.8800),
        self.prob.set_val('DESIGN.hpc_centri.MN', 0.20),
        self.prob.set_val('DESIGN.bld3.cool3:frac_W', 0.1705),
        self.prob.set_val('DESIGN.bld3.cool4:frac_W', 0.1209),
        self.prob.set_val('DESIGN.bld3.MN', 0.2000),

        self.prob.set_val('DESIGN.duct6.dPqP', 0.00),
        self.prob.set_val('DESIGN.duct6.MN', 0.2000),
        self.prob.set_val('DESIGN.burner.dPqP', 0.050),
        self.prob.set_val('DESIGN.burner.MN', 0.15),
        self.prob.set_val('DESIGN.hpt.eff', 0.89),

        self.prob.set_val('DESIGN.hpt.cool3:frac_P', 1.0),
        self.prob.set_val('DESIGN.hpt.cool4:frac_P', 0.0),
        self.prob.set_val('DESIGN.hpt.MN', 0.30),
        self.prob.set_val('DESIGN.duct43.dPqP', 0.0051),
        self.prob.set_val('DESIGN.duct43.MN', 0.30),

        self.prob.set_val('DESIGN.lpt.eff', 0.9),
        self.prob.set_val('DESIGN.lpt.cool1:frac_P', 1.0),
        self.prob.set_val('DESIGN.lpt.cool2:frac_P', 0.0),
        self.prob.set_val('DESIGN.lpt.MN', 0.4),
        self.prob.set_val('DESIGN.itduct.dPqP', 0.00),

        self.prob.set_val('DESIGN.itduct.MN', 0.4),
        self.prob.set_val('DESIGN.pt.eff', 0.85),
        self.prob.set_val('DESIGN.pt.MN', 0.4),
        self.prob.set_val('DESIGN.duct12.dPqP', 0.00),
        self.prob.set_val('DESIGN.duct12.MN', 0.4),

        self.prob.set_val('DESIGN.nozzle.Cv', 0.99),
        self.prob.set_val('DESIGN.LP_Nmech', 12750., units='rpm'),
        self.prob.set_val('DESIGN.lp_shaft.HPX', 1800.0, units='hp'),
        self.prob.set_val('DESIGN.IP_Nmech', 12000., units='rpm'),
        self.prob.set_val('DESIGN.HP_Nmech', 14800., units='rpm'),

        self.prob['DESIGN.balance.FAR'] = 0.02261
        self.prob['DESIGN.balance.W'] = 10.76
        self.prob['DESIGN.balance.pt_PR'] = 4.939
        self.prob['DESIGN.balance.lpt_PR'] = 1.979
        self.prob['DESIGN.balance.hpt_PR'] = 4.236
        self.prob['DESIGN.fc.balance.Pt'] = 5.666
        self.prob['DESIGN.fc.balance.Tt'] = 440.0

    def zbenchmark_case1(self):
        np.seterr(divide='raise')

        self.prob.run_model()
        tol = 1e-3
        print()

        reg_data = 10.774
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 40.419
        pyc = self.prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.02135
        pyc = self.prob['DESIGN.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4.2325
        pyc = self.prob['DESIGN.balance.hpt_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1.9782
        pyc = self.prob['DESIGN.balance.lpt_PR'][0]
        print('LPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4.921
        pyc = self.prob['DESIGN.balance.pt_PR'][0]
        print('PT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.3758
        pyc = self.prob['DESIGN.nozzle.Fl_O:stat:MN'][0]
        print('Nozz MN:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.31342
        pyc = self.prob['DESIGN.perf.PSFC'][0]
        print('PSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1377.8
        pyc = self.prob['DESIGN.duct6.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

if __name__ == "__main__":
    unittest.main()
