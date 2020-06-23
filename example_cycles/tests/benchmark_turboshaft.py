import numpy as np
import unittest
import os

from openmdao.api import Problem, IndepVarComp
import pycycle.api as pyc
from openmdao.utils.units import convert_units as cu
from openmdao.utils.assert_utils import assert_rel_error

from example_cycles.Turboshaft import Turboshaft


class TurboshaftTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()

        self.prob.model = pyc.MPCycle()

        self.prob.model.pyc_add_pnt('DESIGN', Turboshaft())
        self.prob.model.pyc_add_pnt('OD', Turboshaft(design=False, maxiter=10), promotes=[('inlet.ram_recovery', 'DESIGN.inlet.ram_recovery'),
            ('duct1.dPqP', 'DESIGN.duct1.dPqP'), ('icduct.dPqP', 'DESIGN.icduct.dPqP'), ('bld25.cool1:frac_W', 'DESIGN.bld25.cool1:frac_W'),
            ('bld25.cool2:frac_W', 'DESIGN.bld25.cool2:frac_W'), ('duct6.dPqP', 'DESIGN.duct6.dPqP'), ('burner.dPqP', 'DESIGN.burner.dPqP'),
            ('bld3.cool3:frac_W', 'DESIGN.bld3.cool3:frac_W'), ('bld3.cool4:frac_W', 'DESIGN.bld3.cool4:frac_W'), ('duct43.dPqP', 'DESIGN.duct43.dPqP'),
            ('itduct.dPqP', 'DESIGN.itduct.dPqP'), ('duct12.dPqP', 'DESIGN.duct12.dPqP'), ('nozzle.Cv', 'DESIGN.nozzle.Cv'), 
            ('hpt.cool3:frac_P', 'DESIGN.hpt.cool3:frac_P'), ('hpt.cool4:frac_P', 'DESIGN.hpt.cool4:frac_P'), ('lpt.cool1:frac_P', 'DESIGN.lpt.cool1:frac_P'),
            ('lpt.cool2:frac_P', 'DESIGN.lpt.cool2:frac_P')])

        self.prob.model.pyc_connect_des_od('lpc.s_PR', 'lpc.s_PR')
        self.prob.model.pyc_connect_des_od('lpc.s_Wc', 'lpc.s_Wc')
        self.prob.model.pyc_connect_des_od('lpc.s_eff', 'lpc.s_eff')
        self.prob.model.pyc_connect_des_od('lpc.s_Nc', 'lpc.s_Nc')
        self.prob.model.pyc_connect_des_od('hpc_axi.s_PR', 'hpc_axi.s_PR')
        self.prob.model.pyc_connect_des_od('hpc_axi.s_Wc', 'hpc_axi.s_Wc')
        self.prob.model.pyc_connect_des_od('hpc_axi.s_eff', 'hpc_axi.s_eff')
        self.prob.model.pyc_connect_des_od('hpc_axi.s_Nc', 'hpc_axi.s_Nc')
        self.prob.model.pyc_connect_des_od('hpc_centri.s_PR', 'hpc_centri.s_PR')
        self.prob.model.pyc_connect_des_od('hpc_centri.s_Wc', 'hpc_centri.s_Wc')
        self.prob.model.pyc_connect_des_od('hpc_centri.s_eff', 'hpc_centri.s_eff')
        self.prob.model.pyc_connect_des_od('hpc_centri.s_Nc', 'hpc_centri.s_Nc')
        self.prob.model.pyc_connect_des_od('hpt.s_PR', 'hpt.s_PR')
        self.prob.model.pyc_connect_des_od('hpt.s_Wp', 'hpt.s_Wp')
        self.prob.model.pyc_connect_des_od('hpt.s_eff', 'hpt.s_eff')
        self.prob.model.pyc_connect_des_od('hpt.s_Np', 'hpt.s_Np')
        self.prob.model.pyc_connect_des_od('lpt.s_PR', 'lpt.s_PR')
        self.prob.model.pyc_connect_des_od('lpt.s_Wp', 'lpt.s_Wp')
        self.prob.model.pyc_connect_des_od('lpt.s_eff', 'lpt.s_eff')
        self.prob.model.pyc_connect_des_od('lpt.s_Np', 'lpt.s_Np')
        self.prob.model.pyc_connect_des_od('pt.s_PR', 'pt.s_PR')
        self.prob.model.pyc_connect_des_od('pt.s_Wp', 'pt.s_Wp')
        self.prob.model.pyc_connect_des_od('pt.s_eff', 'pt.s_eff')
        self.prob.model.pyc_connect_des_od('pt.s_Np', 'pt.s_Np')

        self.prob.model.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
        self.prob.model.pyc_connect_des_od('duct1.Fl_O:stat:area', 'duct1.area')
        self.prob.model.pyc_connect_des_od('lpc.Fl_O:stat:area', 'lpc.area')
        self.prob.model.pyc_connect_des_od('icduct.Fl_O:stat:area', 'icduct.area')
        self.prob.model.pyc_connect_des_od('hpc_axi.Fl_O:stat:area', 'hpc_axi.area')
        self.prob.model.pyc_connect_des_od('bld25.Fl_O:stat:area', 'bld25.area')
        self.prob.model.pyc_connect_des_od('hpc_centri.Fl_O:stat:area', 'hpc_centri.area')
        self.prob.model.pyc_connect_des_od('bld3.Fl_O:stat:area', 'bld3.area')
        self.prob.model.pyc_connect_des_od('burner.Fl_O:stat:area', 'burner.area')
        self.prob.model.pyc_connect_des_od('hpt.Fl_O:stat:area', 'hpt.area')
        self.prob.model.pyc_connect_des_od('duct43.Fl_O:stat:area', 'duct43.area')
        self.prob.model.pyc_connect_des_od('lpt.Fl_O:stat:area', 'lpt.area')
        self.prob.model.pyc_connect_des_od('itduct.Fl_O:stat:area', 'itduct.area')
        self.prob.model.pyc_connect_des_od('pt.Fl_O:stat:area', 'pt.area')
        self.prob.model.pyc_connect_des_od('duct12.Fl_O:stat:area', 'duct12.area')
        self.prob.model.pyc_connect_des_od('nozzle.Throat:stat:area','balance.rhs:W')


        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.setup(check=False)

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

        self.prob.set_val('OD.balance.rhs:FAR', 1600.0, units='hp')
        self.prob.set_val('OD.LP_Nmech', 12750.0, units='rpm')
        self.prob.set_val('OD.fc.alt', 28000, units='ft')
        self.prob.set_val('OD.fc.MN', .5)

        self.prob['DESIGN.balance.FAR'] = 0.02261
        self.prob['DESIGN.balance.W'] = 10.76
        self.prob['DESIGN.balance.pt_PR'] = 4.939
        self.prob['DESIGN.balance.lpt_PR'] = 1.979
        self.prob['DESIGN.balance.hpt_PR'] = 4.236
        self.prob['DESIGN.fc.balance.Pt'] = 5.666
        self.prob['DESIGN.fc.balance.Tt'] = 440.0

        self.prob['OD.balance.FAR'] = 0.02135
        self.prob['OD.balance.W'] = 10.775
        self.prob['OD.balance.HP_Nmech'] = 14800.000
        self.prob['OD.balance.IP_Nmech'] = 12000.000
        self.prob['OD.hpt.PR'] = 4.233
        self.prob['OD.lpt.PR'] = 1.979
        self.prob['OD.pt.PR'] = 4.919
        self.prob['OD.fc.balance.Pt'] = 5.666
        self.prob['OD.fc.balance.Tt'] = 440.0
        self.prob['OD.nozzle.PR'] = 1.1

    def benchmark_case1(self):
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

        reg_data = 10.235
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 37.711
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.020230
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 10.235
        pyc = self.prob['OD.balance.W'][0]
        print('HPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 11557.
        pyc = self.prob['OD.balance.IP_Nmech'][0]
        print('LPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 14620.
        pyc = self.prob['OD.balance.HP_Nmech'][0]
        print('PT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.35259
        pyc = self.prob['OD.nozzle.Fl_O:stat:MN'][0]
        print('Nozz MN:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.31738
        pyc = self.prob['OD.perf.PSFC'][0]
        print('PSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1346.0
        pyc = self.prob['OD.duct6.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

if __name__ == "__main__":
    unittest.main()
