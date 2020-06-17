import numpy as np
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error

from example_cycles.high_bypass_turbofan import HBTF


class CFM56OffdesignTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = om.Problem()

        self.prob.model.add_subsystem('OD', HBTF(design=False))

        self.prob.setup(check=False)
        self.prob.final_setup()

        self.prob.set_val('OD.fc.MN', 0.8)
        self.prob.set_val('OD.fc.alt', 35000.0, units='ft')
        self.prob.set_val('OD.balance.rhs:FAR', 5500.0, units='lbf')  # 8950.0
        self.prob.set_val('OD.fc.dTs', 0.0, units='degR')
        self.prob.set_val('OD.hpc.cust:frac_W', 0.0445)

        self.prob.set_val('OD.inlet.ram_recovery', 0.9990)
        self.prob.set_val('OD.duct4.dPqP', 0.0048)
        self.prob.set_val('OD.duct6.dPqP', 0.0101)
        self.prob.set_val('OD.burner.dPqP', 0.0540)
        self.prob.set_val('OD.duct11.dPqP', 0.0051)
        self.prob.set_val('OD.duct13.dPqP', 0.0107)
        self.prob.set_val('OD.core_nozz.Cv', 0.9933)
        self.prob.set_val('OD.byp_bld.bypBld:frac_W', 0.005)
        self.prob.set_val('OD.duct15.dPqP', 0.0149)
        self.prob.set_val('OD.byp_nozz.Cv', 0.9939)
        self.prob.set_val('OD.hp_shaft.HPX', 250.0, units='hp')

        self.prob.set_val('OD.hpc.cool1:frac_W', 0.050708)
        self.prob.set_val('OD.hpc.cool1:frac_P', 0.5)
        self.prob.set_val('OD.hpc.cool1:frac_work', 0.5)
        self.prob.set_val('OD.hpc.cool2:frac_W', 0.020274) 
        self.prob.set_val('OD.hpc.cool2:frac_P', 0.55)
        self.prob.set_val('OD.hpc.cool2:frac_work', 0.5)
        self.prob.set_val('OD.bld3.cool3:frac_W', 0.067214)
        self.prob.set_val('OD.bld3.cool4:frac_W', 0.101256)
        self.prob.set_val('OD.hpc.cust:frac_P', 0.5)
        self.prob.set_val('OD.hpc.cust:frac_work', 0.5)
        self.prob.set_val('OD.hpt.cool3:frac_P', 1.0)
        self.prob.set_val('OD.hpt.cool4:frac_P', 0.0)
        self.prob.set_val('OD.lpt.cool1:frac_P', 1.0)
        self.prob.set_val('OD.lpt.cool2:frac_P', 0.0)
        
        self.prob.set_val('OD.fan.s_PR', 0.999912416431)
        self.prob.set_val('OD.fan.s_Wc', 1.03246959658)
        self.prob.set_val('OD.fan.s_eff', 1.00013412617)
        self.prob.set_val('OD.fan.s_Nc', 5091.84571411)
        self.prob.set_val('OD.lpc.s_PR', 1.0)
        self.prob.set_val('OD.lpc.s_Wc', 1.00411122011)
        self.prob.set_val('OD.lpc.s_eff', 0.999972953236)
        self.prob.set_val('OD.lpc.s_Nc', 4640.80978341)
        self.prob.set_val('OD.hpc.s_PR', 0.999352552331)
        self.prob.set_val('OD.hpc.s_Wc', 1.02817130922)
        self.prob.set_val('OD.hpc.s_eff', 1.00007580683)
        self.prob.set_val('OD.hpc.s_Nc', 13544.2035253)
        self.prob.set_val('OD.hpt.s_PR', 0.524557693866)
        self.prob.set_val('OD.hpt.s_Wp', 1.39329803688)
        self.prob.set_val('OD.hpt.s_eff', 0.987775061125)
        self.prob.set_val('OD.hpt.s_Np', 2.75125333383)
        self.prob.set_val('OD.lpt.s_PR', 0.673736258118)
        self.prob.set_val('OD.lpt.s_Wp', 1.48034371393)
        self.prob.set_val('OD.lpt.s_eff', 0.974542303109)
        self.prob.set_val('OD.lpt.s_Np', 1.03027097635)

        self.prob.set_val('OD.balance.rhs:W', 397.755002537, units='inch**2')
        self.prob.set_val('OD.balance.rhs:BPR', 1316.25610748, units='inch**2')

        self.prob.set_val('OD.inlet.area', 2566.76100868, units='inch**2')
        self.prob.set_val('OD.fan.area', 2228.37737592, units='inch**2')
        self.prob.set_val('OD.splitter.area1', 504.011122272, units='inch**2')
        self.prob.set_val('OD.splitter.area2', 1882.18932965, units='inch**2')
        self.prob.set_val('OD.duct4.area', 503.997116848, units='inch**2')
        self.prob.set_val('OD.lpc.area', 293.579637404, units='inch**2')
        self.prob.set_val('OD.duct6.area', 259.649232657, units='inch**2') 
        self.prob.set_val('OD.hpc.area', 49.0540725574, units='inch**2')
        self.prob.set_val('OD.bld3.area', 33.7913500831, units='inch**2') 
        self.prob.set_val('OD.burner.area', 157.7233536, units='inch**2')
        self.prob.set_val('OD.hpt.area', 172.74350706, units='inch**2')
        self.prob.set_val('OD.duct11.area', 202.354676631, units='inch**2')
        self.prob.set_val('OD.lpt.area', 613.494988147, units='inch**2')
        self.prob.set_val('OD.duct13.area', 582.849448774, units='inch**2')
        self.prob.set_val('OD.byp_bld.area', 1882.04141644, units='inch**2')
        self.prob.set_val('OD.duct15.area', 1878.67377328, units='inch**2')


        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)

    def benchmark_case1(self):
        # ADP Point
        np.seterr(divide='raise')

        self.prob['OD.balance.FAR'] = 0.02467
        self.prob['OD.balance.W'] = 320.931
        self.prob['OD.balance.BPR'] = 5.105
        self.prob['OD.balance.lp_Nmech'] = 4666.1
        self.prob['OD.balance.hp_Nmech'] = 14705.7
        self.prob['OD.fc.balance.Pt'] = 5.2
        self.prob['OD.fc.balance.Tt'] = 440.0
        self.prob['OD.hpt.PR'] = 3.6200
        self.prob['OD.lpt.PR'] = 4.3645
        self.prob['OD.fan.map.RlineMap'] = 2.0
        self.prob['OD.lpc.map.RlineMap'] = 2.0
        self.prob['OD.hpc.map.RlineMap'] = 2.0
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 321.251
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 30.094
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.02491
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 14705.7
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4666.1
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 13274.4
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.63101
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1276.48
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 5.105
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

    def benchmark_case2(self):
        # TOC Point
        np.seterr(divide='raise')

        self.prob['OD.balance.FAR'] = 0.02467
        self.prob['OD.balance.W'] = 320.931
        self.prob['OD.balance.BPR'] = 5.105
        self.prob['OD.balance.lp_Nmech'] = 4666.1
        self.prob['OD.balance.hp_Nmech'] = 14705.7
        self.prob['OD.fc.balance.Pt'] = 5.2
        self.prob['OD.fc.balance.Tt'] = 440.0
        self.prob['OD.hpt.PR'] = 3.6200
        self.prob['OD.lpt.PR'] = 4.3645
        self.prob['OD.fan.map.RlineMap'] = 2.0
        self.prob['OD.lpc.map.RlineMap'] = 2.0
        self.prob['OD.hpc.map.RlineMap'] = 2.0

        self.prob['OD.fc.MN'] = 0.8
        self.prob['OD.fc.alt'] = 35000.0
        self.prob['OD.balance.rhs:FAR'] = 5970.0
        self.prob['OD.fc.dTs'] = 0.0
        self.prob['OD.hpc.cust:frac_W'] = 0.0422
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 327.265
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 32.415
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.02616
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 14952.3
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4933.4
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 13889.9
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.64539
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1317.31
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4.898
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

    def benchmark_case3(self):
        # RTO Point
        np.seterr(divide='raise')

        self.prob['OD.balance.FAR'] = 0.03165
        self.prob['OD.balance.W'] = 810.83
        self.prob['OD.balance.BPR'] = 5.1053
        self.prob['OD.balance.lp_Nmech'] = 4975.9
        self.prob['OD.balance.hp_Nmech'] = 16230.1
        self.prob['OD.fc.balance.Pt'] = 15.349
        self.prob['OD.fc.balance.Tt'] = 552.49
        self.prob['OD.hpt.PR'] = 3.591
        self.prob['OD.lpt.PR'] = 4.173
        self.prob['OD.fan.map.RlineMap'] = 2.0
        self.prob['OD.lpc.map.RlineMap'] = 2.0
        self.prob['OD.hpc.map.RlineMap'] = 2.0

        self.prob['OD.fc.MN'] = 0.25
        self.prob['OD.fc.alt'] = 0.0
        self.prob['OD.balance.rhs:FAR'] = 22590.0
        self.prob['OD.fc.dTs'] = 27.0
        self.prob['OD.hpc.cust:frac_W'] = 0.0177
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 825.049
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 28.998
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.02975
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 16222.1
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 5050
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 29930.8
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.47488
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1536.94
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 5.243
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

    def benchmark_case4(self):
        # SLS Point
        np.seterr(divide='raise')

        self.prob['OD.balance.FAR'] = 0.03114
        self.prob['OD.balance.W'] = 771.34
        self.prob['OD.balance.BPR'] = 5.0805
        self.prob['OD.balance.lp_Nmech'] = 4912.7
        self.prob['OD.balance.hp_Nmech'] = 16106.9
        self.prob['OD.fc.balance.Pt'] = 14.696
        self.prob['OD.fc.balance.Tt'] = 545.67
        self.prob['OD.hpt.PR'] = 3.595
        self.prob['OD.lpt.PR'] = 4.147
        self.prob['OD.fan.map.RlineMap'] = 2.0
        self.prob['OD.lpc.map.RlineMap'] = 2.0
        self.prob['OD.hpc.map.RlineMap'] = 2.0

        self.prob['OD.fc.MN'] = 0.00001
        self.prob['OD.fc.alt'] = 0.0
        self.prob['OD.balance.rhs:FAR'] = 27113.0
        self.prob['OD.fc.dTs'] = 27.0
        self.prob['OD.hpc.cust:frac_W'] = 0.0185
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 786.741
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 28.418
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.02912
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 16065.1
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4949.1
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 27113.3
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.36662
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1509.41
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 5.282
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

if __name__ == "__main__":
    unittest.main()
