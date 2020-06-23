import numpy as np
import unittest
import os

import openmdao.api as om
import pycycle.api as pyc
from openmdao.utils.assert_utils import assert_near_equal

from example_cycles.high_bypass_turbofan import HBTF


class CFM56TestCase(unittest.TestCase):

    def setUp(self):

        self.prob = om.Problem()

        self.prob.model = pyc.MPCycle()

        self.prob.model.pyc_add_pnt('DESIGN', HBTF(), promotes=['hp_shaft.HPX'])

        self.prob.model.pyc_add_pnt('OD', HBTF(design=False), promotes=[('inlet.ram_recovery', 'DESIGN.inlet.ram_recovery'), 
            ('duct4.dPqP', 'DESIGN.duct4.dPqP'), ('duct6.dPqP', 'DESIGN.duct6.dPqP'), ('burner.dPqP', 'DESIGN.burner.dPqP'),
            ('duct11.dPqP', 'DESIGN.duct11.dPqP'), ('duct13.dPqP', 'DESIGN.duct13.dPqP'), ('duct15.dPqP', 'DESIGN.duct15.dPqP'),
            ('core_nozz.Cv', 'DESIGN.core_nozz.Cv'), ('byp_bld.bypBld:frac_W', 'DESIGN.byp_bld.bypBld:frac_W'),
            ('byp_nozz.Cv', 'DESIGN.byp_nozz.Cv'), ('hpc.cool1:frac_W', 'DESIGN.hpc.cool1:frac_W'), ('hpc.cool1:frac_P', 'DESIGN.hpc.cool1:frac_P'),
            ('hpc.cool1:frac_work', 'DESIGN.hpc.cool1:frac_work'), ('hpc.cool2:frac_W', 'DESIGN.hpc.cool2:frac_W'), ('hpc.cool2:frac_P', 'DESIGN.hpc.cool2:frac_P'),
            ('hpc.cool2:frac_work', 'DESIGN.hpc.cool2:frac_work'), ('bld3.cool3:frac_W', 'DESIGN.bld3.cool3:frac_W'), ('bld3.cool4:frac_W', 'DESIGN.bld3.cool4:frac_W'),
            ('hpc.cust:frac_P', 'DESIGN.hpc.cust:frac_P'), ('hpc.cust:frac_work', 'DESIGN.hpc.cust:frac_work'), ('hpt.cool3:frac_P', 'DESIGN.hpt.cool3:frac_P'),
            ('hpt.cool4:frac_P', 'DESIGN.hpt.cool4:frac_P'), ('lpt.cool1:frac_P', 'DESIGN.lpt.cool1:frac_P'), ('lpt.cool2:frac_P', 'DESIGN.lpt.cool2:frac_P'), 'hp_shaft.HPX'])

        # Connect all DESIGN map scalars to the off design cases
        self.prob.model.pyc_connect_des_od('fan.s_PR', 'fan.s_PR')
        self.prob.model.pyc_connect_des_od('fan.s_Wc', 'fan.s_Wc')
        self.prob.model.pyc_connect_des_od('fan.s_eff', 'fan.s_eff')
        self.prob.model.pyc_connect_des_od('fan.s_Nc', 'fan.s_Nc')
        self.prob.model.pyc_connect_des_od('lpc.s_PR', 'lpc.s_PR')
        self.prob.model.pyc_connect_des_od('lpc.s_Wc', 'lpc.s_Wc')
        self.prob.model.pyc_connect_des_od('lpc.s_eff', 'lpc.s_eff')
        self.prob.model.pyc_connect_des_od('lpc.s_Nc', 'lpc.s_Nc')
        self.prob.model.pyc_connect_des_od('hpc.s_PR', 'hpc.s_PR')
        self.prob.model.pyc_connect_des_od('hpc.s_Wc', 'hpc.s_Wc')
        self.prob.model.pyc_connect_des_od('hpc.s_eff', 'hpc.s_eff')
        self.prob.model.pyc_connect_des_od('hpc.s_Nc', 'hpc.s_Nc')
        self.prob.model.pyc_connect_des_od('hpt.s_PR', 'hpt.s_PR')
        self.prob.model.pyc_connect_des_od('hpt.s_Wp', 'hpt.s_Wp')
        self.prob.model.pyc_connect_des_od('hpt.s_eff', 'hpt.s_eff')
        self.prob.model.pyc_connect_des_od('hpt.s_Np', 'hpt.s_Np')
        self.prob.model.pyc_connect_des_od('lpt.s_PR', 'lpt.s_PR')
        self.prob.model.pyc_connect_des_od('lpt.s_Wp', 'lpt.s_Wp')
        self.prob.model.pyc_connect_des_od('lpt.s_eff', 'lpt.s_eff')
        self.prob.model.pyc_connect_des_od('lpt.s_Np', 'lpt.s_Np')
        
        #Set up the RHS of the balances!
        self.prob.model.pyc_connect_des_od('core_nozz.Throat:stat:area','balance.rhs:W')
        self.prob.model.pyc_connect_des_od('byp_nozz.Throat:stat:area','balance.rhs:BPR')

        self.prob.model.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
        self.prob.model.pyc_connect_des_od('fan.Fl_O:stat:area', 'fan.area')
        self.prob.model.pyc_connect_des_od('splitter.Fl_O1:stat:area', 'splitter.area1')
        self.prob.model.pyc_connect_des_od('splitter.Fl_O2:stat:area', 'splitter.area2')
        self.prob.model.pyc_connect_des_od('duct4.Fl_O:stat:area', 'duct4.area')
        self.prob.model.pyc_connect_des_od('lpc.Fl_O:stat:area', 'lpc.area')
        self.prob.model.pyc_connect_des_od('duct6.Fl_O:stat:area', 'duct6.area')
        self.prob.model.pyc_connect_des_od('hpc.Fl_O:stat:area', 'hpc.area')
        self.prob.model.pyc_connect_des_od('bld3.Fl_O:stat:area', 'bld3.area')
        self.prob.model.pyc_connect_des_od('burner.Fl_O:stat:area', 'burner.area')
        self.prob.model.pyc_connect_des_od('hpt.Fl_O:stat:area', 'hpt.area')
        self.prob.model.pyc_connect_des_od('duct11.Fl_O:stat:area', 'duct11.area')
        self.prob.model.pyc_connect_des_od('lpt.Fl_O:stat:area', 'lpt.area')
        self.prob.model.pyc_connect_des_od('duct13.Fl_O:stat:area', 'duct13.area')
        self.prob.model.pyc_connect_des_od('byp_bld.Fl_O:stat:area', 'byp_bld.area')
        self.prob.model.pyc_connect_des_od('duct15.Fl_O:stat:area', 'duct15.area')

        self.prob.setup(check=False)

        self.prob.set_val('DESIGN.fc.alt', 35000., units='ft')
        self.prob.set_val('DESIGN.fc.MN', 0.8)
        self.prob.set_val('DESIGN.balance.rhs:FAR', 2857, units='degR')
        self.prob.set_val('DESIGN.balance.rhs:W', 5500.0, units='lbf')  
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
        self.prob.set_val('DESIGN.duct6.dPqP', 0.0101),
        self.prob.set_val('DESIGN.duct6.MN', 0.3563),
        self.prob.set_val('DESIGN.hpc.PR', 9.369),
        self.prob.set_val('DESIGN.hpc.eff', 0.8707),
        self.prob.set_val('DESIGN.hpc.MN', 0.2442),
        self.prob.set_val('DESIGN.bld3.MN', 0.3000)
        self.prob.set_val('DESIGN.burner.dPqP', 0.0540),
        self.prob.set_val('DESIGN.burner.MN', 0.1025),
        self.prob.set_val('DESIGN.hpt.eff', 0.8888),
        self.prob.set_val('DESIGN.hpt.MN', 0.3650),
        self.prob.set_val('DESIGN.duct11.dPqP', 0.0051),
        self.prob.set_val('DESIGN.duct11.MN', 0.3063),
        self.prob.set_val('DESIGN.lpt.eff', 0.8996),
        self.prob.set_val('DESIGN.lpt.MN', 0.4127),
        self.prob.set_val('DESIGN.duct13.dPqP', 0.0107),
        self.prob.set_val('DESIGN.duct13.MN', 0.4463),
        self.prob.set_val('DESIGN.core_nozz.Cv', 0.9933),
        self.prob.set_val('DESIGN.byp_bld.bypBld:frac_W', 0.005),
        self.prob.set_val('DESIGN.byp_bld.MN', 0.4489),
        self.prob.set_val('DESIGN.duct15.dPqP', 0.0149),
        self.prob.set_val('DESIGN.duct15.MN', 0.4589),
        self.prob.set_val('DESIGN.byp_nozz.Cv', 0.9939),
        self.prob.set_val('DESIGN.LP_Nmech', 4666.1, units='rpm'),
        self.prob.set_val('DESIGN.HP_Nmech', 14705.7, units='rpm'),
        self.prob.set_val('DESIGN.hp_shaft.HPX', 250.0, units='hp'),
        self.prob.set_val('DESIGN.hpc.cool1:frac_W', 0.050708),
        self.prob.set_val('DESIGN.hpc.cool1:frac_P', 0.5),
        self.prob.set_val('DESIGN.hpc.cool1:frac_work', 0.5),
        self.prob.set_val('DESIGN.hpc.cool2:frac_W', 0.020274),
        self.prob.set_val('DESIGN.hpc.cool2:frac_P', 0.55),
        self.prob.set_val('DESIGN.hpc.cool2:frac_work', 0.5),
        self.prob.set_val('DESIGN.bld3.cool3:frac_W', 0.067214),
        self.prob.set_val('DESIGN.bld3.cool4:frac_W', 0.101256),
        self.prob.set_val('DESIGN.hpc.cust:frac_W', 0.0445),
        self.prob.set_val('DESIGN.hpc.cust:frac_P', 0.5),
        self.prob.set_val('DESIGN.hpc.cust:frac_work', 0.5),
        self.prob.set_val('DESIGN.hpt.cool3:frac_P', 1.0),
        self.prob.set_val('DESIGN.hpt.cool4:frac_P', 0.0),
        self.prob.set_val('DESIGN.lpt.cool1:frac_P', 1.0),
        self.prob.set_val('DESIGN.lpt.cool2:frac_P', 0.0),

        self.prob.set_val('OD.fc.MN', 0.8)
        self.prob.set_val('OD.fc.alt', 35000.0, units='ft')
        self.prob.set_val('OD.balance.rhs:FAR', 5500.0, units='lbf')  # 8950.0
        self.prob.set_val('OD.fc.dTs', 0.0, units='degR')
        self.prob.set_val('OD.hpc.cust:frac_W', 0.0445)

        self.prob['DESIGN.balance.FAR'] = 0.025
        self.prob['DESIGN.balance.W'] = 316.0
        self.prob['DESIGN.balance.lpt_PR'] = 4.4
        self.prob['DESIGN.balance.hpt_PR'] = 3.6
        self.prob['DESIGN.fc.balance.Pt'] = 5.2
        self.prob['DESIGN.fc.balance.Tt'] = 440.0    

    def benchmark_case1(self):
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
        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.run_model()
        tol = 1e-3
        print()

        reg_data = 321.253
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 30.094
        pyc = self.prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.02491
        pyc = self.prob['DESIGN.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 3.6228
        pyc = self.prob['DESIGN.balance.hpt_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4.3687
        pyc = self.prob['DESIGN.balance.lpt_PR'][0]
        print('LPT PR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13274.4
        pyc = self.prob['DESIGN.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.63101
        pyc = self.prob['DESIGN.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1276.48
        pyc = self.prob['DESIGN.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 321.251
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 30.094
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.02491
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 14705.7
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4666.1
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13274.4
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.63101
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1276.48
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 5.105
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

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
        
        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 327.265
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 32.415
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.02616
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 14952.3
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4933.4
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 13889.9
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.64539
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1317.31
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4.898
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

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
        
        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 825.049
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 28.998
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.02975
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 16222.1
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 5050
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 29930.8
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.47488
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1536.94
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 5.243
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

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
        
        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 786.741
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 28.418
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.02912
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 16065.1
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 4949.1
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 27113.3
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 0.36662
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 1509.41
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        reg_data = 5.282
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_near_equal(pyc, reg_data, tol)

        print()

if __name__ == "__main__":
    unittest.main()
