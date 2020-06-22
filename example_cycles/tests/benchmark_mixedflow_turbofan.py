import numpy as np
import unittest
import os

from openmdao.api import Problem, Group
import pycycle.api as pyc
from openmdao.utils.assert_utils import assert_rel_error

from example_cycles.mixedflow_turbofan import MixedFlowTurbofan

class MixedFlowTurbofanTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = prob = Problem()

        self.prob.model = pyc.MPCycle()

        prob.model.pyc_add_pnt('DESIGN', MixedFlowTurbofan(design=True), promotes=['balance.rhs:FAR_core', 'balance.rhs:FAR_ab', 
            'hp_shaft.HPX'])

        self.od_pts = ['OD0',]

        for i,pt in enumerate(self.od_pts):
            prob.model.pyc_add_pnt(pt, MixedFlowTurbofan(design=False), promotes=['balance.rhs:FAR_core', 'balance.rhs:FAR_ab', 
                ('inlet.ram_recovery', 'DESIGN.inlet.ram_recovery'), ('inlet_duct.dPqP', 'DESIGN.inlet_duct.dPqP'),
                ('splitter_core_duct.dPqP', 'DESIGN.splitter_core_duct.dPqP'), ('lpc_duct.dPqP', 'DESIGN.lpc_duct.dPqP'),
                ('burner.dPqP', 'DESIGN.burner.dPqP'), ('hpt_duct.dPqP', 'DESIGN.hpt_duct.dPqP'), ('lpt_duct.dPqP', 'DESIGN.lpt_duct.dPqP'),
                ('bypass_duct.dPqP', 'DESIGN.bypass_duct.dPqP'), ('mixer_duct.dPqP', 'DESIGN.mixer_duct.dPqP'), 
                ('mixed_nozz.Cfg', 'DESIGN.mixed_nozz.Cfg'), ('afterburner.dPqP', 'DESIGN.afterburner.dPqP'), 
                ('hpc.cool1:frac_W', 'DESIGN.hpc.cool1:frac_W'), ('hpc.cool1:frac_P', 'DESIGN.hpc.cool1:frac_P'),
                ('hpc.cool1:frac_work', 'DESIGN.hpc.cool1:frac_work'), 'hp_shaft.HPX', ('bld3.cool3:frac_W', 'DESIGN.bld3.cool3:frac_W'),
                ('hpt.cool3:frac_P', 'DESIGN.hpt.cool3:frac_P'), ('lpt.cool1:frac_P', 'DESIGN.lpt.cool1:frac_P')])

            # map scalars
        prob.model.pyc_connect_des_od('fan.s_PR', 'fan.s_PR')
        prob.model.pyc_connect_des_od('fan.s_Wc', 'fan.s_Wc')
        prob.model.pyc_connect_des_od('fan.s_eff', 'fan.s_eff')
        prob.model.pyc_connect_des_od('fan.s_Nc', 'fan.s_Nc')
        prob.model.pyc_connect_des_od('lpc.s_PR', 'lpc.s_PR')
        prob.model.pyc_connect_des_od('lpc.s_Wc', 'lpc.s_Wc')
        prob.model.pyc_connect_des_od('lpc.s_eff', 'lpc.s_eff')
        prob.model.pyc_connect_des_od('lpc.s_Nc', 'lpc.s_Nc')
        prob.model.pyc_connect_des_od('hpc.s_PR', 'hpc.s_PR')
        prob.model.pyc_connect_des_od('hpc.s_Wc', 'hpc.s_Wc')
        prob.model.pyc_connect_des_od('hpc.s_eff', 'hpc.s_eff')
        prob.model.pyc_connect_des_od('hpc.s_Nc', 'hpc.s_Nc')
        prob.model.pyc_connect_des_od('hpt.s_PR', 'hpt.s_PR')
        prob.model.pyc_connect_des_od('hpt.s_Wp', 'hpt.s_Wp')
        prob.model.pyc_connect_des_od('hpt.s_eff', 'hpt.s_eff')
        prob.model.pyc_connect_des_od('hpt.s_Np', 'hpt.s_Np')
        prob.model.pyc_connect_des_od('lpt.s_PR', 'lpt.s_PR')
        prob.model.pyc_connect_des_od('lpt.s_Wp', 'lpt.s_Wp')
        prob.model.pyc_connect_des_od('lpt.s_eff', 'lpt.s_eff')
        prob.model.pyc_connect_des_od('lpt.s_Np', 'lpt.s_Np')

            # flow areas
        prob.model.pyc_connect_des_od('mixed_nozz.Throat:stat:area', 'balance.rhs:W')

        prob.model.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
        prob.model.pyc_connect_des_od('fan.Fl_O:stat:area', 'fan.area')
        prob.model.pyc_connect_des_od('splitter.Fl_O1:stat:area', 'splitter.area1')
        prob.model.pyc_connect_des_od('splitter.Fl_O2:stat:area', 'splitter.area2')
        prob.model.pyc_connect_des_od('splitter_core_duct.Fl_O:stat:area', 'splitter_core_duct.area')
        prob.model.pyc_connect_des_od('lpc.Fl_O:stat:area', 'lpc.area')
        prob.model.pyc_connect_des_od('lpc_duct.Fl_O:stat:area', 'lpc_duct.area')
        prob.model.pyc_connect_des_od('hpc.Fl_O:stat:area', 'hpc.area')
        prob.model.pyc_connect_des_od('bld3.Fl_O:stat:area', 'bld3.area')
        prob.model.pyc_connect_des_od('burner.Fl_O:stat:area', 'burner.area')
        prob.model.pyc_connect_des_od('hpt.Fl_O:stat:area', 'hpt.area')
        prob.model.pyc_connect_des_od('hpt_duct.Fl_O:stat:area', 'hpt_duct.area')
        prob.model.pyc_connect_des_od('lpt.Fl_O:stat:area', 'lpt.area')
        prob.model.pyc_connect_des_od('lpt_duct.Fl_O:stat:area', 'lpt_duct.area')
        prob.model.pyc_connect_des_od('bypass_duct.Fl_O:stat:area', 'bypass_duct.area')
        prob.model.pyc_connect_des_od('mixer.Fl_O:stat:area', 'mixer.area')
        prob.model.pyc_connect_des_od('mixer.Fl_I1_calc:stat:area', 'mixer.Fl_I1_stat_calc.area')
        prob.model.pyc_connect_des_od('mixer_duct.Fl_O:stat:area', 'mixer_duct.area')
        prob.model.pyc_connect_des_od('afterburner.Fl_O:stat:area', 'afterburner.area')

        prob.setup(check=False)

        #design variables
        self.prob.set_val('DESIGN.fc.alt', 35000., units='ft') #DV
        self.prob.set_val('DESIGN.fc.MN', 0.8) #DV
        self.prob.set_val('DESIGN.balance.rhs:FAR_core', 3200, units='degR')
        self.prob.set_val('DESIGN.balance.rhs:FAR_ab', 3400, units='degR')
        self.prob.set_val('DESIGN.balance.rhs:W', 5500.0, units='lbf')
        self.prob.set_val('DESIGN.balance.rhs:BPR', 1.05 ,units=None) # defined as 1 over 2
        self.prob.set_val('DESIGN.fan.PR', 3.3) #ADV
        self.prob.set_val('DESIGN.lpc.PR', 1.935)
        self.prob.set_val('DESIGN.hpc.PR', 4.9)

        # #element parameters
        self.prob.set_val('DESIGN.inlet.ram_recovery', 0.9990) 
        self.prob.set_val('DESIGN.inlet.MN', 0.751)

        self.prob.set_val('DESIGN.inlet_duct.dPqP', 0.0107)
        self.prob.set_val('DESIGN.inlet_duct.MN', 0.4463)


        self.prob.set_val('DESIGN.fan.eff', 0.8948)
        self.prob.set_val('DESIGN.fan.MN', 0.4578)

        # #self.prob.set_val('DESIGN.splitter.BPR', 5.105) not needed for mixed flow turbofan. balanced based on mixer total pressure ratio
        self.prob.set_val('DESIGN.splitter.MN1', 0.3104)
        self.prob.set_val('DESIGN.splitter.MN2', 0.4518)

        self.prob.set_val('DESIGN.splitter_core_duct.dPqP', 0.0048)
        self.prob.set_val('DESIGN.splitter_core_duct.MN', 0.3121)

        self.prob.set_val('DESIGN.lpc.eff', 0.9243)
        self.prob.set_val('DESIGN.lpc.MN', 0.3059)

        self.prob.set_val('DESIGN.lpc_duct.dPqP', 0.0101)
        self.prob.set_val('DESIGN.lpc_duct.MN', 0.3563)

        self.prob.set_val('DESIGN.hpc.eff', 0.8707)
        self.prob.set_val('DESIGN.hpc.MN', 0.2442)

        self.prob.set_val('DESIGN.bld3.MN', 0.3000)

        self.prob.set_val('DESIGN.burner.dPqP', 0.0540)
        self.prob.set_val('DESIGN.burner.MN', 0.1025)

        self.prob.set_val('DESIGN.hpt.eff', 0.8888)
        self.prob.set_val('DESIGN.hpt.MN', 0.3650)

        self.prob.set_val('DESIGN.hpt_duct.dPqP', 0.0051)
        self.prob.set_val('DESIGN.hpt_duct.MN', 0.3063)

        self.prob.set_val('DESIGN.lpt.eff', 0.8996)
        self.prob.set_val('DESIGN.lpt.MN', 0.4127)

        self.prob.set_val('DESIGN.lpt_duct.dPqP', 0.0107)
        self.prob.set_val('DESIGN.lpt_duct.MN', 0.4463)

        self.prob.set_val('DESIGN.bypass_duct.dPqP', 0.0107)
        self.prob.set_val('DESIGN.bypass_duct.MN', 0.4463)

        self.prob.set_val('DESIGN.mixer_duct.dPqP', 0.0107)
        self.prob.set_val('DESIGN.mixer_duct.MN', 0.4463)

        self.prob.set_val('DESIGN.afterburner.dPqP', 0.0540)
        self.prob.set_val('DESIGN.afterburner.MN', 0.1025)

        self.prob.set_val('DESIGN.mixed_nozz.Cfg', 0.9933)

        self.prob.set_val('DESIGN.LP_Nmech', 4666.1, units='rpm')
        self.prob.set_val('DESIGN.HP_Nmech', 14705.7, units='rpm')
        self.prob.set_val('DESIGN.hp_shaft.HPX', 250.0, units='hp')

        self.prob.set_val('DESIGN.hpc.cool1:frac_W', 0.050708)
        self.prob.set_val('DESIGN.hpc.cool1:frac_P', 0.5)
        self.prob.set_val('DESIGN.hpc.cool1:frac_work', 0.5)

        self.prob.set_val('DESIGN.bld3.cool3:frac_W', 0.067214)

        self.prob.set_val('DESIGN.hpt.cool3:frac_P', 1.0)
        self.prob.set_val('DESIGN.lpt.cool1:frac_P', 1.0)

        # ####################
        # # OFF DESIGN CASES
        # ####################

        od_alts = [35000,]
        od_MNs = [0.8, ]

        for i,pt in enumerate(self.od_pts):
            self.prob.set_val(pt+'.fc.alt', od_alts[i], units='ft')
            self.prob.set_val(pt+'.fc.MN', od_MNs[i])

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)


    def benchmark_run_des(self):
        ''' Runs the design point and an off design point to make sure they match perfectly '''
        prob = self.prob

        # initial guesses
        self.prob['DESIGN.balance.FAR_core'] = 0.025
        self.prob['DESIGN.balance.FAR_ab'] = 0.025
        self.prob['DESIGN.balance.BPR'] = 1.0
        self.prob['DESIGN.balance.W'] = 100.
        self.prob['DESIGN.balance.lpt_PR'] = 3.5
        self.prob['DESIGN.balance.hpt_PR'] = 2.5
        self.prob['DESIGN.fc.balance.Pt'] = 5.2
        self.prob['DESIGN.fc.balance.Tt'] = 440.0
        self.prob['DESIGN.mixer.balance.P_tot']=100

        for pt in self.od_pts:
            self.prob[pt+'.balance.FAR_core'] = 0.031
            self.prob[pt+'.balance.FAR_ab'] = 0.038
            self.prob[pt+'.balance.BPR'] = 2.2
            self.prob[pt+'.balance.W'] = 60

            # really sensitive to these initial guesses
            self.prob[pt+'.balance.HP_Nmech'] = 15000
            self.prob[pt+'.balance.LP_Nmech'] = 5000

            self.prob[pt+'.fc.balance.Pt'] = 5.2
            self.prob[pt+'.fc.balance.Tt'] = 440.0
            self.prob[pt+'.mixer.balance.P_tot']= 100
            self.prob[pt+'.hpt.PR'] = 2.5
            self.prob[pt+'.lpt.PR'] = 3.5
            self.prob[pt+'.fan.map.RlineMap'] = 2.0
            self.prob[pt+'.lpc.map.RlineMap'] = 2.0
            self.prob[pt+'.hpc.map.RlineMap'] = 2.0

        self.prob.run_model()

        tol = 1e-5

        reg_data = 53.83467876114857
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.inlet.Fl_O:stat:W'][0]
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.0311108
        pyc = self.prob['DESIGN.balance.FAR_core'][0]
        print('Main FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.balance.FAR_core'][0]
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.038716210473225536
        pyc = self.prob['DESIGN.balance.FAR_ab'][0]
        print('Main FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.balance.FAR_ab'][0]
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 2.0430265465465354
        pyc = self.prob['DESIGN.balance.hpt_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.hpt.PR'][0]
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4.098132533864145
        pyc = self.prob['DESIGN.balance.lpt_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.lpt.PR'][0]
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 6802.813118292415
        pyc = self.prob['DESIGN.mixed_nozz.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.mixed_nozz.Fg'][0]
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1287.084732
        pyc = self.prob['DESIGN.hpc.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)
        pyc = self.prob['OD0.hpc.Fl_O:tot:T'][0]
        assert_rel_error(self, pyc, reg_data, tol)

if __name__ == "__main__":
    unittest.main()

