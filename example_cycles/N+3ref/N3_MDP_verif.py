import numpy as np
import time
import pickle
from pprint import pprint

from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver, ArmijoGoldsteinLS, LinearBlockGS, pyOptSparseDriver
from openmdao.api import Problem, IndepVarComp, SqliteRecorder, CaseReader, BalanceComp, ScipyKrylov, PETScKrylov, ExecComp
from openmdao.utils.units import convert_units as cu

import pycycle.api as pyc

from N3ref import N3, viewer

prob = Problem()

prob.model = pyc.MPCycle()

des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=["*"])

des_vars.add_output('inlet:ram_recovery', 0.9980),
des_vars.add_output('fan:PRdes', 1.300),
des_vars.add_output('fan:effDes', 0.96888),
des_vars.add_output('fan:effPoly', 0.97),
des_vars.add_output('splitter:BPR', 23.7281), #23.9878######################
des_vars.add_output('duct2:dPqP', 0.0100),
des_vars.add_output('lpc:PRdes', 3.000),
des_vars.add_output('lpc:effDes', 0.889513),
des_vars.add_output('lpc:effPoly', 0.905),
des_vars.add_output('duct25:dPqP', 0.0150),
des_vars.add_output('hpc:PRdes', 14.103),
des_vars.add_output('OPR', 53.6332) #53.635)
des_vars.add_output('OPR_simple', 55.0)#############this isn't included in main file
des_vars.add_output('hpc:effDes', 0.847001),
des_vars.add_output('hpc:effPoly', 0.89),
des_vars.add_output('hpt:effDes', 0.922649),
des_vars.add_output('hpt:effPoly', 0.91),
des_vars.add_output('duct45:dPqP', 0.0050),
des_vars.add_output('lpt:effDes', 0.940104),
des_vars.add_output('lpt:effPoly', 0.92),
des_vars.add_output('duct5:dPqP', 0.0100),
des_vars.add_output('duct17:dPqP', 0.0150),
des_vars.add_output('fan_shaft:Nmech', 2184.5, units='rpm'),
des_vars.add_output('lp_shaft:Nmech', 6772.0, units='rpm'),
des_vars.add_output('hp_shaft:Nmech', 20871.0, units='rpm'),

# des_vars.add_output('hpc:bld_inlet:frac_work', 0.5),
# des_vars.add_output('hpc:bld_exit:frac_W', 0.02),
# des_vars.add_output('hpc:bld_exit:frac_P', 0.1465),
# des_vars.add_output('hpc:bld_exit:frac_work', 0.5),
# des_vars.add_output('hpc:cust:frac_W', 0.0),
# des_vars.add_output('hpc:cust:frac_P', 0.1465),
# des_vars.add_output('hpc:cust:frac_work', 0.35),
des_vars.add_output('bld3:bld_inlet:frac_W', 0.063660111), #different than NPSS due to Wref
des_vars.add_output('bld3:bld_exit:frac_W', 0.07037185), #different than NPSS due to Wref
# des_vars.add_output('hpt:bld_inlet:frac_P', 1.0),
# des_vars.add_output('hpt:bld_exit:frac_P', 0.0),
# des_vars.add_output('lpt:bld_inlet:frac_P', 1.0),
# des_vars.add_output('lpt:bld_exit:frac_P', 0.0),
# des_vars.add_output('bypBld:frac_W', 0.0),

des_vars.add_output('inlet:MN_out', 0.625),
des_vars.add_output('fan:MN_out', 0.45)
des_vars.add_output('splitter:MN_out1', 0.45)
des_vars.add_output('splitter:MN_out2', 0.45)
des_vars.add_output('duct2:MN_out', 0.45),
des_vars.add_output('lpc:MN_out', 0.45),
des_vars.add_output('bld25:MN_out', 0.45),
des_vars.add_output('duct25:MN_out', 0.45),
des_vars.add_output('hpc:MN_out', 0.30),
des_vars.add_output('bld3:MN_out', 0.30)
des_vars.add_output('burner:MN_out', 0.10),
des_vars.add_output('hpt:MN_out', 0.30),
des_vars.add_output('duct45:MN_out', 0.45),
des_vars.add_output('lpt:MN_out', 0.35),
des_vars.add_output('duct5:MN_out', 0.25),
des_vars.add_output('bypBld:MN_out', 0.45),
des_vars.add_output('duct17:MN_out', 0.45),

# POINT 1: Top-of-climb (TOC)
des_vars.add_output('TOC:alt', 35000., units='ft'),
des_vars.add_output('TOC:MN', 0.8),
des_vars.add_output('TOC:T4max', 3150.0, units='degR'),
des_vars.add_output('TOC:Fn_des', 6073.4, units='lbf'),
des_vars.add_output('TOC:ram_recovery', 0.9980),
des_vars.add_output('TR', 0.926470588)
##################main model also includes TOC:W here##########

# POINT 2: Rolling Takeoff (RTO)
des_vars.add_output('RTO:MN', 0.25),
des_vars.add_output('RTO:alt', 0.0, units='ft'),
des_vars.add_output('RTO:Fn_target', 22800.0, units='lbf'), #8950.0
des_vars.add_output('RTO:dTs', 27.0, units='degR')
des_vars.add_output('RTO:Ath', 5532.3, units='inch**2')
des_vars.add_output('RTO:RlineMap', 1.75)
des_vars.add_output('RTO:T4max', 3400.0, units='degR')
des_vars.add_output('RTO:W', 1916.13, units='lbm/s')
des_vars.add_output('RTO:ram_recovery', 0.9970),
des_vars.add_output('RTO:duct2:dPqP', 0.0073)
des_vars.add_output('RTO:duct25:dPqP', 0.0138)
des_vars.add_output('RTO:duct45:dPqP', 0.0051)
des_vars.add_output('RTO:duct5:dPqP', 0.0058)
des_vars.add_output('RTO:duct17:dPqP', 0.0132)

# POINT 3: Sea-Level Static (SLS)
des_vars.add_output('SLS:MN', 0.001),#################################
des_vars.add_output('SLS:alt', 0.0, units='ft'),
des_vars.add_output('SLS:Fn_target', 28620.9, units='lbf'), #8950.0############################# 
des_vars.add_output('SLS:dTs', 27.0, units='degR')
des_vars.add_output('SLS:Ath', 6315.6, units='inch**2')
des_vars.add_output('SLS:RlineMap', 1.75)
des_vars.add_output('SLS:ram_recovery', 0.9950),
des_vars.add_output('SLS:duct2:dPqP', 0.0058)
des_vars.add_output('SLS:duct25:dPqP', 0.0126)
des_vars.add_output('SLS:duct45:dPqP', 0.0052)
des_vars.add_output('SLS:duct5:dPqP', 0.0043)
des_vars.add_output('SLS:duct17:dPqP', 0.0123)

# POINT 4: Cruise (CRZ)
des_vars.add_output('CRZ:MN', 0.8),
des_vars.add_output('CRZ:alt', 35000.0, units='ft'),
des_vars.add_output('CRZ:Fn_target', 5466.5, units='lbf'), #8950.0#######################################
des_vars.add_output('CRZ:dTs', 0.0, units='degR')
des_vars.add_output('CRZ:Ath', 4747.1, units='inch**2')
des_vars.add_output('CRZ:RlineMap', 1.9397)
des_vars.add_output('CRZ:ram_recovery', 0.9980),
des_vars.add_output('CRZ:duct2:dPqP', 0.0092)
des_vars.add_output('CRZ:duct25:dPqP', 0.0138)
des_vars.add_output('CRZ:duct45:dPqP', 0.0050)
des_vars.add_output('CRZ:duct5:dPqP', 0.0097)
des_vars.add_output('CRZ:duct17:dPqP', 0.0148)
des_vars.add_output('CRZ:VjetRatio', 1.40) #1.41038)##################################################


# TOC POINT (DESIGN)
prob.model.pyc_add_pnt('TOC', N3())

prob.model.connect('TOC:alt', 'TOC.fc.alt')
prob.model.connect('TOC:MN', 'TOC.fc.MN')

prob.model.connect('TOC:ram_recovery', 'TOC.inlet.ram_recovery')
prob.model.connect('fan:PRdes', ['TOC.fan.PR', 'TOC.opr_calc.FPR'])############main file doesn't include opr_calc
prob.model.connect('fan:effPoly', 'TOC.balance.rhs:fan_eff')
prob.model.connect('duct2:dPqP', 'TOC.duct2.dPqP')
prob.model.connect('lpc:PRdes', ['TOC.lpc.PR', 'TOC.opr_calc.LPCPR'])###############main file doesn't include opr_calc.LPCPR
prob.model.connect('lpc:effPoly', 'TOC.balance.rhs:lpc_eff')
prob.model.connect('duct25:dPqP', 'TOC.duct25.dPqP')
prob.model.connect('OPR_simple', 'TOC.balance.rhs:hpc_PR')###################main file has OPR_simple as OPR
prob.model.connect('hpt:effPoly', 'TOC.balance.rhs:hpt_eff')
prob.model.connect('duct45:dPqP', 'TOC.duct45.dPqP')
prob.model.connect('lpt:effPoly', 'TOC.balance.rhs:lpt_eff')
prob.model.connect('duct5:dPqP', 'TOC.duct5.dPqP')
#########################main file doesn't include ext_ratio.core_Cv
prob.model.connect('duct17:dPqP', 'TOC.duct17.dPqP')
#############################main file doesn't include ext_ratio.byp_Cv
prob.model.connect('fan_shaft:Nmech', 'TOC.Fan_Nmech')
prob.model.connect('lp_shaft:Nmech', 'TOC.LP_Nmech')
prob.model.connect('hp_shaft:Nmech', 'TOC.HP_Nmech')

# prob.model.connect('hpc:bld_inlet:frac_work', 'TOC.hpc.bld_inlet:frac_work')
# prob.model.connect('hpc:bld_exit:frac_W', 'TOC.hpc.bld_exit:frac_W')
# prob.model.connect('hpc:bld_exit:frac_P', 'TOC.hpc.bld_exit:frac_P')
# prob.model.connect('hpc:bld_exit:frac_work', 'TOC.hpc.bld_exit:frac_work')
# prob.model.connect('hpc:cust:frac_W', 'TOC.hpc.cust:frac_W')
# prob.model.connect('hpc:cust:frac_P', 'TOC.hpc.cust:frac_P')
# prob.model.connect('hpc:cust:frac_work', 'TOC.hpc.cust:frac_work')
# prob.model.connect('hpt:bld_inlet:frac_P', 'TOC.hpt.bld_inlet:frac_P')
# prob.model.connect('hpt:bld_exit:frac_P', 'TOC.hpt.bld_exit:frac_P')
# prob.model.connect('lpt:bld_inlet:frac_P', 'TOC.lpt.bld_inlet:frac_P')
# prob.model.connect('lpt:bld_exit:frac_P', 'TOC.lpt.bld_exit:frac_P')
# prob.model.connect('bypBld:frac_W', 'TOC.byp_bld.bypBld:frac_W')

prob.model.connect('inlet:MN_out', 'TOC.inlet.MN')
prob.model.connect('fan:MN_out', 'TOC.fan.MN')
prob.model.connect('splitter:MN_out1', 'TOC.splitter.MN1')
prob.model.connect('splitter:MN_out2', 'TOC.splitter.MN2')
prob.model.connect('duct2:MN_out', 'TOC.duct2.MN')
prob.model.connect('lpc:MN_out', 'TOC.lpc.MN')
prob.model.connect('bld25:MN_out', 'TOC.bld25.MN')
prob.model.connect('duct25:MN_out', 'TOC.duct25.MN')
prob.model.connect('hpc:MN_out', 'TOC.hpc.MN')
prob.model.connect('bld3:MN_out', 'TOC.bld3.MN')
prob.model.connect('burner:MN_out', 'TOC.burner.MN')
prob.model.connect('hpt:MN_out', 'TOC.hpt.MN')
prob.model.connect('duct45:MN_out', 'TOC.duct45.MN')
prob.model.connect('lpt:MN_out', 'TOC.lpt.MN')
prob.model.connect('duct5:MN_out', 'TOC.duct5.MN')
prob.model.connect('bypBld:MN_out', 'TOC.byp_bld.MN')
prob.model.connect('duct17:MN_out', 'TOC.duct17.MN')

# OTHER POINTS (OFF-DESIGN)
pts = ['RTO','SLS','CRZ']
OD_statics = True#################main file doesn't include this line#######


prob.model.connect('RTO:Fn_target', 'RTO.balance.rhs:FAR')

prob.model.pyc_add_pnt('RTO', N3(design=False, cooling=True))
prob.model.pyc_add_pnt('SLS', N3(design=False))
prob.model.pyc_add_pnt('CRZ', N3(design=False))


for pt in pts:
    # ODpt.nonlinear_solver.options['maxiter'] = 0

    prob.model.connect(pt+':alt', pt+'.fc.alt')
    prob.model.connect(pt+':MN', pt+'.fc.MN')
    prob.model.connect(pt+':dTs', pt+'.fc.dTs')
    prob.model.connect(pt+':RlineMap',pt+'.balance.rhs:BPR')
    prob.model.connect(pt+':ram_recovery', pt+'.inlet.ram_recovery')

    # prob.model.connect('hpc:bld_inlet:frac_work', pt+'.hpc.bld_inlet:frac_work')
    # prob.model.connect('hpc:bld_exit:frac_W', pt+'.hpc.bld_exit:frac_W')
    # prob.model.connect('hpc:bld_exit:frac_P', pt+'.hpc.bld_exit:frac_P')
    # prob.model.connect('hpc:bld_exit:frac_work', pt+'.hpc.bld_exit:frac_work')
    # prob.model.connect('hpc:cust:frac_W', pt+'.hpc.cust:frac_W')
    # prob.model.connect('hpc:cust:frac_P', pt+'.hpc.cust:frac_P')
    # prob.model.connect('hpc:cust:frac_work', pt+'.hpc.cust:frac_work')
    # prob.model.connect('hpt:bld_inlet:frac_P', pt+'.hpt.bld_inlet:frac_P')
    # prob.model.connect('hpt:bld_exit:frac_P', pt+'.hpt.bld_exit:frac_P')
    # prob.model.connect('lpt:bld_inlet:frac_P', pt+'.lpt.bld_inlet:frac_P')
    # prob.model.connect('lpt:bld_exit:frac_P', pt+'.lpt.bld_exit:frac_P')
    # prob.model.connect('bypBld:frac_W', pt+'.byp_bld.bypBld:frac_W')

prob.model.pyc_add_cycle_param('burner.dPqP', 0.0400),
prob.model.pyc_add_cycle_param('core_nozz.Cv', 0.9999),
prob.model.pyc_add_cycle_param('ext_ratio.core_Cv', 0.9999)###############The main file doesn't include this line
prob.model.pyc_add_cycle_param('byp_nozz.Cv', 0.9975),
prob.model.pyc_add_cycle_param('ext_ratio.byp_Cv', 0.9975)###############The main file doesn't include this line
prob.model.pyc_add_cycle_param('lp_shaft.fracLoss', 0.01),
prob.model.pyc_add_cycle_param('hp_shaft.HPX', 350.0, units='hp'),
prob.model.pyc_add_cycle_param('bld25.sbv:frac_W', 0.0),
prob.model.pyc_add_cycle_param('hpc.bld_inlet:frac_W', 0.0),
prob.model.pyc_add_cycle_param('hpc.bld_inlet:frac_P', 0.1465),
prob.model.pyc_add_cycle_param('hpc.bld_inlet:frac_work', 0.5),
prob.model.pyc_add_cycle_param('hpc.bld_exit:frac_W', 0.02),
prob.model.pyc_add_cycle_param('hpc.bld_exit:frac_P', 0.1465),
prob.model.pyc_add_cycle_param('hpc.bld_exit:frac_work', 0.5),
prob.model.pyc_add_cycle_param('hpc.cust:frac_W', 0.0),
prob.model.pyc_add_cycle_param('hpc.cust:frac_P', 0.1465),
prob.model.pyc_add_cycle_param('hpc.cust:frac_work', 0.35),
prob.model.pyc_add_cycle_param('hpt.bld_inlet:frac_P', 1.0),
prob.model.pyc_add_cycle_param('hpt.bld_exit:frac_P', 0.0),
prob.model.pyc_add_cycle_param('lpt.bld_inlet:frac_P', 1.0),
prob.model.pyc_add_cycle_param('lpt.bld_exit:frac_P', 0.0),
prob.model.pyc_add_cycle_param('byp_bld.bypBld:frac_W', 0.0),

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

prob.model.pyc_connect_des_od('gearbox.gear_ratio', 'gearbox.gear_ratio')
prob.model.pyc_connect_des_od('core_nozz.Throat:stat:area','balance.rhs:W')

prob.model.pyc_connect_des_od('duct2.s_dPqP', 'duct2.s_dPqP')
prob.model.pyc_connect_des_od('duct25.s_dPqP', 'duct25.s_dPqP')
prob.model.pyc_connect_des_od('duct45.s_dPqP', 'duct45.s_dPqP')
prob.model.pyc_connect_des_od('duct5.s_dPqP', 'duct5.s_dPqP')
prob.model.pyc_connect_des_od('duct17.s_dPqP', 'duct17.s_dPqP')

if OD_statics:#############################this section isn't in an if statement in the main file###############
    prob.model.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
    prob.model.pyc_connect_des_od('fan.Fl_O:stat:area', 'fan.area')
    prob.model.pyc_connect_des_od('splitter.Fl_O1:stat:area', 'splitter.area1')
    prob.model.pyc_connect_des_od('splitter.Fl_O2:stat:area', 'splitter.area2')
    prob.model.pyc_connect_des_od('duct2.Fl_O:stat:area', 'duct2.area')
    prob.model.pyc_connect_des_od('lpc.Fl_O:stat:area', 'lpc.area')
    prob.model.pyc_connect_des_od('bld25.Fl_O:stat:area', 'bld25.area')
    prob.model.pyc_connect_des_od('duct25.Fl_O:stat:area', 'duct25.area')
    prob.model.pyc_connect_des_od('hpc.Fl_O:stat:area', 'hpc.area')
    prob.model.pyc_connect_des_od('bld3.Fl_O:stat:area', 'bld3.area')
    prob.model.pyc_connect_des_od('burner.Fl_O:stat:area', 'burner.area')
    prob.model.pyc_connect_des_od('hpt.Fl_O:stat:area', 'hpt.area')
    prob.model.pyc_connect_des_od('duct45.Fl_O:stat:area', 'duct45.area')
    prob.model.pyc_connect_des_od('lpt.Fl_O:stat:area', 'lpt.area')
    prob.model.pyc_connect_des_od('duct5.Fl_O:stat:area', 'duct5.area')
    prob.model.pyc_connect_des_od('byp_bld.Fl_O:stat:area', 'byp_bld.area')
    prob.model.pyc_connect_des_od('duct17.Fl_O:stat:area', 'duct17.area')


prob.model.connect('RTO.balance.hpt_chrg_cool_frac', 'TOC.bld3.bld_exit:frac_W')
prob.model.connect('RTO.balance.hpt_nochrg_cool_frac', 'TOC.bld3.bld_inlet:frac_W')

prob.model.connect('RTO.balance.hpt_chrg_cool_frac', 'SLS.bld3.bld_exit:frac_W')
prob.model.connect('RTO.balance.hpt_nochrg_cool_frac', 'SLS.bld3.bld_inlet:frac_W')

prob.model.connect('RTO.balance.hpt_chrg_cool_frac', 'CRZ.bld3.bld_exit:frac_W')
prob.model.connect('RTO.balance.hpt_nochrg_cool_frac', 'CRZ.bld3.bld_inlet:frac_W')

########## the following section isn't in the main file from here################
bal = prob.model.add_subsystem('bal', BalanceComp())

bal.add_balance('TOC_BPR', val=23.7281, units=None, eq_units=None)
prob.model.connect('bal.TOC_BPR', 'TOC.splitter.BPR')
prob.model.connect('CRZ.ext_ratio.ER', 'bal.lhs:TOC_BPR')###############The main file doesn't include this line
prob.model.connect('CRZ:VjetRatio', 'bal.rhs:TOC_BPR')

bal.add_balance('TOC_W', val=820.95, units='lbm/s', eq_units='degR')
prob.model.connect('bal.TOC_W', 'TOC.fc.W')
prob.model.connect('RTO.burner.Fl_O:tot:T', 'bal.lhs:TOC_W')
prob.model.connect('RTO:T4max','bal.rhs:TOC_W')

bal.add_balance('CRZ_Fn_target', val=5514.4, units='lbf', eq_units='lbf', use_mult=True, mult_val=0.9, ref0=5000.0, ref=7000.0)
prob.model.connect('bal.CRZ_Fn_target', 'CRZ.balance.rhs:FAR')
prob.model.connect('TOC.perf.Fn', 'bal.lhs:CRZ_Fn_target')
prob.model.connect('CRZ.perf.Fn','bal.rhs:CRZ_Fn_target')

bal.add_balance('SLS_Fn_target', val=28620.8, units='lbf', eq_units='lbf', use_mult=True, mult_val=1.2553, ref0=28000.0, ref=30000.0)
prob.model.connect('bal.SLS_Fn_target', 'SLS.balance.rhs:FAR')
prob.model.connect('RTO.perf.Fn', 'bal.lhs:SLS_Fn_target')
prob.model.connect('SLS.perf.Fn','bal.rhs:SLS_Fn_target')
##########################to here##################################################

prob.model.add_subsystem('T4_ratio',
        ExecComp('TOC_T4 = RTO_T4*TR',
                RTO_T4={'value': 3400.0, 'units':'degR'},
                TOC_T4={'value': 3150.0, 'units':'degR'},
                TR={'value': 0.926470588, 'units': None}))
prob.model.connect('RTO:T4max','T4_ratio.RTO_T4')
prob.model.connect('T4_ratio.TOC_T4', 'TOC.balance.rhs:FAR')
prob.model.connect('TR', 'T4_ratio.TR')
prob.model.set_order(['des_vars', 'T4_ratio', 'TOC', 'RTO', 'SLS', 'CRZ', 'bal'])###############the main file doesn't include bal


newton = prob.model.nonlinear_solver = NewtonSolver()
newton.options['atol'] = 1e-6
newton.options['rtol'] = 1e-6
newton.options['iprint'] = 2
newton.options['maxiter'] = 20
newton.options['solve_subsystems'] = True
newton.options['max_sub_solves'] = 10
newton.options['err_on_non_converge'] = True
newton.options['reraise_child_analysiserror'] = False
newton.linesearch =  BoundsEnforceLS()
newton.linesearch.options['bound_enforcement'] = 'scalar'
newton.linesearch.options['iprint'] = -1

prob.model.linear_solver = DirectSolver(assemble_jac=True)

# prob.model.linear_solver = PETScKrylov()
# prob.model.linear_solver.options['iprint'] = 2
# prob.model.linear_solver.precon = DirectSolver()
# prob.model.jacobian = CSCJacobian()

# prob.model.linear_solver = PETScKrylov()
# prob.model.linear_solver.options['iprint'] = 2
# prob.model.linear_solver.options['atol'] = 1e-6
# prob.model.linear_solver.precon = LinearBlockGS()
# prob.model.linear_solver.precon.options['maxiter'] = 2
# prob.model.jacobian = CSCJacobian()

###############
#BROKEN!!!!
##############
# prob.model.linear_solver = ScipyKrylov()
# prob.model.linear_solver.options['iprint'] = 2
# prob.model.linear_solver.precon = LinearBlockGS()
# prob.model.linear_solver.precon = LinearRunOnce()
####################

# prob.model.linear_solver = LinearBlockGS()
# prob.model.linear_solver.options['maxiter'] = 10
# prob.model.linear_solver.options['iprint'] = 2


##################################################the main model sets up an optimization here###########
#it adds a driver and sets the options, adds 6 design variables, 1 objective, and 1 constraint



prob.setup(check=False)

prob['RTO.hpt_cooling.x_factor'] = 0.9

# initial guesses
prob['TOC.balance.FAR'] = 0.02650
prob['bal.TOC_W'] = 820.95############################this isn't included in the main file####################
prob['TOC.balance.lpt_PR'] = 10.937
prob['TOC.balance.hpt_PR'] = 4.185
prob['TOC.fc.balance.Pt'] = 5.272
prob['TOC.fc.balance.Tt'] = 444.41

for pt in pts:

    if pt == 'RTO':
        prob[pt+'.balance.FAR'] = 0.02832
        prob[pt+'.balance.W'] = 1916.13
        prob[pt+'.balance.BPR'] = 25.5620
        prob[pt+'.balance.fan_Nmech'] = 2132.6
        prob[pt+'.balance.lp_Nmech'] = 6611.2
        prob[pt+'.balance.hp_Nmech'] = 22288.2
        prob[pt+'.fc.balance.Pt'] = 15.349
        prob[pt+'.fc.balance.Tt'] = 552.49
        prob[pt+'.hpt.PR'] = 4.210
        prob[pt+'.lpt.PR'] = 8.161
        prob[pt+'.fan.map.RlineMap'] = 1.7500
        prob[pt+'.lpc.map.RlineMap'] = 2.0052
        prob[pt+'.hpc.map.RlineMap'] = 2.0589
        prob[pt+'.gearbox.trq_base'] = 52509.1

    if pt == 'SLS':
        prob[pt+'.balance.FAR'] = 0.02541
        prob[pt+'.balance.W'] = 1734.44#############################################
        prob[pt+'.balance.BPR'] = 27.3467
        prob[pt+'.balance.fan_Nmech'] = 1953.1
        prob[pt+'.balance.lp_Nmech'] = 6054.5
        prob[pt+'.balance.hp_Nmech'] = 21594.0
        prob[pt+'.fc.balance.Pt'] = 14.696
        prob[pt+'.fc.balance.Tt'] = 545.67
        prob[pt+'.hpt.PR'] = 4.245
        prob[pt+'.lpt.PR'] = 7.001
        prob[pt+'.fan.map.RlineMap'] = 1.7500
        prob[pt+'.lpc.map.RlineMap'] = 1.8632
        prob[pt+'.hpc.map.RlineMap'] = 2.0281
        prob[pt+'.gearbox.trq_base'] = 41779.4

    if pt == 'CRZ':
        prob[pt+'.balance.FAR'] = 0.02510
        prob[pt+'.balance.W'] = 802.79
        prob[pt+'.balance.BPR'] = 24.3233
        prob[pt+'.balance.fan_Nmech'] = 2118.7
        prob[pt+'.balance.lp_Nmech'] = 6567.9
        prob[pt+'.balance.hp_Nmech'] = 20574.1
        prob[pt+'.fc.balance.Pt'] = 5.272
        prob[pt+'.fc.balance.Tt'] = 444.41
        prob[pt+'.hpt.PR'] = 4.197
        prob[pt+'.lpt.PR'] = 10.803
        prob[pt+'.fan.map.RlineMap'] = 1.9397
        prob[pt+'.lpc.map.RlineMap'] = 2.1075
        prob[pt+'.hpc.map.RlineMap'] = 1.9746
        prob[pt+'.gearbox.trq_base'] = 22369.7



st = time.time()

prob.set_solver_print(level=-1)
prob.set_solver_print(level=2, depth=1)
prob.run_model()

###########the following lines aren't included in the main model from here##########
data = prob.compute_totals(of=['TOC.perf.Fn','RTO.perf.Fn','SLS.perf.Fn','CRZ.perf.Fn',
                                'TOC.perf.TSFC','RTO.perf.TSFC','SLS.perf.TSFC','CRZ.perf.TSFC',
                                # 'bal.TOC_BPR','TOC.hpc_CS.CS',], wrt=['OPR', 'RTO:T4max'])
                                'TOC.hpc_CS.CS',], wrt=['OPR_simple', 'RTO:T4max'])
pprint(data)

with open('derivs.pkl','wb') as f:
    pickle.dump(data, file=f)
#############to here###############

# ##################################
# # Check Totals: Hand calcualted values
# #   'TOC.perf.TSFC' wrt 'des_vars.OPR' = -1.87E-4
# #   'TOC.perf.TSFC' wrt 'des_vars.RTO:T4max' = 6.338e-6
# # prob.check_totals(step_calc='rel', step=1e-3)
# ##################################

# print('OPR', prob['fan:PRdes']*prob['lpc:PRdes']*prob['TOC.balance.hpc_PR'])
# print('OPR', prob['OPR'])
# print('T4max', prob['RTO:T4max'])
# print('T4max', prob['T4_ratio.TOC_T4'])
# print('TSFC', prob['TOC.perf.TSFC'])
# print('Fn', prob['TOC.perf.Fn'])

# prob['RTO:T4max'] *= (1.0+1e-4)
# # prob['OPR'] *= (1.0+1e-4)
# prob.run_model()

# print('OPR', prob['fan:PRdes']*prob['lpc:PRdes']*prob['TOC.balance.hpc_PR'])
# print('OPR', prob['OPR'])
# print('T4max', prob['RTO:T4max'])
# print('T4max', prob['T4_ratio.TOC_T4'])
# print('TSFC', prob['TOC.perf.TSFC'])
# print('Fn', prob['TOC.perf.Fn'])

# prob['RTO:T4max'] /= (1.0+1e-4)
# prob['OPR'] *= (1.0+1e-4)
# prob.run_model()

# print('OPR', prob['fan:PRdes']*prob['lpc:PRdes']*prob['TOC.balance.hpc_PR'])
# print('OPR', prob['OPR'])
# print('T4max', prob['RTO:T4max'])
# print('T4max', prob['T4_ratio.TOC_T4'])
# print('TSFC', prob['TOC.perf.TSFC'])
# print('Fn', prob['TOC.perf.Fn'])


# exit()

# prob.model.list_outputs(residuals=True)


# prob['OPR'] = 65.0
# prob['RTO:T4max'] = 3150.0
# prob.run_model()

# prob['OPR'] = 70.0
# prob['RTO:T4max'] = 3373.2184409
# prob.run_model()


for pt in ['TOC']+pts:
    viewer(prob, pt)


##########excluding the line that prints the time, the remainder of the lines are not included in the main model#########
print()
print('Diameter', prob['TOC.fan_dia.FanDia'][0])
print('ER', prob['CRZ.ext_ratio.ER'])
print("time", time.time() - st)

prob.model.list_outputs(explicit=True, residuals=True, residuals_tol=1e-6)

print('TOC')
print(prob['TOC.inlet.Fl_O:stat:W'] - 820.92037027)#
print(prob['TOC.inlet.Fl_O:tot:P'] - 5.26210728)#
print(prob['TOC.hpc.Fl_O:tot:P'] - 282.22391512)#
print(prob['TOC.burner.Wfuel'] - 0.74642969)#
print(prob['TOC.inlet.F_ram'] - 19866.48480269)#
print(prob['TOC.core_nozz.Fg'] - 1556.44941177)#
print(prob['TOC.byp_nozz.Fg'] - 24436.69962673)#
print(prob['TOC.perf.TSFC'] - 0.43859869)#
print(prob['TOC.perf.OPR'] - 53.63325)#
print(prob['TOC.balance.FAR'] - 0.02650755)#
print(prob['TOC.hpc.Fl_O:tot:T'] - 1530.58386828)#
print('............................')
print('RTO')
print(prob['RTO.inlet.Fl_O:stat:W'] - 1916.01614631)#
print(prob['RTO.inlet.Fl_O:tot:P'] - 15.3028198)#
print(prob['RTO.hpc.Fl_O:tot:P'] - 638.95720683)#
print(prob['RTO.burner.Wfuel'] - 1.73329552)#
print(prob['RTO.inlet.F_ram'] - 17047.53270726)#
print(prob['RTO.core_nozz.Fg'] - 2220.78852731)#
print(prob['RTO.byp_nozz.Fg'] - 37626.74417995)#
print(prob['RTO.perf.TSFC'] - 0.27367824)#
print(prob['RTO.perf.OPR'] - 41.7542136)#
print(prob['RTO.balance.FAR'] - 0.02832782)#
print(prob['RTO.balance.fan_Nmech'] - 2132.71615737)#
print(prob['RTO.balance.lp_Nmech'] - 6611.46890258)#
print(prob['RTO.balance.hp_Nmech'] - 22288.52228766)#
print(prob['RTO.hpc.Fl_O:tot:T'] - 1721.14599533)#
print('............................')
print('SLS')
print(prob['SLS.inlet.Fl_O:stat:W'] - 1735.52737576)#
print(prob['SLS.inlet.Fl_O:tot:P'] - 14.62243072)#
print(prob['SLS.hpc.Fl_O:tot:P'] - 522.99027178)#
print(prob['SLS.burner.Wfuel'] - 1.32250739)#
print(prob['SLS.inlet.F_ram'] - 61.76661874)#
print(prob['SLS.core_nozz.Fg'] - 1539.99663978)#
print(prob['SLS.byp_nozz.Fg'] - 27142.60997896)#
print(prob['SLS.perf.TSFC'] - 0.16634825)#
print(prob['SLS.perf.OPR'] - 35.76630191)#
print(prob['SLS.balance.FAR'] - 0.02544378)#
print(prob['SLS.balance.fan_Nmech'] - 1954.97855672)#
print(prob['SLS.balance.lp_Nmech'] - 6060.47827242)#
print(prob['SLS.balance.hp_Nmech'] - 21601.07508077)#
print(prob['SLS.hpc.Fl_O:tot:T'] - 1628.85845903)#
print('............................')
print('CRZ')
print(prob['CRZ.inlet.Fl_O:stat:W'] - 802.76200548)#
print(prob['CRZ.inlet.Fl_O:tot:P'] - 5.26210728)#
print(prob['CRZ.hpc.Fl_O:tot:P'] - 264.63649163)#
print(prob['CRZ.burner.Wfuel'] - 0.67514702)#
print(prob['CRZ.inlet.F_ram'] - 19427.04768877)#
print(prob['CRZ.core_nozz.Fg'] - 1383.93102366)#
print(prob['CRZ.byp_nozz.Fg'] - 23557.11447723)#
print(prob['CRZ.perf.TSFC'] - 0.44079257)#
print(prob['CRZ.perf.OPR'] - 50.29097236)#
print(prob['CRZ.balance.FAR'] - 0.02510864)#
print(prob['CRZ.balance.fan_Nmech'] - 2118.65676797)#
print(prob['CRZ.balance.lp_Nmech'] - 6567.88447364)#
print(prob['CRZ.balance.hp_Nmech'] - 20574.08438737)#
print(prob['CRZ.hpc.Fl_O:tot:T'] - 1494.29261337)#
