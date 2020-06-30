import numpy as np
import time
import pickle
from pprint import pprint

import openmdao.api as om
import pycycle.api as pyc

from N3ref import N3, viewer

prob = om.Problem()
prob.model = pyc.MPCycle()

des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])

des_vars.add_output('inlet:ram_recovery', 0.9980),
des_vars.add_output('fan:PRdes', 1.300),
des_vars.add_output('fan:effDes', 0.96888),
des_vars.add_output('fan:effPoly', 0.97),
des_vars.add_output('splitter:BPR', 23.7281), #23.9878####################
des_vars.add_output('duct2:dPqP', 0.0100),
des_vars.add_output('lpc:PRdes', 3.000),
des_vars.add_output('lpc:effDes', 0.889513),
des_vars.add_output('lpc:effPoly', 0.905),
des_vars.add_output('duct25:dPqP', 0.0150),
des_vars.add_output('hpc:PRdes', 14.103),
des_vars.add_output('OPR', 53.6332) #53.635)
des_vars.add_output('OPR_simple', 55.0)######the main file doesn't have this line
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

des_vars.add_output('bld3:bld_inlet:frac_W', 0.063660111), #different than NPSS due to Wref
des_vars.add_output('bld3:bld_exit:frac_W', 0.07037185), #different than NPSS due to Wref

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
#####main file includes TOC:W here

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
des_vars.add_output('SLS:MN', 0.001),
des_vars.add_output('SLS:alt', 0.0, units='ft'),
des_vars.add_output('SLS:Fn_target', 28620.9, units='lbf'), #8950.0#########################
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
des_vars.add_output('CRZ:Fn_target', 5466.5, units='lbf'), #8950.0##############################
des_vars.add_output('CRZ:dTs', 0.0, units='degR')
des_vars.add_output('CRZ:Ath', 4747.1, units='inch**2')
des_vars.add_output('CRZ:RlineMap', 1.9397)
des_vars.add_output('CRZ:ram_recovery', 0.9980),
des_vars.add_output('CRZ:duct2:dPqP', 0.0092)
des_vars.add_output('CRZ:duct25:dPqP', 0.0138)
des_vars.add_output('CRZ:duct45:dPqP', 0.0050)
des_vars.add_output('CRZ:duct5:dPqP', 0.0097)
des_vars.add_output('CRZ:duct17:dPqP', 0.0148)
des_vars.add_output('CRZ:VjetRatio', 1.40) #1.41038)#################################


# TOC POINT (DESIGN)
prob.model.pyc_add_pnt('TOC', N3())

prob.model.connect('TOC:alt', 'TOC.fc.alt')
prob.model.connect('TOC:MN', 'TOC.fc.MN')

prob.model.connect('TOC:ram_recovery', 'TOC.inlet.ram_recovery')
prob.model.connect('fan:PRdes', ['TOC.fan.PR', 'TOC.opr_calc.FPR'])###main file doesn't include opr_calc.FPR here
prob.model.connect('fan:effPoly', 'TOC.balance.rhs:fan_eff')
prob.model.connect('duct2:dPqP', 'TOC.duct2.dPqP')
prob.model.connect('lpc:PRdes', ['TOC.lpc.PR', 'TOC.opr_calc.LPCPR'])###main file doesn't include opr_calc.LPCPR here
prob.model.connect('lpc:effPoly', 'TOC.balance.rhs:lpc_eff')
prob.model.connect('duct25:dPqP', 'TOC.duct25.dPqP')
prob.model.connect('OPR_simple', 'TOC.balance.rhs:hpc_PR')###this line doesn't exist in main file
prob.model.connect('hpt:effPoly', 'TOC.balance.rhs:hpt_eff')
prob.model.connect('duct45:dPqP', 'TOC.duct45.dPqP')
prob.model.connect('lpt:effPoly', 'TOC.balance.rhs:lpt_eff')
prob.model.connect('duct5:dPqP', 'TOC.duct5.dPqP')
###the main file doesn't include ext_ratio.core_Cv
prob.model.connect('duct17:dPqP', 'TOC.duct17.dPqP')
###main file doesn't include ext_ratio.byp_Cv
prob.model.connect('fan_shaft:Nmech', 'TOC.Fan_Nmech')
prob.model.connect('lp_shaft:Nmech', 'TOC.LP_Nmech')
prob.model.connect('hp_shaft:Nmech', 'TOC.HP_Nmech')

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


prob.model.connect('RTO:Fn_target', 'RTO.balance.rhs:FAR')

prob.model.pyc_add_pnt('RTO', N3(design=False, cooling=True))
prob.model.pyc_add_pnt('SLS', N3(design=False))
prob.model.pyc_add_pnt('CRZ', N3(design=False))


for pt in pts:

    prob.model.connect(pt+':alt', pt+'.fc.alt')
    prob.model.connect(pt+':MN', pt+'.fc.MN')
    prob.model.connect(pt+':dTs', pt+'.fc.dTs')
    prob.model.connect(pt+':RlineMap',pt+'.balance.rhs:BPR')
    prob.model.connect(pt+':ram_recovery', pt+'.inlet.ram_recovery')

prob.model.pyc_add_cycle_param('burner.dPqP', 0.0400),
prob.model.pyc_add_cycle_param('core_nozz.Cv', 0.9999),
prob.model.pyc_add_cycle_param('ext_ratio.core_Cv', 0.9999)
prob.model.pyc_add_cycle_param('byp_nozz.Cv', 0.9975),
prob.model.pyc_add_cycle_param('ext_ratio.byp_Cv', 0.9975),
prob.model.pyc_add_cycle_param('lp_shaft.fracLoss', 0.01)
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

prob.model.pyc_connect_des_od('duct2.s_dPqP', 'duct2.s_dPqP')
prob.model.pyc_connect_des_od('duct25.s_dPqP', 'duct25.s_dPqP')
prob.model.pyc_connect_des_od('duct45.s_dPqP', 'duct45.s_dPqP')
prob.model.pyc_connect_des_od('duct5.s_dPqP', 'duct5.s_dPqP')
prob.model.pyc_connect_des_od('duct17.s_dPqP', 'duct17.s_dPqP')


prob.model.connect('RTO.balance.hpt_chrg_cool_frac', 'TOC.bld3.bld_exit:frac_W')
prob.model.connect('RTO.balance.hpt_nochrg_cool_frac', 'TOC.bld3.bld_inlet:frac_W')

prob.model.connect('RTO.balance.hpt_chrg_cool_frac', 'SLS.bld3.bld_exit:frac_W')
prob.model.connect('RTO.balance.hpt_nochrg_cool_frac', 'SLS.bld3.bld_inlet:frac_W')

prob.model.connect('RTO.balance.hpt_chrg_cool_frac', 'CRZ.bld3.bld_exit:frac_W')
prob.model.connect('RTO.balance.hpt_nochrg_cool_frac', 'CRZ.bld3.bld_inlet:frac_W')

###the main model includes 4 connect statements here:
    #splitter:BPR to TOC.splitter.BPR
    #TOC:W to TOC.fc.W
    #CRZ:Fn_target to CRZ.balance.rhs:FAR
    #SLS:Fn_target to SLS.balance.rhs:FAR

#the main code doesn't include from here#############
bal = prob.model.add_subsystem('bal', om.BalanceComp())

bal.add_balance('TOC_BPR', val=23.7281, units=None, eq_units=None)
prob.model.connect('bal.TOC_BPR', 'TOC.splitter.BPR')
prob.model.connect('CRZ.ext_ratio.ER', 'bal.lhs:TOC_BPR')
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
#to here######################################

prob.model.add_subsystem('T4_ratio',
                         om.ExecComp('TOC_T4 = RTO_T4*TR',
                                     RTO_T4={'value': 3400.0, 'units':'degR'},
                                     TOC_T4={'value': 3150.0, 'units':'degR'},
                                     TR={'value': 0.926470588, 'units': None}))
prob.model.connect('RTO:T4max','T4_ratio.RTO_T4')
prob.model.connect('T4_ratio.TOC_T4', 'TOC.balance.rhs:FAR')
prob.model.connect('TR', 'T4_ratio.TR')
prob.model.set_order(['des_vars', 'T4_ratio', 'TOC', 'RTO', 'SLS', 'CRZ', 'bal'])


newton = prob.model.nonlinear_solver = om.NewtonSolver()
newton.options['atol'] = 1e-6
newton.options['rtol'] = 1e-6
newton.options['iprint'] = 2
newton.options['maxiter'] = 20
newton.options['solve_subsystems'] = True
newton.options['max_sub_solves'] = 10
newton.options['err_on_non_converge'] = True
newton.options['reraise_child_analysiserror'] = False
newton.linesearch =  om.BoundsEnforceLS()
# newton.linesearch.options['maxiter'] = 2
newton.linesearch.options['bound_enforcement'] = 'scalar'
newton.linesearch.options['iprint'] = -1

prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs']
prob.driver.opt_settings={'Major step limit': 0.05}

prob.model.add_design_var('fan:PRdes', lower=1.20, upper=1.4)
prob.model.add_design_var('lpc:PRdes', lower=2.5, upper=4.0)###################
prob.model.add_design_var('OPR_simple', lower=40.0, upper=70.0, ref0=40.0, ref=70.0)
prob.model.add_design_var('RTO:T4max', lower=3000.0, upper=3600.0, ref0=3000.0, ref=3600.0)
prob.model.add_design_var('CRZ:VjetRatio', lower=1.35, upper=1.45, ref0=1.35, ref=1.45)
prob.model.add_design_var('TR', lower=0.5, upper=0.95, ref0=0.5, ref=0.95)

prob.model.add_objective('CRZ.perf.TSFC', ref0=0.4, ref=0.5)###############################

# to add the constraint to the model
prob.model.add_constraint('TOC.fan_dia.FanDia', upper=100.0, ref=100.0)
prob.model.add_constraint('TOC.perf.Fn', lower=5800.0, ref=6000.0)########This isn't included in the main model

prob.setup(check=False)

prob['RTO.hpt_cooling.x_factor'] = 0.9

# initial guesses
prob['TOC.balance.FAR'] = 0.02650
prob['bal.TOC_W'] = 820.95
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
        prob[pt+'.balance.W'] = 2000 # 1734.44
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
# prob.run_model()##############################this is uncommented in main model
prob.run_driver()########################this isn't in main model
# exit()


for pt in ['TOC']+pts:
    viewer(prob, pt)

#######lines from here##############
print()
print('Diameter', prob['TOC.fan_dia.FanDia'][0])
print('ER', prob['CRZ.ext_ratio.ER'])
#####to here aren't in main model#####################

print("time", time.time() - st)

print('TOC')
print(prob['TOC.inlet.Fl_O:stat:W'] - 810.91780825)#
print(prob['TOC.inlet.Fl_O:tot:P'] - 5.26210728)#
print(prob['TOC.hpc.Fl_O:tot:P'] - 359.19407379)#
print(prob['TOC.burner.Wfuel'] - 0.84696398)#
print(prob['TOC.inlet.F_ram'] - 19624.42022041)#
print(prob['TOC.core_nozz.Fg'] - 2062.17720394)#
print(prob['TOC.byp_nozz.Fg'] - 24607.96791651)#
print(prob['TOC.perf.TSFC'] - 0.43275466)#
print(prob['TOC.perf.OPR'] - 68.2605)#
print(prob['TOC.balance.FAR'] - 0.02157518)#
print(prob['TOC.hpc.Fl_O:tot:T'] - 1629.57914093)#
print('............................')
print('RTO')
print(prob['RTO.inlet.Fl_O:stat:W'] - 1836.24241256)#
print(prob['RTO.inlet.Fl_O:tot:P'] - 15.3028198)#
print(prob['RTO.hpc.Fl_O:tot:P'] - 746.29459765)#
print(prob['RTO.burner.Wfuel'] - 1.74761057)#
print(prob['RTO.inlet.F_ram'] - 16337.75511063)#
print(prob['RTO.core_nozz.Fg'] - 2531.69432791)#
print(prob['RTO.byp_nozz.Fg'] - 36606.06077695)#
print(prob['RTO.perf.TSFC'] - 0.27593851)#
print(prob['RTO.perf.OPR'] - 48.76843661)#
print(prob['RTO.balance.FAR'] - 0.02201472)#
print(prob['RTO.balance.fan_Nmech'] - 2047.49245139)#
print(prob['RTO.balance.lp_Nmech'] - 6347.27346339)#
print(prob['RTO.balance.hp_Nmech'] - 21956.80317383)#
print(prob['RTO.hpc.Fl_O:tot:T'] - 1785.68517839)#
print('............................')
print('SLS')
print(prob['SLS.inlet.Fl_O:stat:W'] - 1665.15413784)#
print(prob['SLS.inlet.Fl_O:tot:P'] - 14.62243072)#
print(prob['SLS.hpc.Fl_O:tot:P'] - 618.67644706)#
print(prob['SLS.burner.Wfuel'] - 1.3646179)#
print(prob['SLS.inlet.F_ram'] - 59.26206766)#
print(prob['SLS.core_nozz.Fg'] - 1803.28636041)#
print(prob['SLS.byp_nozz.Fg'] - 26876.81570145)#
print(prob['SLS.perf.TSFC'] - 0.17164501)#
print(prob['SLS.perf.OPR'] - 42.31009597)#
print(prob['SLS.balance.FAR'] - 0.02009085)#
print(prob['SLS.balance.fan_Nmech'] - 1885.03384389)#
print(prob['SLS.balance.lp_Nmech'] - 5843.64806172)#
print(prob['SLS.balance.hp_Nmech'] - 21335.89542087)#
print(prob['SLS.hpc.Fl_O:tot:T'] - 1698.10451317)#
print('............................')
print('CRZ')
print(prob['CRZ.inlet.Fl_O:stat:W'] - 792.88599794)#
print(prob['CRZ.inlet.Fl_O:tot:P'] - 5.26210728)#
print(prob['CRZ.hpc.Fl_O:tot:P'] - 336.48275934)#
print(prob['CRZ.burner.Wfuel'] - 0.76513434)#
print(prob['CRZ.inlet.F_ram'] - 19188.04575789)#
print(prob['CRZ.core_nozz.Fg'] - 1832.22026473)#
print(prob['CRZ.byp_nozz.Fg'] - 23696.97790339)#
print(prob['CRZ.perf.TSFC'] - 0.43438218)#
print(prob['CRZ.perf.OPR'] - 63.9444887)#
print(prob['CRZ.balance.FAR'] - 0.02044895)#
print(prob['CRZ.balance.fan_Nmech'] - 2118.19810925)#
print(prob['CRZ.balance.lp_Nmech'] - 6566.46262111)#
print(prob['CRZ.balance.hp_Nmech'] - 20575.59530495)#
print(prob['CRZ.hpc.Fl_O:tot:T'] - 1591.47697124)#
