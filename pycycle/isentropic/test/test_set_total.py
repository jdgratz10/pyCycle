import openmdao.api as om 
import unittest
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.isentropic.set_total import SetTotal

class TestSetTotal(unittest.TestCase):

	def test_not_for_statics(self):

		modes = ('T', 'h', 'S')

		tol = 1e-5

		T_vals = [330, 331.6108312, 311.11617479]
		P_vals = [1.013, 1.013, 1.013]
		h_vals = [6.61315093, 7., 2.07810702]
		S_vals = [1.66416781, 1.66533935, 1.65]
		Cp_vals = [0.24015494, 0.24015494, 0.24015494]
		Cv_vals = [0.17165404, 0.17165404, 0.17165404]
		rho_vals = [1.04114346,1.03608601, 1.10433777]
		R_vals = [0.07046821, 0.07046821, 0.07046821]
		gamma_vals = [1.4, 1.4, 1.4]

		for i, mode in enumerate(modes):

			prob = self._run_problem(mode, False)

			assert_near_equal(prob.get_val('flow:T', units='degK'), T_vals[i], tol)
			assert_near_equal(prob.get_val('flow:P', units='bar'), P_vals[i], tol)
			assert_near_equal(prob.get_val('flow:h', units='cal/g'), h_vals[i], tol)
			assert_near_equal(prob.get_val('flow:S', units='cal/(g*degK)'), S_vals[i], tol)
			assert_near_equal(prob.get_val('flow:Cp', units='cal/(g*degK)'), Cp_vals[i], tol)
			assert_near_equal(prob.get_val('flow:Cv', units='cal/(g*degK)'), Cv_vals[i], tol)
			assert_near_equal(prob.get_val('flow:rho', units='kg/m**3'), rho_vals[i], tol)
			assert_near_equal(prob.get_val('flow:R', units='cal/(g*degK)'), R_vals[i], tol)
			assert_near_equal(prob.get_val('flow:gamma', units=None), gamma_vals[i], tol)

	def test_for_statics_Ps(self):

		modes = ('T', 'h', 'S')

		tol = 1e-5

		S_vals = [1.66416781, 1.66533935, 1.65]
		area_vals = [0.03830476, 0.04089826, 0.02361266]
		Cp_vals = [0.24015494, 0.24015494, 0.24015494]
		Cv_vals = [0.17153924,0.17153924, 0.17153924]
		rho_vals = [1.04114346, 1.03608601, 1.10433777]
		R_vals = [0.07046821, 0.07046821, 0.07046821]
		V_vals = [170.60574212, 160.56702096, 260.92170516]
		Vsonic_vals = [369.07400071, 369.97368583, 358.35855592]
		MN_vals = [0.46225348, 0.43399579, 0.72810235]
		gamma_vals = [1.4, 1.4, 1.4]

		for i, mode in enumerate(modes):

			prob = self._run_problem(mode, 'Ps')

			assert_near_equal(prob.get_val('S', units='cal/(g*degK)'), S_vals[i], tol)
			assert_near_equal(prob.get_val('area', units='m**2'), area_vals[i], tol)
			assert_near_equal(prob.get_val('Cp', units='cal/(g*degK)'), Cp_vals[i], tol)
			assert_near_equal(prob.get_val('Cv', units='cal/(g*degK)'), Cv_vals[i], tol)
			assert_near_equal(prob.get_val('rho', units='kg/m**3'), rho_vals[i], tol)
			assert_near_equal(prob.get_val('R', units='cal/(g*degK)'), R_vals[i], tol)
			assert_near_equal(prob.get_val('V', units='m/s'), V_vals[i], tol)
			assert_near_equal(prob.get_val('Vsonic', units='m/s'), Vsonic_vals[i], tol)
			assert_near_equal(prob.get_val('MN', units=None), MN_vals[i], tol)
			assert_near_equal(prob.get_val('gamma', units=None), gamma_vals[i], tol)

	def test_for_statics_MN(self):

		modes = ('T', 'h', 'S')

		tol = 1e-5

		S_vals = [1.73159152, 1.73701541, 1.65]
		area_vals = [0.0909788, 0.10320369, 0.02596179]
		Cp_vals = [0.24015494, 0.24015494, 0.24015494]
		Cv_vals = [0.17153924,0.17153924, 0.17153924]
		rho_vals = [0.33771634, 0.29698853, 1.19996315]
		R_vals = [0.07046821, 0.07046821, 0.07046821]
		V_vals = [221.44440043, 221.9842115, 218.4008994]
		Vsonic_vals = [369.07400071, 369.97368583, 364.001499]
		gamma_vals = [1.4, 1.4, 1.4]

		for i, mode in enumerate(modes):

			prob = self._run_problem(mode, 'MN')

			assert_near_equal(prob.get_val('S', units='cal/(g*degK)'), S_vals[i], tol)
			assert_near_equal(prob.get_val('area', units='m**2'), area_vals[i], tol)
			assert_near_equal(prob.get_val('Cp', units='cal/(g*degK)'), Cp_vals[i], tol)
			assert_near_equal(prob.get_val('Cv', units='cal/(g*degK)'), Cv_vals[i], tol)
			assert_near_equal(prob.get_val('rho', units='kg/m**3'), rho_vals[i], tol)
			assert_near_equal(prob.get_val('R', units='cal/(g*degK)'), R_vals[i], tol)
			assert_near_equal(prob.get_val('V', units='m/s'), V_vals[i], tol)
			assert_near_equal(prob.get_val('Vsonic', units='m/s'), Vsonic_vals[i], tol)
			assert_near_equal(prob.get_val('gamma', units=None), gamma_vals[i], tol)

	def test_for_statics_area(self):

		modes = ('T', 'h', 'S')

		tol = 1e-5

		S_vals = [1.75951838, 1.76010526, 1.65]
		Cp_vals = [0.24015494, 0.24015494, 0.24015494]
		Cv_vals = [0.17153924,0.17153924, 0.17153924]
		rho_vals = [0.07976151, 0.08474823, 1.42370908]
		R_vals = [0.07046821, 0.07046821, 0.07046821]
		V_vals = [170.60574212, 160.56702096, 9.55797168]
		Vsonic_vals = [369.07400071, 369.97368583, 376.87780758]
		MN_vals = [0.46225348, 0.43399579, 0.02536093]
		gamma_vals = [1.4, 1.4, 1.4]

		for i, mode in enumerate(modes):

			prob = self._run_problem(mode, 'area')

			assert_near_equal(prob.get_val('S', units='cal/(g*degK)'), S_vals[i], tol)
			assert_near_equal(prob.get_val('Cp', units='cal/(g*degK)'), Cp_vals[i], tol)
			assert_near_equal(prob.get_val('Cv', units='cal/(g*degK)'), Cv_vals[i], tol)
			assert_near_equal(prob.get_val('rho', units='kg/m**3'), rho_vals[i], tol)
			assert_near_equal(prob.get_val('R', units='cal/(g*degK)'), R_vals[i], tol)
			assert_near_equal(prob.get_val('V', units='m/s'), V_vals[i], tol)
			assert_near_equal(prob.get_val('Vsonic', units='m/s'), Vsonic_vals[i], tol)
			assert_near_equal(prob.get_val('MN', units=None), MN_vals[i], tol)
			assert_near_equal(prob.get_val('gamma', units=None), gamma_vals[i], tol)

	def _run_problem(self, mode, for_statics):

		prob = om.Problem()
			
		prob.model = SetTotal(mode=mode, for_statics=for_statics)

		self._defaults(mode, for_statics, prob)

		prob.setup()
		prob.model.set_solver_print(level=0)
		prob.run_model()

		return(prob)

	def _defaults(self, mode, for_statics, problem):

		prob = problem 

		if not for_statics:
			if mode == 'T':
				prob.model.set_input_defaults('T', 330, units='degK')
				prob.model.set_input_defaults('P', 1.013, units='bar')

			elif mode == 'h':
				prob.model.set_input_defaults('P', 1.013, units='bar')
				prob.model.set_input_defaults('h', 7, units='cal/g')

			else:
				prob.model.set_input_defaults('P', 1.013, units='bar')
				prob.model.set_input_defaults('S', 1.65, units='cal/(g*degK)')

		elif for_statics == 'Ps':
			if mode == 'T':
				prob.model.set_input_defaults('T', 330, units='degK')
				prob.model.set_input_defaults('P', 1.013, units='bar')
				prob.model.set_input_defaults('ht', 10, units='cal/g')
				prob.model.set_input_defaults('W', 15, units='lbm/s')

			elif mode == 'h':
				prob.model.set_input_defaults('P', 1.013, units='bar')
				prob.model.set_input_defaults('ht', 10, units='cal/g')
				prob.model.set_input_defaults('W', 15, units='lbm/s')
				prob.model.set_input_defaults('h', 7, units='cal/g')

			else:
				prob.model.set_input_defaults('P', 1.013, units='bar')
				prob.model.set_input_defaults('ht', 10, units='cal/g')
				prob.model.set_input_defaults('W', 15, units='lbm/s')
				prob.model.set_input_defaults('S', 1.65, units='cal/(g*degK)')

		elif for_statics == 'MN':
			if mode == 'T':
				prob.model.set_input_defaults('T', 330, units='degK')
				prob.model.set_input_defaults('MN', 0.6, units=None)
				prob.model.set_input_defaults('ht', 10, units='cal/g')
				prob.model.set_input_defaults('W', 15, units='lbm/s')

			elif mode == 'h':
				prob.model.set_input_defaults('h', 7, units='cal/g')
				prob.model.set_input_defaults('MN', 0.6, units=None)
				prob.model.set_input_defaults('ht', 10, units='cal/g')
				prob.model.set_input_defaults('W', 15, units='lbm/s')

			else:
				prob.model.set_input_defaults('S', 1.65, units='cal/(g*degK)')
				prob.model.set_input_defaults('MN', 0.6, units=None)
				prob.model.set_input_defaults('ht', 10, units='cal/g')
				prob.model.set_input_defaults('W', 15, units='lbm/s')
				
		else:
			if mode == 'T':
				prob.model.set_input_defaults('T', 330, units='degK')
				prob.model.set_input_defaults('area', .5, units='m**2')
				prob.model.set_input_defaults('ht', 10, units='cal/g')
				prob.model.set_input_defaults('W', 15, units='lbm/s')

			elif mode == 'h':
				prob.model.set_input_defaults('h', 7, units='cal/g')
				prob.model.set_input_defaults('area', .5, units='m**2')
				prob.model.set_input_defaults('ht', 10, units='cal/g')
				prob.model.set_input_defaults('W', 15, units='lbm/s')

			else:
				prob.model.set_input_defaults('S', 1.65, units='cal/(g*degK)')
				prob.model.set_input_defaults('area', .5, units='m**2')
				prob.model.set_input_defaults('ht', 10, units='cal/g')
				prob.model.set_input_defaults('W', 15, units='lbm/s')

		return()

if __name__ == "__main__":
	unittest.main()