import openmdao.api as om 
import unittest
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.isentropic.explicit_isentropic import ExplicitIsentropic

class TestSetTotal(unittest.TestCase):

	def test_not_for_statics(self):

		modes = ('T', 'h', 'S')

		tol = 1e-5

		Cv_vals = [0.71428571, 0.71428571, 0.71428571]
		rho_vals = [0.23022727, 0.23022727, 0.23022727]

		for i, mode in enumerate(modes):

			prob = self._run_problem(mode, False)

			assert_near_equal(prob['Cv'], Cv_vals[i], tol)
			assert_near_equal(prob['rho'], rho_vals[i], tol)

	def test_for_statics_Ps(self):

		modes = ('T', 'h', 'S')

		tol = 1e-5

		area_vals = [0.00235978, 0.00235978, 0.00235978]
		Cv_vals = [0.71428571, 0.71428571, 0.71428571]
		V_vals = [184.06520584, 184.06520584, 184.06520584]
		Vsonic_vals = [784.85667481, 784.85667481, 784.85667481]
		MN_vals = [0.23452079, 0.23452079, 0.23452079]

		for i, mode in enumerate(modes):

			prob = self._run_problem(mode, 'Ps')

			assert_near_equal(prob['area'], area_vals[i], tol)
			assert_near_equal(prob['Cv'], Cv_vals[i], tol)
			assert_near_equal(prob['V'], V_vals[i], tol)
			assert_near_equal(prob['Vsonic'], Vsonic_vals[i], tol)
			assert_near_equal(prob['MN'], MN_vals[i], tol)

	def test_for_statics_MN(self):

		modes = ('T', 'h', 'S')

		tol = 1e-5

		area_vals = [0.00061491, 0.00061491, 0.00061491]
		Cv_vals = [0.71428571, 0.71428571, 0.71428571]
		V_vals = [706.37100733, 706.37100733, 706.37100733]
		Vsonic_vals = [784.85667481, 784.85667481, 784.85667481]

		for i, mode in enumerate(modes):

			prob = self._run_problem(mode, 'MN')

			assert_near_equal(prob['area'], area_vals[i], tol)
			assert_near_equal(prob['Cv'], Cv_vals[i], tol)
			assert_near_equal(prob['V'], V_vals[i], tol)
			assert_near_equal(prob['Vsonic'], Vsonic_vals[i], tol)

	def test_for_statics_area(self):

		modes = ('T', 'h', 'S')

		tol = 1e-5

		Cv_vals = [0.71428571, 0.71428571, 0.71428571]
		V_vals = [184.06520584, 1745.31372538, 0.36196117]
		Vsonic_vals = [784.85667481, 784.85667481, 784.85667481]
		MN_vals = [0.23452079, 2.2237356, 0.00046118]

		for i, mode in enumerate(modes):

			prob = self._run_problem(mode, 'area')

			assert_near_equal(prob['Cv'], Cv_vals[i], tol)
			assert_near_equal(prob['V'], V_vals[i], tol)
			assert_near_equal(prob['Vsonic'], Vsonic_vals[i], tol)
			assert_near_equal(prob['MN'], MN_vals[i], tol)

	def _run_problem(self, mode, for_statics):

		prob = om.Problem()
			
		prob.model = ExplicitIsentropic(mode=mode, for_statics=for_statics)

		prob.setup()
		prob.model.set_solver_print(level=0)
		prob.run_model()

		return(prob)

if __name__ == "__main__":
	unittest.main()