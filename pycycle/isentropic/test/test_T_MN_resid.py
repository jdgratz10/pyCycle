import openmdao.api as om 
import unittest
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.isentropic.T_MN_resid import TmnResid

class TestSetTotal(unittest.TestCase):

	def test_mode_S(self):

		prob = om.Problem()
			
		prob.model = TmnResid()

		prob.model.set_input_defaults('ht', 10, units='cal/g')
		prob.model.set_input_defaults('gamma', 1.4, units=None)
		prob.model.set_input_defaults('R', 0.06860664330, units='cal/(g*degK)')
		prob.model.set_input_defaults('Tt', 344.10276662, units='degK')
		prob.model.set_input_defaults('Cp', 0.24015494, units='cal/(g*degK)')

		prob.setup()
		prob.model.set_solver_print(level=0)
		prob.run_model()

		T_val = 344.09677278

		tol = 1e-5

		assert_near_equal(prob['T'], T_val, tol)


if __name__ == "__main__":
	unittest.main()