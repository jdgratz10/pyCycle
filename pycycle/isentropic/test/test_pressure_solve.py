import openmdao.api as om 
import unittest
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.isentropic.pressure_solve import PressureSolve
from pycycle.isentropic import properties

class TestSetTotal(unittest.TestCase):

	def test_mode_T_h(self):

		### Setup problem in T mode ###

		prob_T = om.Problem()
			
		prob_T.model = PressureSolve(mode='T', S_data=properties.AIR_MIX_entropy)

		prob_T.model.set_input_defaults('T', 330, units='degK')
		prob_T.model.set_input_defaults('Tt', 344, units='degK')
		prob_T.model.set_input_defaults('gamma', 1.4, units=None)
		prob_T.model.set_input_defaults('MN', 0.6, units=None)

		prob_T.setup()
		prob_T.model.set_solver_print(level=0)
		prob_T.run_model()

		### Setup problem in h mode ###

		prob_h = om.Problem()
			
		prob_h.model = PressureSolve(mode='h', S_data=properties.AIR_MIX_entropy)

		prob_h.model.set_input_defaults('T', 330, units='degK')
		prob_h.model.set_input_defaults('Tt', 344, units='degK')
		prob_h.model.set_input_defaults('gamma', 1.4, units=None)
		prob_h.model.set_input_defaults('MN', 0.6, units=None)

		prob_h.setup()
		prob_h.model.set_solver_print(level=0)
		prob_h.run_model()

		P_val = 0.32624162
		Pt_val = 0.41612242
		S_val = 1.73185254

		tol = 1e-5

		assert_near_equal(prob_T['P'], P_val, tol)
		assert_near_equal(prob_T['Pt'], Pt_val, tol)
		assert_near_equal(prob_T['S'], S_val, tol)

		assert_near_equal(prob_h['P'], P_val, tol)
		assert_near_equal(prob_h['Pt'], Pt_val, tol)
		assert_near_equal(prob_h['S'], S_val, tol)

	def test_mode_S(self):

		prob = om.Problem()
			
		prob.model = PressureSolve(mode='S', S_data=properties.AIR_MIX_entropy)

		prob.model.set_input_defaults('S_desired', 1.65, units='cal/(g*degK)')
		prob.model.set_input_defaults('T', 330, units='degK')

		prob.setup()
		prob.model.set_solver_print(level=0)
		prob.run_model()

		P_val = 1.24436218

		tol = 1e-5

		assert_near_equal(prob['P'], P_val, tol)


if __name__ == "__main__":
	unittest.main()