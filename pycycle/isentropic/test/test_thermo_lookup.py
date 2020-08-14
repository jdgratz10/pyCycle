import openmdao.api as om 
import unittest
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.isentropic.thermo_lookup import EnthalpyFromTemp, TempFromSP, TempFromGamma, TempFromEnthalpy
from pycycle.isentropic import properties

class TestSetTotal(unittest.TestCase):

	def test_enthalpy_from_temp(self):

		prob = om.Problem()
			
		prob.model = om.Group()
		prob.model.add_subsystem('enthalpy', EnthalpyFromTemp(), promotes=['*'])

		prob.model.set_input_defaults('Cp', 1.65, units='cal/(g*degK)')
		prob.model.set_input_defaults('T', 330, units='degK')

		prob.setup()
		prob.model.set_solver_print(level=0)
		prob.run_model()

		h_val = 45.43607987

		tol = 1e-5

		assert_near_equal(prob['h'], h_val, tol)

	def test_temp_from_SP(self):

		prob = om.Problem()
			
		prob.model = TempFromSP(S_data=properties.AIR_MIX_entropy)

		prob.model.set_input_defaults('S', 1.65, units='cal/(g*degK)')
		prob.model.set_input_defaults('P', 1.013, units='bar')

		prob.setup()
		prob.model.set_solver_print(level=0)
		prob.run_model()

		T_val = 311.11617479

		tol = 1e-5

		assert_near_equal(prob['T'], T_val, tol)

	def test_temp_from_gamma(self):

		prob = om.Problem()
			
		prob.model = om.Group()
		prob.model.add_subsystem('temp', TempFromGamma(), promotes=['*'])

		prob.model.set_input_defaults('gamma', 1.4, units=None)
		prob.model.set_input_defaults('MN', 0.6, units=None)
		prob.model.set_input_defaults('Tt', 330, units='degK')

		prob.setup()
		prob.model.set_solver_print(level=0)
		prob.run_model()

		T_val = 307.8358209

		tol = 1e-5

		assert_near_equal(prob['T'], T_val, tol)

	def test_temp_from_enthalpy(self):

		prob = om.Problem()
			
		prob.model = om.Group()
		prob.model.add_subsystem('temp', TempFromEnthalpy(), promotes=['*'])

		prob.model.set_input_defaults('Cp', 1.65, units='cal/(g*degK)')
		prob.model.set_input_defaults('h', 10, units='cal/g')

		prob.setup()
		prob.model.set_solver_print(level=0)
		prob.run_model()

		T_val = 308.52358796

		tol = 1e-5

		assert_near_equal(prob['T'], T_val, tol)


if __name__ == "__main__":
	unittest.main()