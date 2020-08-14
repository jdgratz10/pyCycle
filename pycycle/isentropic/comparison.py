import openmdao.api as om 
from pycycle.cea.thermo_data import janaf


def defaults(mode, for_statics, problem):

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


class setup_problem():

	def __init__(self, mode, for_statics, comp_type):
		self.mode = mode
		self.for_statics = for_statics
		self.comp_type = comp_type

		return(None)

	def run_problem(self):

		mode = self.mode
		for_statics = self.for_statics
		comp_type = self.comp_type

		self.prob = om.Problem()

		if comp_type == 'CEA':
			from pycycle.cea.set_total import SetTotal
			self.prob.model = SetTotal(mode=mode, for_statics=for_statics, thermo_data=janaf)

		elif comp_type == 'isentropic':
			from pycycle.isentropic.set_total import SetTotal
			self.prob.model = SetTotal(mode=mode, for_statics=for_statics)

		else:
			raise ValueError('You have specified an unallowable type to setup_problem')

		defaults(mode, for_statics, self.prob)

		self.prob.setup()
		self.prob.model.set_solver_print(level=0)
		# if comp_type == 'isentropic':
			# self.prob.model.set_solver_print(level=2)
		self.prob.run_model()

		prob = self.prob

		return(prob)

def run_test():

	modes = ('T', 'h', 'S')
	statics = (False, 'Ps', 'MN', 'area')

	for static in statics:
		for mode in modes:

			problem = setup_problem(mode, static, 'CEA')
			problem.run_problem()
			prob1 = problem.run_problem()

			problem = setup_problem(mode, static, 'isentropic')
			problem.run_problem()
			prob2 = problem.run_problem()

			print()
			print('for_statics', static)
			print('mode', mode)
			print('deviations (actual, isentropic)')

			if not static:
				print('T', prob1.get_val('flow:T', units='degK'), prob2.get_val('flow:T', units='degK'))
				print('P', prob1.get_val('flow:P', units='bar'), prob2.get_val('flow:P', units='bar'))
				print('h', prob1.get_val('flow:h', units='cal/g'), prob2.get_val('flow:h', units='cal/g'))
				print('S', prob1.get_val('flow:S', units='cal/(g*degK)'), prob2.get_val('flow:S', units='cal/(g*degK)'))
				print('gamma', prob1.get_val('flow:gamma', units=None), prob2.get_val('flow:gamma', units=None))
				print('Cp', prob1.get_val('flow:Cp', units='cal/(g*degK)'), prob2.get_val('flow:Cp', units='cal/(g*degK)'))
				print('Cv', prob1.get_val('flow:Cv', units='cal/(g*degK)'), prob2.get_val('flow:Cv', units='cal/(g*degK)'))
				print('rho', prob1.get_val('flow:rho', units='kg/m**3'), prob2.get_val('flow:rho', units='kg/m**3'))
				print('R', prob1.get_val('flow:R', units='cal/(g*degK)'), prob2.get_val('flow:R', units='cal/(g*degK)'))
				print()

			elif static == 'Ps':
				print('S', prob1.get_val('S', units='cal/(g*degK)'), prob2.get_val('S', units='cal/(g*degK)'))
				print('area', prob1.get_val('area', units='m**2'), prob2.get_val('area', units='m**2'))
				print('Cp', prob1.get_val('Cp', units='cal/(g*degK)'), prob2.get_val('Cp', units='cal/(g*degK)'))
				print('Cv', prob1.get_val('Cv', units='cal/(g*degK)'), prob2.get_val('Cv', units='cal/(g*degK)'))
				print('rho', prob1.get_val('rho', units='kg/m**3'), prob2.get_val('rho', units='kg/m**3'))
				print('R', prob1.get_val('R', units='cal/(g*degK)'), prob2.get_val('R', units='cal/(g*degK)'))
				print('V', prob1.get_val('V', units='m/s'), prob2.get_val('V', units='m/s'))
				print('Vsonic', prob1.get_val('Vsonic', units='m/s'), prob2.get_val('Vsonic', units='m/s'))
				print('MN', prob1.get_val('MN', units=None), prob2.get_val('MN', units=None))
				print('gamma', prob1.get_val('gamma', units=None), prob2.get_val('gamma', units=None))
				print()
					

			elif static == 'MN':
				print('WARNING: in modes T and h the original data was unconverged')
				print('S', prob1.get_val('S', units='cal/(g*degK)'), prob2.get_val('S', units='cal/(g*degK)'))
				print('area', prob1.get_val('area', units='m**2'), prob2.get_val('area', units='m**2'))
				print('Cp', prob1.get_val('Cp', units='cal/(g*degK)'), prob2.get_val('Cp', units='cal/(g*degK)'))
				print('Cv', prob1.get_val('Cv', units='cal/(g*degK)'), prob2.get_val('Cv', units='cal/(g*degK)'))
				print('rho', prob1.get_val('rho', units='kg/m**3'), prob2.get_val('rho', units='kg/m**3'))
				print('R', prob1.get_val('R', units='cal/(g*degK)'), prob2.get_val('R', units='cal/(g*degK)'))
				print('V', prob1.get_val('V', units='m/s'), prob2.get_val('V', units='m/s'))
				print('Vsonic', prob1.get_val('Vsonic', units='m/s'), prob2.get_val('Vsonic', units='m/s'))
				print('gamma', prob1.get_val('gamma', units=None), prob2.get_val('gamma', units=None))
				print()
					

			else:
				print('S', prob1.get_val('S', units='cal/(g*degK)'), prob2.get_val('S', units='cal/(g*degK)'))
				print('Cp', prob1.get_val('Cp', units='cal/(g*degK)'), prob2.get_val('Cp', units='cal/(g*degK)'))
				print('Cv', prob1.get_val('Cv', units='cal/(g*degK)'), prob2.get_val('Cv', units='cal/(g*degK)'))
				print('rho', prob1.get_val('rho', units='kg/m**3'), prob2.get_val('rho', units='kg/m**3'))
				print('R', prob1.get_val('R', units='cal/(g*degK)'), prob2.get_val('R', units='cal/(g*degK)'))
				print('V', prob1.get_val('V', units='m/s'), prob2.get_val('V', units='m/s'))
				print('Vsonic', prob1.get_val('Vsonic', units='m/s'), prob2.get_val('Vsonic', units='m/s'))
				print('MN', prob1.get_val('MN', units=None), prob2.get_val('MN', units=None))
				print('gamma', prob1.get_val('gamma', units=None), prob2.get_val('gamma', units=None))
				print()

if __name__ == "__main__":
	run_test()