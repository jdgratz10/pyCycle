import openmdao.api as om

from pycycle.constants import AIR_MIX
from pycycle.cea.chem_eq import ChemEq
from pycycle.isentropic.props_calcs import PropsCalcs
from pycycle.cea.species_data import Thermo


class Properties(om.Group):

    def initialize(self):
        self.options.declare('thermo', desc='thermodynamic data set')

    def setup(self):
        thermo = self.options['thermo']

        # chem_eq calculations
        in_chemeq = ('b0', 'P', 'T')
        out_chemeq = ('n', 'n_moles')

        self.add_subsystem('chem_eq', ChemEq(thermo=thermo, mode='T'),
                           promotes_inputs=in_chemeq,
                           promotes_outputs=out_chemeq,
                           )

        num_element = thermo.num_element

        in_tp2props = ('T', 'P', 'n', 'n_moles')
        out_tp2props = ('h', 'S', 'rho', 'R')

        self.add_subsystem('tp2props', PropsCalcs(thermo=thermo),
                           promotes_inputs=in_tp2props,
                           promotes_outputs=out_tp2props
                           )

def run_calcs(thermo, T, P):

    prob = om.Problem()
    prob.model = Properties(thermo=thermo)

    prob.model.set_input_defaults('T', T, units='degK')
    prob.model.set_input_defaults('P', P, units="bar")
    prob.set_solver_print(level=-1)
    prob.setup()

    prob.run_model()

    S = prob.get_val('S', units='cal/(g*degK)')

    return(S)


if __name__ == "__main__":

    import numpy as np
    import sys
    np.set_printoptions(threshold=sys.maxsize)

    from pycycle.cea import species_data
    from pycycle import constants

    thermo_data = species_data.janaf
    init_reacts = constants.AIR_FUEL_MIX

    thermo = Thermo(thermo_data, init_reacts)

    T_range = np.linspace(50,2000,3902) #units are Kelvin
    P_range = np.linspace(.5, 21, 83) #units are bar

    S = np.empty((len(T_range),len(P_range)))

    for i, T in enumerate(T_range):
      for j, P in enumerate(P_range):
        S[i, j] = run_calcs(thermo, T, P)
        print(i, j)


    f = open( 'AIR_FUEL_MIX_entropy_full.py', 'w' )
    f.write('import numpy as np\n')
    f.write('S = ' + repr(S))
    f.close()



    