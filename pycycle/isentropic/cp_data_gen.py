import numpy as np
import sys
from pycycle.constants import R_UNIVERSAL_SI, AIR_MIX, R_UNIVERSAL_ENG, AIR_FUEL_MIX
from pycycle.cea.species_data import Thermo
from pycycle.cea.thermo_data import janaf
from pycycle.cea.set_total import Properties
import openmdao.api as om

np.set_printoptions(threshold=sys.maxsize)

thermo_data = janaf
init_reacts = AIR_FUEL_MIX









thermo = Thermo(thermo_data, init_reacts=init_reacts)

compounds = init_reacts.keys()
compound_wts_full = thermo.wt_mole

compound_locations = np.empty((len(compounds)))
compound_wts = np.empty((len(compounds)))

concs = np.zeros((len(thermo.products)))

for i, compound in enumerate(compounds):
	compound_locations[i] = thermo.products.index(compound)
	compound_wts[i] = compound_wts_full[int(compound_locations[i])]

n = list(init_reacts.values())
n = n*compound_wts
n = n/np.sum(n)
n = n/compound_wts
b0 = thermo.b0

for i, compound in enumerate(compounds):
	concs[int(compound_locations[i])] = n[i]

n = concs
n_moles = np.sum(n)

p = om.Problem()
p.model = Properties(thermo=thermo)
p.model.set_input_defaults('n', n)
p.model.set_input_defaults('n_moles', n_moles)
p.model.set_input_defaults('b0', b0)

T_range = np.linspace(50,2000,3902) #units are Kelvin
T_range = np.array([300, 310, 1050])


Cp = np.empty((len(T_range)))

for i, T in enumerate(T_range):

	p.model.set_input_defaults('T', T, units='degK')
	p.setup()
	p.run_model()

	Cp[i] = p['Cp'] # units are cal/(g*degK) 

print(Cp)   

# f = open( 'AIR_MIX_Cp.py', 'w' )
# f.write('import numpy as np\n')
# f.write('Cp = ' + repr(Cp))
# f.close()