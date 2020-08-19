import numpy as np
import sys
from pycycle.constants import R_UNIVERSAL_SI, AIR_MIX, R_UNIVERSAL_ENG, AIR_FUEL_MIX
from pycycle.cea.species_data import Thermo
from pycycle.cea.thermo_data import janaf

np.set_printoptions(threshold=sys.maxsize)

thermo_data = janaf
init_reacts = AIR_FUEL_MIX

thermo = Thermo(thermo_data, init_reacts=init_reacts)
compounds = init_reacts.keys()
compound_wts_full = thermo.wt_mole

compound_locations = np.empty((len(compounds)))
compound_wts = np.empty((len(compounds)))

for i, compound in enumerate(compounds):
	compound_locations[i] = thermo.products.index(compound)
	compound_wts[i] = compound_wts_full[int(compound_locations[i])]

n = list(init_reacts.values())
n = n*compound_wts
n = n/np.sum(n)
n = n/compound_wts
n_moles = np.sum(n)


T_range = np.linspace(50,2000,3902) #units are Kelvin
P_range = np.linspace(.5, 21, 83) #units are bar

AIR_MIX_MW = 28.9651784 #g/mol or kg.kmol?
R = R_UNIVERSAL_SI/AIR_MIX_MW

S = np.empty((len(T_range),len(P_range)))

for i, T in enumerate(T_range):
    S0_T_full = thermo.S0([T])
    S0_T = np.empty((len(compounds)))

    for ct, location in enumerate(compound_locations):
    	S0_T[ct] = S0_T_full[int(location)]
    
    for j, P in enumerate(P_range):
    	S[i, j] = R_UNIVERSAL_ENG*np.sum(n*(S0_T-np.log(n)+np.log(n_moles)-np.log(P))) #units="cal/(g*degK)"

f = open( 'AIR_FUEL_MIX_entropy.py', 'w' )
f.write('import numpy as np\n')
f.write('S = ' + repr(S))
f.close()