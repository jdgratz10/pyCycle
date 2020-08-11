import numpy as np
import sys
from pycycle.constants import R_UNIVERSAL_SI, AIR_MIX, R_UNIVERSAL_ENG
from species_data import Thermo
from pycycle.cea.thermo_data import janaf

np.set_printoptions(threshold=sys.maxsize)

thermo_data = janaf
init_reacts = AIR_MIX

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


T_range = np.linspace(50,2000,3902) #units are Kelvin

H = np.empty((len(T_range)))

for i, T in enumerate(T_range):
    H0_T_full = thermo.H0([T])
    H0_T = np.empty((len(compounds)))

    for ct, location in enumerate(compound_locations):
    	H0_T[ct] = H0_T_full[int(location)]
    
    H[i] = R_UNIVERSAL_ENG*np.sum(n*H0_T)*T #units="cal/g"
print(H)

f = open( 'AIR_MIX_enthalpy.py', 'w' )
f.write('import numpy as np\n')
f.write('H = ' + repr(H))
f.close()