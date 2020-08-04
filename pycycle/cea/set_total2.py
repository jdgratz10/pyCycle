import numpy as np

from openmdao.api import ExplicitComponent

from pycycle.constants import R_UNIVERSAL_SI

class SetIsentropic(ExplicitComponent):

    def initialize(self):
        self.options.declare('gamma', default=1.4, desc='ratio of specific heats')
        self.options.declare('for_statics', default=False, values=(False, 'Ps', 'MN', 'area'), desc='type of static calculation to perform')
        self.options.declare('mode', default='T', values=('T', 'h', 'S'), desc='mode to calculate in')

    def setup(self):

        for_statics = self.options['for_statics']

        if for_statics is False or for_statics is 'Ps':
            self.add_input('P', val=1.013, units='bar', desc='pressure')

        if mode is 'T':
            self.add_input('T', val=518., units="degK", desc='temperature')

        elif mode is 'h':
            self.add_input('h', val=1., units="J/kg", desc='enthalpy')

        else:
            self.add_input('S', val=1., units='J/(kg*K)', desc='entropy')

        
        self.add_input('gamma', val=1.4)
        self.add_input('n_moles', shape=1)
        
        self.add_input('ht', val=0., units="J/kg", desc="Total enthalpy reference condition")
        self.add_input('hs', val=0., units="J/kg", desc="Static enthalpy")
        self.add_input('W', val=0.0, desc="mass flow rate", units="kg/s")
        self.add_input('rho', val=1.0, desc="density", units="kg/m**3")

        self.add_output('MN', val=1.0, desc="computed mach number")
        self.add_output('V', val=1.0, units="m/s", desc="computed speed", res_ref=1e3)
        self.add_output('Vsonic', val=1.0, units="m/s", desc="computed speed of sound", res_ref=1e3)
        self.add_output('area', val=1.0, units="m**2", desc="computed area")

        self.declare_partials('V', ['ht', 'hs'])
        self.declare_partials('Vsonic', ['gamma', 'n_moles', 'Ts'])
        self.declare_partials('MN', ['gamma', 'n_moles', 'Ts', 'hs', 'ht'])
        self.declare_partials('area', ['rho', 'W', 'hs', 'ht'])

    def compute(self, inputs, outputs):

        outputs['Vsonic'] = Vsonic = np.sqrt(inputs['gamma'] * R_UNIVERSAL_SI * inputs['n_moles'] * inputs['Ts'])

        # If ht < hs then V will be imaginary, so use an inverse relationship to allow solution process to continue
        if inputs['ht'] >= inputs['hs']:
            outputs['V'] = V = np.sqrt(2.0 * (inputs['ht'] - inputs['hs']))
        else:
            # print('Warning: in', self.pathname, 'ht < hs, inverting relationship to get a real velocity, ht = ', inputs['ht'], 'hs = ', inputs['hs'])
            outputs['V'] = V = np.sqrt(2.0 * (inputs['hs'] - inputs['ht']))

        outputs['MN'] = V / Vsonic
        outputs['area'] = inputs['W'] / (inputs['rho'] * V)


    def compute_partials(self, inputs, J):

        Vsonic = np.sqrt(inputs['gamma'] * R_UNIVERSAL_SI * inputs['n_moles'] * inputs['Ts'])

        J['Vsonic','gamma'] = Vsonic / (2.0 * inputs['gamma'])
        J['Vsonic','n_moles'] = Vsonic / (2.0 * inputs['n_moles'])
        J['Vsonic','Ts'] = Vsonic / (2.0 * inputs['Ts'])

        if inputs['ht'] >= inputs['hs']:
            V = np.sqrt(2.0 * (inputs['ht'] - inputs['hs']))
            J['V','ht'] = 1.0 / V
            J['V','hs'] = -1.0 / V
        else:
            V = np.sqrt(2.0 * (inputs['hs'] - inputs['ht']))
            J['V','hs'] = 1.0 / V
            J['V','ht'] = -1.0 / V

        J['MN','ht'] = 1.0 / Vsonic * J['V','ht']
        J['MN','hs'] = 1.0 / Vsonic * J['V','hs']
        J['MN','gamma'] = -V / Vsonic**2 * J['Vsonic','gamma']
        J['MN','n_moles'] = -V / Vsonic**2 * J['Vsonic','n_moles']
        J['MN','Ts'] = -V / Vsonic**2 * J['Vsonic','Ts']

        J['area','W'] = 1.0 / (inputs['rho'] * V)
        J['area','rho'] = -inputs['W'] / (inputs['rho']**2 * V)
        J['area','ht'] = -inputs['W'] / (inputs['rho'] * V**2) * J['V','ht']
        J['area','hs'] = -inputs['W'] / (inputs['rho'] * V**2) * J['V','hs']

if __name__ == "__main__":
    import openmdao.api as om 

    prob = om.Problem()
    prob.model = SetIsentropic()
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)