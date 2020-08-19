import numpy as np

from openmdao.api import ExplicitComponent

from pycycle.constants import P_REF, R_UNIVERSAL_ENG, R_UNIVERSAL_SI, MIN_VALID_CONCENTRATION


class PropsCalcs(ExplicitComponent):
    """computes, S, H, Cp, Cv, gamma, given a converged equilibirum mixture"""

    def initialize(self):
        self.options.declare('thermo', desc='thermodynamic data object', recordable=False)

    def setup(self):

        thermo = self.options['thermo']

        self.add_input('T', val=284., units="degK", desc="Temperature")
        self.add_input('P', val=1., units='bar', desc="Pressure")
        self.add_input('n', val=np.ones(thermo.num_prod),
                       desc="molar concentration of the mixtures, last element is the total molar concentration")
        self.add_input('n_moles', val=1., desc="1/molar_mass for gaseous mixture")

        self.add_output('h', val=1., units="cal/g", desc="enthalpy")
        self.add_output('S', val=1., units="cal/(g*degK)", desc="entropy")
        self.add_output('rho', val=0.0004, units="g/cm**3", desc="density")

        self.add_output('R', val=1., units='(N*m)/(kg*degK)', desc='Specific gas constant')

        # partial derivs setup
        self.declare_partials('h', ['n', 'T'])
        self.declare_partials('S', ['n', 'T', 'P'])
        self.declare_partials('S', 'n_moles')
        self.declare_partials('rho', ['T', 'P', 'n_moles'])

        self.declare_partials('R', 'n_moles', val=R_UNIVERSAL_SI)


    def compute(self, inputs, outputs):
        thermo = self.options['thermo']
        num_prod = thermo.num_prod
        num_element = thermo.num_element

        T = inputs['T']
        P = inputs['P']

        nj = inputs['n'][:num_prod]
        
        n_moles = inputs['n_moles']

        self.H0_T = H0_T = thermo.H0(T)
        self.S0_T = S0_T = thermo.S0(T)
        self.nj_H0 = nj_H0 = nj*H0_T

        outputs['h'] = np.sum(nj_H0)*R_UNIVERSAL_ENG*T

        try:
            val = (S0_T+np.log(n_moles/nj/(P/P_REF)))
        except FloatingPointError:
            P = 1e-5
            val = (S0_T+np.log(n_moles/nj/(P/P_REF)))


        outputs['S'] = R_UNIVERSAL_ENG * np.sum(nj*val)

        outputs['rho'] = P/(n_moles*R_UNIVERSAL_SI*T)*100  # 1 Bar is 100 Kpa

        outputs['R'] = R_UNIVERSAL_SI*n_moles  #(m**3 * Pa)/(mol*degK)

    def compute_partials(self, inputs, J):

        thermo = self.options['thermo']
        num_prod = thermo.num_prod
        num_element = thermo.num_element

        T = inputs['T']
        P = inputs['P']
        nj = inputs['n']
        n_moles = inputs['n_moles']

        H0_T = thermo.H0(T)
        S0_T = thermo.S0(T)
        nj_H0 = nj * H0_T

        dH0_dT = thermo.H0_applyJ(T, 1.)
        dS0_dT = thermo.S0_applyJ(T, 1.)
        dCp0_dT = thermo.Cp0_applyJ(T, 1.)
        sum_nj_R = n_moles*R_UNIVERSAL_SI

        drho_dT = P/(sum_nj_R*T**2)*100
        drho_dnmoles = -P/(n_moles**2*R_UNIVERSAL_SI*T)*100

        J['h', 'T'] = R_UNIVERSAL_ENG*(np.sum(nj*dH0_dT)*T + np.sum(nj*H0_T))
        J['h', 'n'] = R_UNIVERSAL_ENG*T*H0_T

        J['S', 'n'] = R_UNIVERSAL_ENG*(S0_T + np.log(n_moles) - np.log(P/P_REF) - np.log(nj) - 1)

        _trace = np.where(nj <= MIN_VALID_CONCENTRATION+1e-20)
        J['S', 'n'][0, _trace] = 0
        J['S', 'T'] = R_UNIVERSAL_ENG*np.sum(nj*dS0_dT)
        J['S', 'P'] = -R_UNIVERSAL_ENG*np.sum(nj/P)
        J['S', 'n_moles'] = R_UNIVERSAL_ENG*np.sum(nj)/n_moles
        J['rho', 'T'] = -P/(sum_nj_R*T**2)*100
        J['rho', 'n_moles'] = -P/(n_moles**2*R_UNIVERSAL_SI*T)*100
        J['rho', 'P'] = 1/(sum_nj_R*T)*100

if __name__ == "__main__":

    from openmdao.api import Problem, Group, IndepVarComp

    from pycycle.cea import species_data
    from pycycle import constants

    thermo = species_data.Thermo(species_data.co2_co_o2, constants.CO2_CO_O2_MIX)

    p = Problem()
    model = p.model = Group()

    indeps = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
    indeps.add_output('T', 2761.56784655, units='degK')
    indeps.add_output('P', 1.034210, units='bar')
    indeps.add_output('n', val=np.array([2.272e-02, 1.000e-10, 1.136e-02]))
    indeps.add_output('n_moles', val=0.0340831628675)

    model.add_subsystem('calcs', PropsCalcs(thermo=thermo), promotes=['*'])

    p.setup()
    p.run_model()

    p.model.list_inputs()
    p.model.list_outputs()