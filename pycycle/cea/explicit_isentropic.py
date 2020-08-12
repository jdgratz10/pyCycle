import numpy as np

from openmdao.api import ExplicitComponent

from pycycle.constants import R_UNIVERSAL_SI

class ExplicitIsentropic(ExplicitComponent):

    def initialize(self):
        self.options.declare('gamma', default=1.4, desc='ratio of specific heats')
        self.options.declare('for_statics', default=False, values=(False, 'Ps', 'MN', 'area'), desc='type of static calculation to perform')
        self.options.declare('mode', default='T', values=('T', 'h', 'S'), desc='mode to calculate in')
        self.options.declare('fl_name', default="flow", desc='flowstation name of the output flow variables')

    def setup(self):

        for_statics = self.options['for_statics']
        mode = self.options['mode']
        fl_name = self.options['fl_name']

        self.add_input('R', val = 4400.0, units='J/(kg*K)', desc='molecular weight of gas mixture')
        self.add_input('Cp', val=1.0, units='cal/(g*degK)', desc='specific heat at constant pressure')
        self.add_input('T', val=100., units='degK', desc='entropy')

        self.add_output('rho', val=1.0, units="kg/m**3", desc="density")

        if for_statics == 'MN':
            self.add_input('MN', val=.9, desc="computed mach number")
            self.add_input('W', val=.1, desc="mass flow rate", units="kg/s")
            self.add_input('P', val=1.013, units='bar', desc='pressure')

            self.add_output('area', val=1.2, units="m**2", desc="computed area")

            self.declare_partials('V', ('MN', 'R', 'T'))
            self.declare_partials('area', ('W', 'R', 'T', 'P', 'MN'))
            self.declare_partials('rho', ('P', 'R', 'T'))
            

        elif for_statics == 'Ps':
            self.add_input('P', val=1.013, units='bar', desc='pressure')
            self.add_input('Tt', val=101.1, units="degK", desc="Total enthalpy reference condition")
            self.add_input('W', val=0.1, desc="mass flow rate", units="kg/s")

            self.add_output('area', val=1.2, units="m**2", desc="computed area")
            self.add_output('MN', val=.9, desc="computed mach number")

            self.declare_partials('MN', ('Tt', 'T'))
            self.declare_partials('V', ('R', 'T', 'Tt'))
            self.declare_partials('area', ('W', 'R', 'T', 'P', 'Tt'))
            self.declare_partials('rho', ('P', 'R', 'T'))

        elif for_statics == 'area':
            self.add_input('W', val=.1, desc="mass flow rate", units="kg/s")
            self.add_input('area', val=1.2, units="m**2", desc="computed area")
            
            if mode == 'T':
                self.add_input('Tt', val=101.1, units="degK", desc="Total temperature")

                self.add_output('MN', val=.9, desc="computed mach number")
                self.add_output('P', val=101300, units='Pa', desc='pressure')

                self.declare_partials('MN', ('Tt', 'T'))
                self.declare_partials('V', 'Tt')
                self.declare_partials('P', ('W', 'R', 'T', 'Tt', 'area'))   
                self.declare_partials('rho', ('R', 'T', 'W', 'Tt', 'area'))             

            elif mode == 'h':
                self.add_input('Tt', val=1.1, units="degK", desc="Total enthalpy reference condition")

                self.add_output('MN', val=.9, desc="computed mach number")
                self.add_output('P', val=101300, units='Pa', desc='pressure')

                self.declare_partials('MN', ('Tt', 'T'))
                self.declare_partials('V', 'Tt')
                self.declare_partials('P', ('W', 'R', 'T', 'Tt', 'area')) 
                self.declare_partials('rho', ('R', 'T', 'W', 'Tt', 'area'))

            else:
                self.add_input('P', val=101300, units='Pa', desc='pressure')

                self.add_output('MN')

                self.declare_partials('V', ('W', 'area', 'P'))
                self.declare_partials('MN', ('R', 'T', 'W', 'area', 'P')) 
                self.declare_partials('rho', ('P', 'R', 'T'))

            self.declare_partials('V', ('T', 'R'))


        if for_statics is False:
            self.add_input('P', val=1.013, units='bar', desc='pressure')

            self.add_output('Cv', val=1.1, units="Btu/(lbm*degR)", desc="Specific heat at constant volume")

            self.declare_partials('Cv', 'Cp')
            self.declare_partials('rho', ('P', 'R', 'T'))

        else:
            self.add_output('Cv', val=1.1, units="cal/(g*degK)", desc="Specific heat at constant volume")
            self.add_output('V', val=1.1, units="m/s", desc="computed speed", res_ref=1e3)
            self.add_output('Vsonic', val=1.1, units="m/s", desc="computed speed of sound", res_ref=1e3)

            self.declare_partials('Cv', 'Cp', val=1)
            self.declare_partials('Vsonic', ('R', 'T'))

    def compute(self, inputs, outputs):

        for_statics = self.options['for_statics']
        mode = self.options['mode']
        fl_name = self.options['fl_name']
        gamma = self.options['gamma']

        ## get necessary input values:

        R = inputs['R']
        T = inputs['T']
        Cp = inputs['Cp']

        # print(self.pathname)
        # print(T)
        # print()

        ## complete the necessary calculations:

        Cv = Cp/gamma

        if for_statics is False:

            P = inputs['P']

            rho = 1e5*P/(R*T) ##necessary 1e5 for unit conversion

            outputs['Cv'] = Cv
            outputs['rho'] = rho

        else:
            
            Vsonic = (gamma*R*T)**.5

            outputs['Cv'] = Cv
            outputs['Vsonic'] = Vsonic

        if for_statics == 'Ps':

            P = inputs['P']
            W = inputs['W']
            Tt = inputs['Tt']

            if T <= Tt:
                MN = (2/(gamma - 1) * (Tt/T - 1))**.5
            else:
                print('Warning: Tt is less than T, MN calculated is unphysical')
                MN = (2/(gamma - 1) * (1 - Tt/T))**.5

            V = MN*Vsonic
            area = W*R*T/(1e5*P*V) # based on definition of mass flow rate, 1e5 is for unit conversion
            rho = 1e5*P/(R*T) ##necessary 1e5 for unit conversion

            outputs['MN'] = MN
            outputs['V'] = V
            outputs['area'] = area
            outputs['rho'] = rho

        elif for_statics == 'MN':

            MN = inputs['MN']
            P = inputs['P']
            W = inputs['W']

            V = MN*Vsonic
            area = W*R*T/(1e5*P*V) # based on definition of mass flow rate, 1e5 is for unit conversion
            rho = 1e5*P/(R*T) ##necessary 1e5 for unit conversion

            outputs['V'] = V
            outputs['area'] = area
            outputs['rho'] = rho

        elif for_statics == 'area':

            W = inputs['W']
            area = inputs['area']

            if mode =='T' or mode == 'h':

                Tt = inputs['Tt']

                if T < Tt:
                    MN = (2/(gamma - 1) * (Tt/T - 1))**.5
                else:
                    print('Warning: Tt is less than T, MN calculated is unphysical')
                    MN = (2/(gamma - 1) * (1 - Tt/T))**.5

                V = MN*Vsonic
                P = W*R*T/(area*V)
                rho = P/(R*T) ##necessary 1e5 for unit conversion

                outputs['MN'] = MN
                outputs['V'] = V
                outputs['P'] = P
                outputs['rho'] = rho

            else:
                
                P = inputs['P']

                V = R*T*W/(area*P)
                MN = V/Vsonic
                rho = P/(R*T) ##necessary 1e5 for unit conversion

                outputs['rho'] = rho

                outputs['V'] = V
                outputs['MN'] = MN
                rho = P/(R*T) ##necessary 1e5 for unit conversion

                outputs['rho'] = rho

            
    def compute_partials(self, inputs, J):

        for_statics = self.options['for_statics']
        mode = self.options['mode']
        fl_name = self.options['fl_name']
        gamma = self.options['gamma']

        ## get necessary input values:

        R = inputs['R']
        T = inputs['T']
        Cp = inputs['Cp']

        if for_statics is False:

            P = inputs['P']

            J['rho', 'P'] = 1e5/(R*T)
            J['rho', 'R'] = -1e5*P/(T*R**2)
            J['rho', 'T'] = -1e5*P/(R*T**2)
            J['Cv', 'Cp'] = 1/gamma

        else:
            
            Vsonic = (gamma*R*T)**.5

            dVsonic_dR = .5*(gamma*T/R)**.5
            dVsonic_dT = .5*(gamma*R/T)**.5

            J['Vsonic', 'R'] = dVsonic_dR
            J['Vsonic', 'T'] = dVsonic_dT
            J['Cv', 'Cp'] = 1/gamma

        if for_statics == 'Ps':

            P = inputs['P']
            W = inputs['W']
            Tt = inputs['Tt']

            if T <= Tt:
                MN = (2/(gamma - 1) * (Tt/T - 1))**.5

                dMN_dTt = .5*(2/(gamma - 1) * Tt/T - 2/(gamma - 1))**(-.5) * 2/(T*(gamma - 1))
                dMN_dT = .5*(2/(gamma - 1) * (Tt/T) - 2/(gamma - 1))**(-.5) * (-2/(gamma - 1) * Tt/(T**2))
                temp = gamma*R*T*(Tt/T - 1)/(gamma - 1)
                dArea_dT = (R*W/(2**.5 * P*temp**.5) - R*T*W*(gamma*R*(Tt/T - 1)/(gamma - 1) - gamma*R*Tt/((gamma - 1)*T))/(2*2**.5 * P*temp**1.5))/1e5
                dArea_dR = (T*W/(2**.5 * P*temp**.5) - gamma*R*T**2*W*(Tt/T - 1)/(2*2**.5 * (gamma - 1)*P*temp**1.5))/1e5
                dArea_dTt = (-gamma*R**2*T*W/(2*2**.5 * (gamma - 1)*P*temp**1.5))/1e5
            else:
                print('Warning: Tt is less than T, MN calculated is unphysical')
                MN = (2/(gamma - 1) * (1 - Tt/T))**.5

                dMN_dTt = .5*(2/(gamma - 1) - 2/(gamma - 1) * (Tt/T))**(-.5) * (-2/(gamma - 1)/T)
                dMN_dT = .5*(2/(gamma - 1) - 2/(gamma - 1) * (Tt/T))**(-.5) * (2/(gamma - 1) * (Tt/T**2))
                temp = -gamma*R*T*(Tt/T - 1)/(gamma - 1)
                dArea_dT = (R*W/(2**.5 * P*temp**.5) + R*T*W*(gamma*R*(Tt/T - 1)/(gamma - 1) - gamma*R*Tt/((gamma - 1)*T))/(2*2**.5 * P*temp**1.5))/1e5
                dArea_dR = (T*W/(2**.5 * P*temp**.5) + gamma*R*T**2*W*(Tt/T - 1)/(2*2**.5 * (gamma - 1)*P*temp**1.5))/1e5
                dArea_dTt = (gamma*R**2*T*W/(2*2**.5 * (gamma - 1)*P*temp**1.5))/1e5

            V = MN*Vsonic
            area = W*R*T/(1e5*P*V) # based on definition of mass flow rate, 1e5 is for unit conversion

            dV_dR = MN*dVsonic_dR
            dV_dT = MN*dVsonic_dT + dMN_dT*Vsonic
            dV_dTt = dMN_dTt*Vsonic

            J['MN', 'Tt'] = dMN_dTt
            J['MN', 'T'] = dMN_dT
            J['V', 'R'] = dV_dR
            J['V', 'T'] = dV_dT
            J['V', 'Tt'] = dV_dTt
            J['area', 'W'] = R*T/(1e5*P*V)
            J['area', 'R'] = dArea_dR
            J['area', 'T'] = dArea_dT
            J['area', 'P'] = -W*R*T/(1e5*MN*Vsonic*P**2)
            J['area', 'Tt'] = dArea_dTt
            J['rho', 'P'] = 1e5/(R*T)
            J['rho', 'R'] = -1e5*P/(T*R**2)
            J['rho', 'T'] = -1e5*P/(R*T**2)

        elif for_statics == 'MN':

            MN = inputs['MN']
            P = inputs['P']
            W = inputs['W']

            V = MN*Vsonic
            area = W*R*T/(1e5*P*V)

            J['V', 'MN'] = Vsonic
            J['V', 'R'] = MN*dVsonic_dR
            J['V', 'T'] = MN*dVsonic_dT
            J['area', 'W'] = R*T/(1e5*P*V)
            J['area', 'R'] = .5*W*T**.5 / (1e5*MN*P*(gamma*R)**.5)
            J['area', 'T'] = .5*W*R**.5 / (1e5*MN*P*(gamma*T)**.5)
            J['area', 'P'] = -W*R*T/(1e5*V*P**2)
            J['area', 'MN'] = -W*R*T/(1e5*P*Vsonic*MN**2)
            J['rho', 'P'] = 1e5/(R*T)
            J['rho', 'R'] = -1e5*P/(T*R**2)
            J['rho', 'T'] = -1e5*P/(R*T**2)

        elif for_statics == 'area':

            W = inputs['W']
            area = inputs['area']

            if mode =='T' or mode == 'h':

                Tt = inputs['Tt']

                if T <= Tt:
                    MN = (2/(gamma - 1) * (Tt/T - 1))**.5

                    dMN_dTt = .5*(2/(gamma - 1) * Tt/T - 2/(gamma - 1))**(-.5) * 2/(T*(gamma - 1))
                    dMN_dT = .5*(2/(gamma - 1) * (Tt/T) - 2/(gamma - 1))**(-.5) * (-2/(gamma - 1) * Tt/(T**2))
                    temp = (Tt/T - 1)/(gamma - 1)
                    gRT = gamma*R*T
                    dP_dT = -gamma*R**2*T*W/(2*2**.5*area*gRT**1.5*temp**.5) + R*W/(2**.5*area*gRT**.5*temp**.5) + R*Tt*W/(2*2**.5*area*(gamma - 1)*T*gRT**.5*temp**1.5)
                    dP_dTt = -R*W/(2*2**.5*area*(gamma - 1)*gRT**.5*temp**1.5)
                else:
                    print('Warning: Tt is less than T, MN calculated is unphysical')
                    MN = (2/(gamma - 1) * (1 - Tt/T))**.5

                    dMN_dTt = .5*(2/(gamma - 1) - 2/(gamma - 1) * (Tt/T))**(-.5) * (-2/(gamma - 1)/T)
                    dMN_dT = .5*(2/(gamma - 1) - 2/(gamma - 1) * (Tt/T))**(-.5) * (2/(gamma - 1) * (Tt/T**2))
                    temp = (1 - Tt/T)/(gamma - 1)
                    gRT = gamma*R*T
                    dP_dT = -gamma*R**2*T*W/(2*2**.5*area*gRT**1.5*temp**.5) + R*W/(2**.5*area*gRT**.5*temp**.5) - R*Tt*W/(2*2**.5*area*(gamma - 1)*T*gRT**.5*temp**1.5)
                    dP_dTt = R*W/(2*2**.5*area*(gamma - 1)*gRT**.5*temp**1.5)

                V = MN*Vsonic
                P = W*R*T/(area*V)

                J['MN', 'Tt'] = dMN_dTt
                J['MN', 'T'] = dMN_dT
                J['V', 'Tt'] = dV_dTt = Vsonic*dMN_dTt
                J['V', 'T'] = dV_dT = MN*dVsonic_dT + dMN_dT*Vsonic
                J['V', 'R'] = dV_dR = MN*dVsonic_dR
                J['P', 'W'] = R*T/(area*V)
                J['P', 'R'] = .5*W*T**.5 / (area*MN*(gamma*R)**.5)
                J['P', 'T'] = dP_dT
                J['P', 'Tt'] = dP_dTt
                J['P', 'area'] = -W*R*T/(V*area**2)
                J['rho', 'W'] = 1/(area*V)
                J['rho', 'area'] = -W/(area**2 * V)
                J['rho', 'R'] = -W/(area*V**2) * dV_dR
                J['rho', 'T'] = -W/(area*V**2) * dV_dT
                J['rho', 'Tt'] = -W/(area*V**2) * dV_dTt

            else:

                P = inputs['P']

                V = R*T*W/(area*P)

                J['V', 'W'] = dV_dW = R*T/(area*P)
                J['V', 'R'] = dV_dR = T*W/(area*P)
                J['V', 'T'] = dV_dT = R*W/(area*P)
                J['V', 'area'] = dV_dArea = -R*T*W/(P*area**2)
                J['V', 'P'] = dV_dP = -R*T*W/(area*P**2)

                J['MN', 'W'] = dV_dW/Vsonic
                J['MN', 'R'] = (dV_dR*Vsonic - dVsonic_dR*V)/Vsonic**2
                J['MN', 'T'] = (dV_dT*Vsonic - dVsonic_dT*V)/Vsonic**2
                J['MN', 'area'] = dV_dArea/Vsonic
                J['MN', 'P'] = dV_dP/Vsonic

                J['rho', 'T'] = -P/(R*T**2)
                J['rho', 'R'] = -P/(R**2 * T)
                J['rho', 'P'] = 1/(R*T)

if __name__ == "__main__":
    import openmdao.api as om 

    prob = om.Problem()
    prob.model = ExplicitIsentropic(for_statics='area', mode='S')
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)