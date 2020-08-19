""" Class definition for Combustor."""

import numpy as np

import openmdao.api as om

from pycycle.isentropic.set_total import SetTotal
from pycycle.cea.set_static import SetStatic
from pycycle.cea.species_data import Thermo, janaf
from pycycle.constants import AIR_FUEL_MIX, AIR_MIX
from pycycle.elements.duct import PressureLoss
from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough
from pycycle.constants import R_UNIVERSAL_ENG
from pycycle.isentropic.entropy_map_data import AIR_MIX_entropy


class MixFuel(om.ExplicitComponent):
    """
    MixFuel calculates fuel and air mixture.
    """

    def initialize(self):
        self.options.declare('fuel_type', default="JP-7",
                             desc='Type of fuel.')

    def setup(self):

        # inputs
        self.add_input('Fl_I:stat:W', val=0.0, desc='weight flow', units='lbm/s')
        self.add_input('Fl_I:FAR', val=0.0, desc='Fuel to air ratio')
        self.add_input('Fl_I:tot:h', val=0.0, desc='total enthalpy', units='Btu/lbm')
        self.add_input('Fl_I:tot:R', val=.05, desc='specific gas constant of incoming flow', units='Btu/(lbm*degR)')
        self.add_input('fuel_MW', val=1.1, desc='molecular weight of fuel', units='lbm/mol')

        # outputs
        self.add_output('mass_avg_h', shape=1, units='Btu/lbm',
                        desc="mass flow rate averaged specific enthalpy")
        self.add_output('Wout', shape=1, units="lbm/s", desc="total massflow out")
        self.add_output('Wfuel', shape=1, units="lbm/s", desc="total fuel massflow out")
        self.add_output('MW_out', units='lbm/mol', desc='total molecular weight of flow out')

        self.declare_partials('mass_avg_h', ['Fl_I:FAR', 'Fl_I:tot:h'])
        self.declare_partials('Wout', ['Fl_I:stat:W', 'Fl_I:FAR'])
        self.declare_partials('Wfuel', ['Fl_I:stat:W', 'Fl_I:FAR'])
        self.declare_partials('MW_out', ['Fl_I:FAR', 'Fl_I:tot:R', 'fuel_MW'], method='cs')

    def compute(self, inputs, outputs):
        FAR = inputs['Fl_I:FAR']
        Wair = inputs['Fl_I:stat:W']
        R_air = inputs['Fl_I:tot:R']
        fuel_MW = inputs['fuel_MW']

        air_MW = R_UNIVERSAL_ENG/R_air

        Wout = Wair * (1+FAR)
        Wfuel = Wair * FAR

        x_air = Wair/Wout
        x_fuel = Wfuel/Wout

        y_air = (x_air/air_MW)/(x_air/air_MW + x_fuel/fuel_MW)
        y_fuel = (x_fuel/fuel_MW)/(x_fuel/fuel_MW + x_air/air_MW)

        MW_out = y_air*air_MW + y_fuel*fuel_MW

        self.fuel_ht = 0  # makes ht happy

        outputs['mass_avg_h'] = (inputs['Fl_I:tot:h']+FAR*self.fuel_ht)/(1+FAR)

        outputs['Wout'] = Wout

        outputs['Wfuel'] = Wfuel

        outputs['MW_out'] = MW_out

    def compute_partials(self, inputs, J):
        FAR = inputs['Fl_I:FAR']
        W = inputs['Fl_I:stat:W']
        ht = inputs['Fl_I:tot:h']
        R_air = inputs['Fl_I:tot:R']
        fuel_MW = inputs['fuel_MW']

        # AssertionError: 4.2991138611171866e-05 not less than or equal to 1e-05 : DESIGN.burner.mix_fuel: init_prod_amounts  w.r.t Fl_I:tot:n
        J['mass_avg_h', 'Fl_I:FAR'] = -ht/(1+FAR)**2 + self.fuel_ht/(1+FAR)**2  # - self.fuel_ht*FAR/(1+FAR)**2
        J['mass_avg_h', 'Fl_I:tot:h'] = 1.0/(1.0 + FAR)

        J['Wout', 'Fl_I:stat:W'] = (1.0 + FAR)
        J['Wout', 'Fl_I:FAR'] = W

        J['Wfuel', 'Fl_I:stat:W'] = FAR
        J['Wfuel', 'Fl_I:FAR'] = W

        J['MW_out', 'Fl_I:FAR'] = (((R_air*fuel_MW + R_UNIVERSAL_ENG*FAR)*R_UNIVERSAL_ENG*fuel_MW) - (R_UNIVERSAL_ENG*fuel_MW + FAR*R_UNIVERSAL_ENG*fuel_MW)*R_UNIVERSAL_ENG)/(R_air*fuel_MW + R_UNIVERSAL_ENG*FAR)**2
        J['MW_out', 'Fl_I:tot:R'] = -R_UNIVERSAL_ENG*fuel_MW*(1 + FAR)/(R_air*fuel_MW + R_UNIVERSAL_ENG*FAR)**2 * fuel_MW
        J['MW_out', 'fuel_MW'] = ((R_air*fuel_MW + R_UNIVERSAL_ENG*FAR)*R_UNIVERSAL_ENG*(1 + FAR) - R_UNIVERSAL_ENG*fuel_MW*(1 + FAR)*R_air)/(R_air*fuel_MW + R_UNIVERSAL_ENG*FAR)**2


class IsentropicCombustor(om.Group):
    """
    A combustor that adds a fuel to an incoming flow mixture and burns it

    --------------
    Flow Stations
    --------------
    Fl_I
    Fl_O

    -------------
    Design
    -------------
        inputs
        --------
        Fl_I:FAR
        dPqP
        MN

        outputs
        --------
        Wfuel


    -------------
    Off-Design
    -------------
        inputs
        --------
        Fl_I:FAR
        dPqP
        area

        outputs
        --------
        Wfuel

    """

    def initialize(self):
        self.options.declare('inflow_thermo_data', default=None,
                             desc='Thermodynamic data set for incoming flow. This only needs to be set if different thermo data is used for incoming flow and outgoing flow.', recordable=False)
        self.options.declare('thermo_data', default=janaf,
                             desc='Thermodynamic data set for flow. This is used for incoming and outgoing flow unless inflow_thermo_data is set, in which case it is used only for outgoing flow.', recordable=False)
        self.options.declare('inflow_elements', default=AIR_MIX,
                             desc='set of elements present in the air flow')
        self.options.declare('air_fuel_elements', default=AIR_FUEL_MIX,
                             desc='set of elements present in the fuel')
        self.options.declare('design', default=True,
                             desc='Switch between on-design and off-design calculation.')
        self.options.declare('statics', default=True,
                             desc='If True, calculate static properties.')
        self.options.declare('fuel_type', default="JP-7",
                             desc='Type of fuel.')
        self.options.declare('gamma', default=1.4, 
                              desc='ratio of specific heats, only used in isentropic mode')
        self.options.declare('S_data', default=AIR_MIX_entropy, desc='entropy property data')
        self.options.declare('h_base', default=0, desc='enthalpy at base temperature (units are cal/g)')
        self.options.declare('T_base', default=302.4629819, desc='base temperature (units are degK)')
        self.options.declare('Cp', default=0.24015494, desc='constant specific heat that is assumed (units are cal/(g*degK)')

    def setup(self):
        design = self.options['design']
        statics = self.options['statics']
        gamma = self.options['gamma']
        S_data=self.options['S_data']
        h_base = self.options['h_base']
        T_base = self.options['T_base']
        Cp = self.options['Cp']

        # Create combustor flow station
        in_flow = FlowIn(fl_name='Fl_I', num_prods=1, num_elements=1)
        self.add_subsystem('in_flow', in_flow, promotes=['Fl_I:tot:*', 'Fl_I:stat:*'])

        # Perform combustor engineering calculations
        self.add_subsystem('mix_fuel',
                           MixFuel(),
                           promotes=['Fl_I:stat:W','Fl_I:FAR', 'Fl_I:tot:h', 'Wfuel', 'Wout', 'Fl_I:tot:R', 'fuel_MW'])#MW_out, fuel_MW

        # Pressure loss
        prom_in = [('Pt_in', 'Fl_I:tot:P'),'dPqP']
        self.add_subsystem('p_loss', PressureLoss(), promotes_inputs=prom_in)

        # Calculate vitiated flow station properties
        # vit_flow = SetTotal(mode='h', fl_name="Fl_O:tot", gamma=gamma)
        vit_flow = SetTotal(mode='h', fl_name="Fl_O:tot", gamma=gamma, S_data=S_data, h_base=h_base, T_base=T_base, Cp=Cp)
        self.add_subsystem('vitiated_flow', vit_flow, promotes_outputs=['Fl_O:*'])
        self.connect("mix_fuel.mass_avg_h", "vitiated_flow.h")
        self.connect('mix_fuel.MW_out', 'vitiated_flow.MW.MW')
        self.connect("p_loss.Pt_out","vitiated_flow.P")

        if statics:
            if design:
                # Calculate static properties.
                # out_stat = SetStatic(mode="MN", fl_name="Fl_O:stat", computation_mode='isentropic', thermo_data=janaf, gamma=gamma)
                out_stat = SetStatic(mode="MN", fl_name="Fl_O:stat", computation_mode='isentropic', thermo_data=janaf, gamma=gamma, S_data=S_data, h_base=h_base, T_base=T_base, Cp=Cp)
                prom_in = ['MN']
                prom_out = ['Fl_O:stat:*']
                self.add_subsystem('out_stat', out_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)

                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('Wout','out_stat.W')

            else:
                # Calculate static properties.
                # out_stat = SetStatic(mode="area", fl_name="Fl_O:stat", computation_mode='isentropic', thermo_data=janaf, gamma=gamma)
                out_stat = SetStatic(mode="area", fl_name="Fl_O:stat", computation_mode='isentropic', thermo_data=janaf, gamma=gamma, S_data=S_data, h_base=h_base, T_base=T_base, Cp=Cp)
                prom_in = ['area']
                prom_out = ['Fl_O:stat:*']
                self.add_subsystem('out_stat', out_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)

                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('Wout','out_stat.W')

        else:
            self.add_subsystem('W_passthru', PassThrough('Wout', 'Fl_O:stat:W', 1.0, units= "lbm/s"),
                               promotes=['*'])

        self.add_subsystem('FAR_pass_thru', PassThrough('Fl_I:FAR', 'Fl_O:FAR', 0.0),
                           promotes=['*'])
        self.set_input_defaults('Fl_I:tot:n', units=None)


if __name__ == "__main__":

    from pycycle.connect_flow import connect_flow
    from pycycle.elements.flow_start import FlowStart
    from pycycle.cea import species_data
    from pycycle.constants import AIR_MIX, AIR_FUEL_MIX
    from pycycle.isentropic.AIR_FUEL_MIX_entropy import AIR_FUEL_MIX_entropy

    S_data = AIR_FUEL_MIX_entropy
    T_base = 1297.91021 #degK
    h_base = 86.73820575 #cal/g
    Cp = 0.29460272 #cal/(g*degK)

    p = om.Problem()
    p.model = om.Group()
    p.model.add_subsystem('flow_start', FlowStart(
        thermo_data=species_data.janaf, elements=AIR_MIX, computation_mode='isentropic', gamma=1.37))
    p.model.add_subsystem('combustor', IsentropicCombustor(design=True, thermo_data=species_data.janaf,
        inflow_elements=AIR_MIX, air_fuel_elements=AIR_FUEL_MIX, fuel_type='JP-7', gamma=1.3, T_base=T_base, h_base=h_base, Cp=Cp))

    connect_flow(p.model, 'flow_start.Fl_O', 'combustor.Fl_I')

    p.model.set_input_defaults('flow_start.P', 198.34034952, units='lbf/inch**2')
    p.model.set_input_defaults('flow_start.T', 1190.0900001, units='degR')
    p.model.set_input_defaults('flow_start.W', 100.0, units='lbm/s')
    p.model.set_input_defaults('combustor.Fl_I:FAR', .017, units=None)
    p.model.set_input_defaults('combustor.MN', .02, units=None)
    p.model.set_input_defaults('combustor.dPqP', .03, units=None)

    p.set_solver_print(level=-1)

    p.setup(force_alloc_complex=True)
    p.run_model()
    # p.check_partials(method='cs', compact_print=True)

    print('flow_start.Fl_O:tot:T', p.get_val('flow_start.Fl_O:tot:T', units='degR'), 1190.0900001)
    print('flow_start.Fl_O:tot:P', p.get_val('flow_start.Fl_O:tot:P', units='bar'), 13.67508573)
    print('flow_start.Fl_O:tot:h', p.get_val('flow_start.Fl_O:tot:h', units='cal/g'), 88.23638223)
    print('flow_start.Fl_O:tot:S', p.get_val('flow_start.Fl_O:tot:S', units='cal/(g*degK)'), 1.65712206)
    print('flow_start.Fl_O:tot:gamma', p.get_val('flow_start.Fl_O:tot:gamma', units=None), 1.36872902)
    print('flow_start.Fl_O:tot:Cp', p.get_val('flow_start.Fl_O:tot:Cp', units='cal/(g*degK)'), 0.25466771)
    print('flow_start.Fl_O:tot:Cv', p.get_val('flow_start.Fl_O:tot:Cv', units='cal/(g*degK)'), 0.18606141)
    print('flow_start.Fl_O:tot:rho', p.get_val('flow_start.Fl_O:tot:rho', units='lbm/inch**3'), 0.00026032)
    print()
    print('flow_start.Fl_O:stat:T', p.get_val('flow_start.Fl_O:stat:T', units='degR'), 1137.36400187)
    print('flow_start.Fl_O:stat:P', p.get_val('flow_start.Fl_O:stat:P', units='bar'), 11.56433131)
    print('flow_start.Fl_O:stat:h', p.get_val('flow_start.Fl_O:stat:h', units='cal/g'), 80.80131257)
    print('flow_start.Fl_O:stat:S', p.get_val('flow_start.Fl_O:stat:S', units='cal/(g*degK)'), 1.65712206)
    print('flow_start.Fl_O:stat:gamma', p.get_val('flow_start.Fl_O:stat:gamma', units=None), 1.37208827)
    print('flow_start.Fl_O:stat:Cp', p.get_val('flow_start.Fl_O:stat:Cp', units='cal/(g*degK)'), 0.25298835)
    print('flow_start.Fl_O:stat:Cv', p.get_val('flow_start.Fl_O:stat:Cv', units='cal/(g*degK)'), 0.18438195)
    print('flow_start.Fl_O:stat:rho', p.get_val('flow_start.Fl_O:stat:rho', units='lbm/inch**3'), 0.00023034)
    print()
    print('combustor.Fl_O:tot:T', p.get_val('combustor.Fl_O:tot:T', units='degR'), 2336.38034075)
    print('combustor.Fl_O:tot:P', p.get_val('combustor.Fl_O:tot:P', units='bar'), 13.26483316)
    print('combustor.Fl_O:tot:h', p.get_val('combustor.Fl_O:tot:h', units='cal/g'), 86.76143779)
    print('combustor.Fl_O:tot:S', p.get_val('combustor.Fl_O:tot:S', units='cal/(g*degK)'), 1.86403665)
    print('combustor.Fl_O:tot:gamma', p.get_val('combustor.Fl_O:tot:gamma', units=None), 1.30377937)
    print('combustor.Fl_O:tot:Cp', p.get_val('combustor.Fl_O:tot:Cp', units='cal/(g*degK)'), 0.29460609)
    print('combustor.Fl_O:tot:Cv', p.get_val('combustor.Fl_O:tot:Cv', units='cal/(g*degK)'), 0.22596247)
    print('combustor.Fl_O:tot:rho', p.get_val('combustor.Fl_O:tot:rho', units='lbm/inch**3'), 0.00012855)
    print()
    print('combustor.Fl_O:stat:T', p.get_val('combustor.Fl_O:stat:T', units='degR'), 2336.23838313)
    print('combustor.Fl_O:stat:P', p.get_val('combustor.Fl_O:stat:P', units='bar'), 13.26137482)
    print('combustor.Fl_O:stat:h', p.get_val('combustor.Fl_O:stat:h', units='cal/g'), 86.73820575)
    print('combustor.Fl_O:stat:S', p.get_val('combustor.Fl_O:stat:S', units='cal/(g*degK)'), 1.86403665)
    print('combustor.Fl_O:stat:gamma', p.get_val('combustor.Fl_O:stat:gamma', units=None), 1.3037839)
    print('combustor.Fl_O:stat:Cp', p.get_val('combustor.Fl_O:stat:Cp', units='cal/(g*degK)'), 0.29460272)
    print('combustor.Fl_O:stat:Cv', p.get_val('combustor.Fl_O:stat:Cv', units='cal/(g*degK)'), 0.2259591)
    print('combustor.Fl_O:stat:rho', p.get_val('combustor.Fl_O:stat:rho', units='lbm/inch**3'), 0.00012852)

