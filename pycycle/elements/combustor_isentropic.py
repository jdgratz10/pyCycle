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

    def setup(self):
        design = self.options['design']
        statics = self.options['statics']
        gamma = self.options['gamma']

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
        vit_flow = SetTotal(mode='h', fl_name="Fl_O:tot", gamma=gamma)
        self.add_subsystem('vitiated_flow', vit_flow, promotes_outputs=['Fl_O:*'])
        self.connect("mix_fuel.mass_avg_h", "vitiated_flow.h")
        self.connect('mix_fuel.MW_out', 'vitiated_flow.MW.MW')
        self.connect("p_loss.Pt_out","vitiated_flow.P")

        if statics:
            if design:
                # Calculate static properties.
                out_stat = SetStatic(mode="MN", fl_name="Fl_O:stat", computation_mode='isentropic', thermo_data=janaf, gamma=gamma)
                prom_in = ['MN']
                prom_out = ['Fl_O:stat:*']
                self.add_subsystem('out_stat', out_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)

                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('Wout','out_stat.W')

            else:
                # Calculate static properties.
                out_stat = SetStatic(mode="area", fl_name="Fl_O:stat", computation_mode='isentropic', thermo_data=janaf, gamma=gamma)
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

    p = om.Problem()
    p.model = om.Group()
    p.model.add_subsystem('comp', MixFuel(), promotes=['*'])

    p.model.add_subsystem('d1', om.IndepVarComp('Fl_I:stat:W', val=1.0, units='lbm/s', desc='weight flow'),
                          promotes=['*'])
    p.model.add_subsystem('d2', om.IndepVarComp('Fl_I:FAR', val=0.2, desc='Fuel to air ratio'), promotes=['*'])
    p.model.add_subsystem('d3', om.IndepVarComp('Fl_I:tot:h', val=1.0, units='Btu/lbm', desc='total enthalpy'),
                          promotes=['*'])

    p.setup(check=False, force_alloc_complex=True)
    p.run_model()

    p.check_partials(compact_print=True, method='cs')
