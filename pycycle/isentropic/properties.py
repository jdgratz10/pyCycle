import openmdao.api as om

from pycycle.isentropic.entropy_map_data import AIR_MIX_entropy

class PropertyMap(om.Group):
    """runs design and off-design mode Turbine map calculations"""

    def initialize(self):
        self.options.declare('map_data', default=AIR_MIX_entropy)
        self.options.declare('interp_method', default='slinear')
        self.options.declare('extrap', default=True)

    def setup(self):

        map_data = self.options['map_data']
        method = self.options['interp_method']
        extrap = self.options['extrap']

        params = map_data.param_data
        outputs = map_data.output_data

        # Define map which will be used
        readmap = om.MetaModelStructuredComp(method=method, extrapolate=extrap)

        for p in params:
            readmap.add_input(p['name'], val=p['default'], units=p['units'],
                        training_data=p['values'])
        for o in outputs:
            readmap.add_output(o['name'], val=o['default'], units=o['units'],
                        training_data=o['values'])

        self.add_subsystem('readMap', readmap, promotes_inputs=['*'],
                                promotes_outputs=['*'])

if __name__ == "__main__":

    from pycycle.isentropic.enthalpy_map_data import AIR_MIX_enthalpy
    from pycycle.isentropic.AIR_FUEL_MIX_enthalpy import AIR_FUEL_MIX_enthalpy

    p = om.Problem()
    p.model.add_subsystem('map', PropertyMap(
        map_data=AIR_FUEL_MIX_enthalpy), promotes=['*'])

    p.model.set_input_defaults('T', 660, units='degK')
    # p.model.set_input_defaults('h', 0, units='cal/g')
    # p.model.set_input_defaults('P', 1.013, units='bar')

    p.setup(check=True)
    p.run_model()
    # p.check_partials(compact_print=True)

    print(p.get_val('T', units='degK')) #AIR_FUEL_MIX T_base = 1050 K
    print(p.get_val('h', units='cal/g')) #AIR_FUEL_MIX h_base = 1341.3817795 cal/g