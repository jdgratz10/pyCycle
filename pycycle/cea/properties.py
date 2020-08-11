import openmdao.api as om

from pycycle.cea.entropy_map_data import AIR_MIX_entropy
from pycycle.cea.cp_map_data import AIR_MIX_Cp
from pycycle.cea.enthalpy_map_data import AIR_MIX_enthalpy
from pycycle.cea.entropy_map_data import AIR_MIX_entropy

class PropertyMap(om.Group):
    """runs design and off-design mode Turbine map calculations"""

    def initialize(self):
        self.options.declare('map_data', default=AIR_MIX_entropy)
        self.options.declare('interp_method', default='slinear')
        self.options.declare('extrap', default=True)
        self.options.declare('get_temp', default=False, values=(True, False), desc='Finds temperature from enthalpy when set to true. Cannot be used for entropy.')

    def setup(self):

        map_data = self.options['map_data']
        method = self.options['interp_method']
        extrap = self.options['extrap']
        get_temp = self.options['get_temp']

        params = map_data.param_data
        outputs = map_data.output_data

        # Define map which will be used
        readmap = om.MetaModelStructuredComp(method=method, extrapolate=extrap)

        if not get_temp:
            for p in params:
                readmap.add_input(p['name'], val=p['default'], units=p['units'],
                            training_data=p['values'])
            for o in outputs:
                readmap.add_output(o['name'], val=o['default'], units=o['units'],
                            training_data=o['values'])

        else:
            for p in params:
                readmap.add_output(p['name'], val=p['default'], units=p['units'],
                            training_data=p['values'])
            for o in outputs:
                readmap.add_input(o['name'], val=o['default'], units=o['units'],
                            training_data=o['values'])

        self.add_subsystem('readMap', readmap, promotes_inputs=['*'],
                                promotes_outputs=['*'])

if __name__ == "__main__":

    p = om.Problem()
    des_vars = p.model.add_subsystem(
        'des_vars', om.IndepVarComp(), promotes=['*'])
    # des_vars.add_output('P', 1.2, units='bar')
    # des_vars.add_output('T', 400, units='degK')
    des_vars.add_output('h', 10, units='cal/g')
    p.model.add_subsystem('map', PropertyMap(
        map_data=AIR_MIX_enthalpy, get_temp=True), promotes=['*'])

    p.setup(check=True)
    p.run_model()
    p.check_partials(compact_print=True)

    # print(p['P'])
    print(p['T'])
    # print(p['S'])
    # print(p['Cp'])
    print(p['h'])