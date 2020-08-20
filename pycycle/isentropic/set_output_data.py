import inspect
import numpy as np

from openmdao.api import ExplicitComponent

_full_out_args = inspect.getfullargspec(ExplicitComponent.add_output)
_allowed_out_args = set(_full_out_args.args[3:] + _full_out_args.kwonlyargs)


class UnitCompBase(ExplicitComponent):

    def __init__(self, fl_name):

        super(UnitCompBase, self).__init__()

        self.fl_name = fl_name

    def setup(self):
        rel2meta = self._var_rel2meta

        fl_name = self.fl_name

        for in_name in self._var_rel_names['input']:

            meta = rel2meta[in_name]
            val = meta['value'].copy()
            new_meta = {k:v for k, v in meta.items() if k in _allowed_out_args}

            out_name = '{0}:{1}'.format(fl_name, in_name)
            self.add_output(out_name, val=val, **new_meta)

        rel2meta = self._var_rel2meta

        for in_name, out_name in zip(self._var_rel_names['input'], self._var_rel_names['output']):

            shape = rel2meta[in_name]['shape']
            size = np.prod(shape)
            row_col = np.arange(size, dtype=int)

            self.declare_partials(of=out_name, wrt=in_name,
                                  val=np.ones(size), rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        outputs._data[:] = inputs._data
        # if 'burner' in self.pathname:
        #     print(self.pathname)
        #     print(outputs['Fl_O:tot:T'])


class SetOutputData(UnitCompBase):

    def setup(self):

        self.add_input('T', val=284., units="degR", desc="Temperature")
        self.add_input('P', val=1., units='lbf/inch**2', desc="Pressure")
        self.add_input('h', val=1., units="Btu/lbm", desc="enthalpy")
        self.add_input('S', val=1., units="Btu/(lbm*degR)", desc="entropy")
        self.add_input('gamma', val=1.4, desc="ratio of specific heats")
        self.add_input('Cp', val=1., units="Btu/(lbm*degR)", desc="Specific heat at constant pressure")
        self.add_input('Cv', val=1., units="Btu/(lbm*degR)", desc="Specific heat at constant volume")
        self.add_input('rho', val=1., units="lbm/ft**3", desc="density")
        self.add_input('R', val=1.0, units="Btu/(lbm*degR)", desc='Total specific gas constant')
        self.add_input('b0', val=-1, units=None, desc='Not used in isentropic mode, but necessary for flow connection')
        self.add_input('n', val=-1, units=None, desc='Not used in isentropic mode, but necessary for flow connection')

        super(SetOutputData, self).setup()

class IOMatching(ExplicitComponent):

    def setup(self):
        
        self.add_input('b0', val=-1, units=None, desc='Atomic abundances, but not used in this mode')
        self.add_input('P', val=1.013, units='bar', desc='Static pressure')

        self.add_output('n', units=None, val=-1, desc='Molar concentrations, but not used in this mode')
        self.add_output('n_moles', units=None, val=-1, desc='Numer of moles, but not used in this mode')
        self.add_output('Ps', units='bar', desc='Static pressure')

        self.declare_partials('n', 'b0', val=1)
        self.declare_partials('n_moles', 'b0', val=1)
        self.declare_partials('Ps', 'P', val=1)

    def compute(self, inputs, outputs):
        
        outputs['n'] = inputs['b0']
        outputs['n_moles'] = inputs['b0']
        outputs['Ps'] = inputs['P']


if __name__ == "__main__":

    from openmdao.api import Problem, Group

    p = Problem()
    p.model = Group()

    p.model.add_subsystem('units', SetOutputData(fl_name='flow'), promotes=['*'])

    p.setup()

    p.run_model()

    p.check_partials(compact_print=True)



    prob = Problem()
    prob.model = Group()

    prob.model.add_subsystem('units', IOMatching(), promotes=['*'])

    prob.setup()

    prob.run_model()

    prob.check_partials(compact_print=True)
