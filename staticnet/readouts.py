from attorch.layers import (SpatialXFeatureLinear, elu1,
                            SpatialTransformerPooled2d, SpatialTransformerPyramid2d)
from attorch.module import ModuleDict

from staticnet import logger as log


class Readout:
    def initialize(self, *args, **kwargs):
        raise NotImplementedError('initialize is not implemented for ', self.__class__.__name__)

    def __repr__(self):
        s = super().__repr__()
        s += ' [{} regularizers: '.format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: not x.startswith('_') and
                                     ('gamma' in x or 'pool' in x or 'positive' in x), dir(self)):
            ret.append('{} = {}'.format(attr, getattr(self, attr)))
        return s + '|'.join(ret) + ']\n'


class SpatialXFeaturesReadout(Readout, ModuleDict):
    def __init__(self, in_shape, neurons, gamma_readout, positive=True, normalize=True, **kwargs):
        log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self.positive = positive
        self.normalize = normalize
        self.gamma_readout = gamma_readout

        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            self.add_module(k, SpatialXFeatureLinear(in_shape, neur, normalize=normalize, positive=positive))

    def initialize(self, mu_dict):
        log.info('Initializing ' +  self.__class__.__name__)
        for k, mu in mu_dict.items():
            self[k].initialize(init_noise=1e-6)
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key):
        return self[readout_key].l1() * self.gamma_readout


class SpatialTransformerPooled2dReadout(Readout, ModuleDict):
    _BaseReadout = None

    def __init__(self, in_shape, neurons, positive=False, gamma_features=0, pool_steps=0, **kwargs):
        log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self._positive = positive
        self.gamma_features = gamma_features
        self._pool_steps = pool_steps
        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            self.add_module(k, SpatialTransformerPooled2d(in_shape, neur, positive=positive, pool_steps=pool_steps))

    @property
    def positive(self):
        return self._positive

    @positive.setter
    def positive(self, value):
        self._positive = value
        for k in self:
            self[k].positive = value

    def initialize(self, mu_dict):
        log.info('Initializing with mu_dict: ' + ', '.join(['{}: {}'.format(k, len(m)) for k, m in mu_dict.items()]))

        for k, mu in mu_dict.items():
            self[k].initialize()
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key):
        return self[readout_key].feature_l1() * self.gamma_features

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        self._pool_steps = value
        for k in self:
            self[k].poolsteps = value


class SpatialTransformer2dReadout(SpatialTransformerPooled2dReadout):
    def __init__(self, in_shape, neurons, positive=False, gamma_features=0, **kwargs):
        log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__(in_shape, neurons, positive=positive,
                         gamma_features=gamma_features,
                         _pool_steps=0, **kwargs)


class PooledReadout(Readout):
    @property
    def positive(self):
        return self._positive

    @positive.setter
    def positive(self, value):
        self._positive = value
        for k in self:
            self[k].positive = value

    def initialize(self, mu_dict):
        log.info('Initializing with mu_dict: ' + ', '.join(['{}: {}'.format(k, len(m)) for k, m in mu_dict.items()]))
        for k, mu in mu_dict.items():
            self[k].initialize()
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key, subs_idx=None):
        return self[readout_key].feature_l1(subs_idx=subs_idx) * self.gamma_features

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        self._pool_steps = value
        for k in self:
            self[k].poolsteps = value


class _SpatialTransformerPyramid(Readout, ModuleDict):
    _BaseReadout = None

    def __init__(self, in_shape, neurons, positive=False, gamma_features=0, scale_n=3, downsample=True,
                 type=None, _skip_upsampling=False, **kwargs):
        log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self._positive = positive
        self.gamma_features = gamma_features
        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            self.add_module(k, self._BaseReadout(in_shape, neur, positive=positive, scale_n=scale_n,
                                                 downsample=downsample, _skip_upsampling=_skip_upsampling, type=type))

    @property
    def positive(self):
        return self._positive

    @positive.setter
    def positive(self, value):
        self._positive = value
        for k in self:
            self[k].positive = value

    def initialize(self, mu_dict):
        log.info('Initializing with mu_dict: ' + ', '.join(['{}: {}'.format(k, len(m)) for k, m in mu_dict.items()]))

        for k, mu in mu_dict.items():
            self[k].initialize()
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key):
        return self[readout_key].feature_l1() * self.gamma_features


class SpatialTransformerPyramid2dReadout(_SpatialTransformerPyramid):
    _BaseReadout = SpatialTransformerPyramid2d

class ModifiedSpatialTransformerPyramid2dReadout(_SpatialTransformerPyramid):
    _BaseReadout = SpatialTransformerPyramid2d
