from collections import OrderedDict, namedtuple
from functools import partial
from inspect import isclass
from itertools import product, repeat
import torch
from tqdm import tqdm

from attorch.layers import Elu1
from attorch.losses import PoissonLoss
import datajoint as dj

import numpy as np

from attorch.train import early_stopping
from attorch.utils import cycle_datasets
from neuro_data.static_images.configs import DataConfig
from staticnet import cores, modulators, readouts, shifters
from staticnet.base import CorePlusReadout2d


from .utils import key_hash, to_native, correlation_closure
from .mixins import TrainMixin, TestMixin
from . import logger as log

experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')

schema = dj.schema('neurostatic_configs', locals())

Datasets = namedtuple('Datasets', ['trainsets', 'valsets', 'testsets', 'mu_dict', 'n_neurons'])


class Config:
    _config_type = None

    @property
    def definition(self):
        return """
        # parameters for {cn}

        {ct}_hash                   : varchar(256) # unique identifier for configuration
        {extra_foreign} 
        ---
        {ct}_type                   : varchar(50)  # type
        {ct}_ts=CURRENT_TIMESTAMP : timestamp      # automatic
        """.format(ct=self._config_type, cn=self.__class__.__name__,
                   extra_foreign=self._extra_foreign if hasattr(self, '_extra_foreign') else '')

    def fill(self):
        type_name = self._config_type + '_type'
        hash_name = self._config_type + '_hash'
        for rel in [getattr(self, member) for member in dir(self)
                    if isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part)]:
            log.info('Checking ' + rel.__name__)
            for key in rel().content:
                key[type_name] = rel.__name__
                key[hash_name] = key_hash(key)

                if not key in (rel()).proj():
                    self.insert1(key, ignore_extra_fields=True)
                    log.info('Inserting ' + repr(key))
                    rel().insert1(key, ignore_extra_fields=True)

    def parameters(self, key, selection=None, exclude_timestamp=True):
        type_name = self._config_type + '_type'
        ts_name = self._config_type + '_ts'
        key = (self & key).fetch1()  # complete parameters
        part = getattr(self, key[type_name])
        ret = (self * part() & key).fetch1()
        ret = to_native(ret)
        if exclude_timestamp:
            del ret[ts_name]
        if selection is None:
            return ret
        else:
            if isinstance(selection, list):
                return tuple(ret[k] for k in selection)
            else:
                return ret[selection]

    def select_hashes(self):
        configs = [getattr(self, member) for member in dir(self) if
                   isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part)]
        print('\n'.join(['({}) {}'.format(i, rel.__name__) for i, rel in enumerate(configs)]))

        choices = input('Please select configuration [comma separated list]: ')
        choices = list(map(int, choices.split(',')))

        hashes = []
        for choice in choices:
            restriction = dict()
            rel = configs[int(choice)]()
            while restriction != '':
                old_restriction = restriction
                print(old_restriction)
                print(rel & old_restriction)
                restriction = input('Please enter a restriction [ENTER for exit]: ')
            hashes.extend((rel & old_restriction).fetch('{}_hash'.format(self._config_type)))
        return '{}_hash'.format(self._config_type), hashes




@schema
class Seed(dj.Lookup):
    definition = """
    # random seed for training

    seed                 :  int # random seed
    ---
    """

    @property
    def contents(self):
        yield from zip([1009, 1215, 2606, 99999, 101, 102, 103, 104, 105, 106, 107, 108,
                        109, 110, 111, 112])


@schema
class CoreConfig(Config, dj.Lookup):
    _config_type = 'core'

    class Stacked2d(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels           : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        hidden_kern           : int      # kernel size at hidden convolutional layers
        layers                : int      # layers
        gamma_hidden          : double   # regularization constant for hidden layers in CNN
        gamma_input           : double   # regularization constant for input  convolutional layers
        skip                  : int      # introduce skip connections if skip > 1
        final_nonlinearity    : bool     # end last layer of core with nonlinearity
        bias                  : bool     # use bias or not
        pad_input             : bool     # pad input layers
        """

        @property
        def content(self):
            for p in product([32], [7], [3], [3], [.1], [50], [3], [True], [False], [False]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d
            for p in product([32], [18], [1], [3], [0], [50], [3], [True], [False], [False]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d
            for p in product([32], [7], [7], [3], [0.1], [50], [0], [True], [False], [False]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d
            for p in product([32], [13], [7], [3], [0.1], [50], [0], [True], [False], [False]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class ModStacked2d(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels           : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        hidden_kern           : int      # kernel size at hidden convolutional layers
        layers                : int      # layers
        gamma_hidden          : double   # regularization constant for hidden layers in CNN
        gamma_input           : double   # regularization constant for input  convolutional layers
        skip                  : int      # introduce skip connections if skip > 1
        final_nonlinearity    : bool     # end last layer of core with nonlinearity
        bias                  : bool     # use bias or not
        pad_input             : bool     # pad input layers
        laplace_padding       : int      # laplace padding
        """

        @property
        def content(self):
            for p in product([32], [11, 15], [3, 7], [3], [0.1, 1.0], [50, 100], [3], [True], [False], [False], [1]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class Linear(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        gamma_input           : double   # regularization constant for input  convolutional layers
        """

        @property
        def content(self):
            for p in product([32], [13], [50]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class ModLinear(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        gamma_input           : double   # regularization constant for input  convolutional layers
        laplace_padding       : int      # laplace padding
        """

        @property
        def content(self):
            for p in product([32], [13, 19, 23], [50], [0, 1]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d


    class SigmoidLaplace(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        hidden_kern           : int      # kernel size at hidden convolutional layers
        layers                : int      # layers
        gamma_hidden          : double   # regularization constant for hidden layers in CNN
        gamma_input           : double   # regularization constant for input  convolutional layers
        skip                  : int      # introduce skip connections if skip > 1
        final_nonlinearity    : bool     # end last layer of core with nonlinearity
        bias                  : bool     # use bias or not
        pad_input             : bool     # pad input layers
        laplace_padding       : int      # laplace padding
        sigmoid_scale         : float    # scaling for distance in sigmoid weighting
        sigmoid_center        : float    # center of sigmoid distance
        """

        @property
        def content(self):
            for p in product([32], [11, 15], [3, 7], [3], [0.1, 1.0], [50, 100], [3],
                             [True], [False], [False], [1], [1, 2], [0.5]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class LinearSigmoidLaplace(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        gamma_input           : double   # regularization constant for input  convolutional layers
        laplace_padding       : int      # laplace padding
        sigmoid_scale         : float    # scaling for distance in sigmoid weighting
        sigmoid_center        : float    # center of sigmoid distance
        """

        @property
        def content(self):
            for p in product([32], [13, 19, 23], [50, 100], [1], [1, 2], [0.5]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class GaussianLaplace(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        hidden_kern           : int      # kernel size at hidden convolutional layers
        layers                : int      # layers
        gamma_hidden          : double   # regularization constant for hidden layers in CNN
        gamma_input           : double   # regularization constant for input  convolutional layers
        skip                  : int      # introduce skip connections if skip > 1
        final_nonlinearity    : bool     # end last layer of core with nonlinearity
        bias                  : bool     # use bias or not
        pad_input             : bool     # pad input layers
        laplace_padding       : int      # laplace padding
        gauss_sigma           : float    # sigma of Gaussian regularization weight
        gauss_bias            : float    # base line shift
        """

        @property
        def content(self):
            for p in product([32], [11, 15], [3, 7], [3], [0.1, 1.0], [50, 100], [3],
                             [True], [False], [False], [1], [0.5, 1], [0.0, 0.5]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class StackedLinearGaussianLaplace(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        hidden_kern           : int      # kernel size at hidden convolutional layers
        layers                : int      # layers
        gamma_hidden          : double   # regularization constant for hidden layers in CNN
        gamma_input           : double   # regularization constant for input  convolutional layers
        skip                  : int      # introduce skip connections if skip > 1
        final_nonlinearity    : bool     # end last layer of core with nonlinearity
        bias                  : bool     # use bias or not
        pad_input             : bool     # pad input layers
        laplace_padding       : int      # laplace padding
        gauss_sigma           : float    # sigma of Gaussian regularization weight
        gauss_bias            : float    # base line shift
        """

        @property
        def content(self):
            for p in product([32], [15], [7], [3], [0.1, 1.0], [50, 100], [3],
                             [True], [False], [False], [1], [0.5, 1], [0.0, 0.5]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class LinearGaussianLaplace(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        gamma_input           : double   # regularization constant for input  convolutional layers
        laplace_padding       : int      # laplace padding
        gauss_sigma           : float    # sigma of Gaussian regularization weight
        gauss_bias            : float    # base line shift
        """

        @property
        def content(self):
            for p in product([32], [13, 19, 23], [50, 100], [1], [0.5, 1], [0.0, 0.5]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    def build(self, input_channels, key, **kwargs):
        core_key = self.parameters(key)

        core_type = core_key.pop('core_type')

        # let parameters be specially parsed if `build_key` is defined
        CoreConfig = getattr(self, core_type)
        core_config = CoreConfig()
        if hasattr(core_config, 'build_key'):
            core_key = core_config.build_key(core_key)

        # get the core module
        core_name = '{}Core'.format(core_type)
        assert hasattr(cores, core_name), '''Cannot find core for {core_name}.
                                             Core needs to be names "{core_name}Core"
                                             in architectures.cores'''.format(core_name=core_name)
        Core = getattr(cores, core_name)

        return Core(input_channels=input_channels, **core_key, **kwargs)


@schema
class ReadoutConfig(Config, dj.Lookup):
    _config_type = 'ro'

    def build(self, in_shape, neurons, key):
        ro_key = self.parameters(key)
        ro_type = ro_key.pop('ro_type')
        ro_name = '{}Readout'.format(ro_type)
        ro_table = getattr(self, ro_type)()

        # let ro specific part table modify the key
        if hasattr(ro_table, 'build_key'):
            ro_key = ro_table.build_key(ro_key)

        assert hasattr(readouts, ro_name), '''Cannot find readout for {ro_name}.
                                             Core needs to be names "{ro_name}"
                                             in architectures.readout'''.format(ro_name=ro_name)
        Readout = getattr(readouts, ro_name)
        return Readout(in_shape, neurons, **ro_key)

    class SpatialXFeatures(dj.Part):
        definition = """
        -> master
        ---
        gamma_readout          : float # regularization constant for features
        positive               : bool  # whether the features will be restricted to be positive
        normalize              : bool  # whether the spatial features will be normalized
        """

        @property
        def content(self):
            for p in product([1], [False], [True]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class SpatialTransformerPooled2d(dj.Part):
        definition = """
        -> master
        ---
        gamma_features         : float # regularization constant for features
        positive               : bool  # whether the features will be restricted to be positive
        pool_steps             : tinyint  # number of pooling steps in the readout
        """

        @property
        def content(self):
            for p in product([20], [False], [4]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d
                 
    class ModifiedSpatialTransformerPyramid2d(dj.Part):
        definition = """
        -> master
        ---
        type                   : varchar(20) # regularization constant for features
        gamma_features         : float   # regularization constant for features
        positive               : bool    # whether the features will be restricted to be positive
        scale_n                : tinyint # number of pooling steps in the readout
        downsample             : bool    # whether to downsample lowpass representations
        """

        @property
        def content(self):
            for p in product(['gauss5x5'], [1], [False], [5], [True, False]):
                d = dict(zip(self.heading.dependent_attributes, p))
                d['_skip_upsampling'] = True
                yield d

    class SpatialTransformerPyramid2d(dj.Part):
        definition = """
        -> master
        ---
        type                   : varchar(20) # regularization constant for features
        gamma_features         : float   # regularization constant for features
        positive               : bool    # whether the features will be restricted to be positive
        scale_n                : tinyint # number of pooling steps in the readout
        downsample             : bool    # whether to downsample lowpass representations
        """

        @property
        def content(self):
            for p in product(['gauss5x5'], [1], [False], [5], [True, False]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d


    class SpatialTransformer2d(dj.Part):
        definition = """
        -> master
        ---
        gamma_features         : float # regularization constant for features
        positive               : bool  # whether the features will be restricted to be positive
        """

        @property
        def content(self):
            for p in product([20, 1], [False]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d


@schema
class ModulatorConfig(Config, dj.Lookup):
    _config_type = 'mod'

    def build(self, data_keys, input_features, key):
        mod_key = self.parameters(key)
        mod_name = '{}Modulator'.format(mod_key.pop('mod_type'))
        assert hasattr(modulators, mod_name), '''Cannot find modulator for {mod_name}.
                                             Core needs to be names "{mod_name}"
                                             in architectures.readout'''.format(mod_name=mod_name)
        Modulator = getattr(modulators, mod_name)
        return Modulator(data_keys, input_features, **mod_key)

    class No(dj.Part):
        definition = """
        -> master
        ---
        """

        @property
        def content(self):
            yield dict()

    class MLP(dj.Part):
        definition = """
        -> master
        ---
        layers                    : tinyint  # layers of MLP
        hidden_channels           : int      # hidden channels
        gamma_modulator           : double   # regularization constant for input  convolutional layers
        """

        @property
        def content(self):
            for p in product([2], [10], [0.0]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d


@schema
class ShifterConfig(Config, dj.Lookup):
    _config_type = 'shift'

    def build(self, data_keys, input_features, key):
        shift_key = self.parameters(key)
        shift_name = '{}Shifter'.format(shift_key.pop('shift_type'))
        assert hasattr(shifters, shift_name), '''Cannot find modulator for {shift_name}.
                                             Core needs to be names "{shift_name}"
                                             in architectures.readout'''.format(shift_name=shift_name)
        Shifter = getattr(shifters, shift_name)
        return Shifter(data_keys, input_features, **shift_key)

    class MLP(dj.Part):
        definition = """
          -> master
          ---
          shift_layers            : tinyint  # layers of MLP
          hidden_channels_shifter : int      # hidden channels
          gamma_shifter           : double   # regularization constant for input  convolutional layers
          """

        @property
        def content(self):
            for p in product([3], [5], [0.0]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class StaticAffine2d(dj.Part):
        definition = """
        -> master
        ---
        gamma_shifter           : double   # regularization constant affine weights
        bias                    : bool     # whether to include bias term
        """

        @property
        def content(self):
            for p in product([1e-3], [True]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class No(dj.Part):
        definition = """
        -> master
        ---
        """

        @property
        def content(self):
            yield dict()


@schema
class TrainConfig(Config, dj.Lookup):
    _config_type = 'train'

    class Default(dj.Part):
        definition = """
        -> master
        ---
        batch_size             : smallint # 
        schedule               : longblob # learning rate schedule
        max_epoch              : int      # maximum number of epochs
        """

        @property
        def content(self):
            yield dict(batch_size=250, schedule=np.array([0.005, 0.001]), max_epoch=500)
            yield dict(batch_size=125, schedule=np.array([0.005, 0.001]), max_epoch=500)

        def train(self, key, model, trainloaders, valloaders):
            log.info('Training'.ljust(40, '-'))
            # set some parameters
            criterion = PoissonLoss()

            def objective(model, readout_key, inputs, beh, eye_pos, targets):
                outputs = model(inputs, readout_key, eye_pos=eye_pos, behavior=beh)
                return criterion(outputs, targets) \
                       + model.core.regularizer() \
                       + model.readout.regularizer(readout_key) \
                       + (model.shifter.regularizer(readout_key) if model.shift else 0) \
                       + (model.modulator.regularizer(readout_key) if model.modulate else 0)

            def run(model, objective, optimizer, stop_closure, trainloaders, epoch=0,
                    interval=1, patience=10, max_iter=10, maximize=True, tolerance=1e-6, cuda=True,
                    restore_best=True, accumulate_gradient=1
                    ):
                log.info('Training models with {} and state {}'.format(optimizer.__class__.__name__,
                                                                       repr(model.state)))
                optimizer.zero_grad()
                iteration = 0

                for epoch, val_obj in early_stopping(model, stop_closure,
                                                     interval=interval, patience=patience,
                                                     start=epoch, max_iter=max_iter, maximize=maximize,
                                                     tolerance=tolerance, restore_best=restore_best):
                    for batch_no, (readout_key, data) in tqdm(enumerate(cycle_datasets(trainloaders)),
                                                              desc=self.__class__.__name__.ljust(
                                                                  25) + '  | Epoch {}'.format(epoch)):
                        obj = objective(model, readout_key, *data)
                        obj.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        iteration += 1
                return model, epoch

            # --- initialize
            mu_dict = {k: dl.dataset.transformed_mean().responses for k, dl in trainloaders.items()}
            model.readout.initialize(mu_dict)
            model.core.initialize()
            if model.shifter is not None:
                biases = {k: -dl.dataset.transformed_mean().pupil_center for k, dl in trainloaders.items()}
                model.shifter.initialize(bias=biases)
            if model.modulator is not None:
                model.modulator.initialize()

            # --- train
            log.info('Shipping model to GPU')
            model = model.cuda()
            model.train(True)
            print(model)
            epoch = 0

            schedule = key['schedule']
            model.shift = True
            for opt, lr in zip(repeat(torch.optim.Adam), schedule):
                log.info('Training with learning rate {}'.format(lr))

                optimizer = opt(model.parameters(), lr=lr)

                model, epoch = run(model, objective, optimizer,
                                   partial(correlation_closure, loaders=valloaders), trainloaders,
                                   epoch=epoch, max_iter=key['max_epoch'], patience=10)
            model.eval()
            return model

    def train_key(self, key):
        return dict(key, **self.parameters(key))

    def train(self, key, **kwargs):
        train_key = self.train_key(key)
        Trainer = getattr(self, train_key.pop('train_type'))
        return Trainer().train(train_key, **kwargs)


@schema
class NetworkConfig(Config, dj.Lookup):
    _config_type = 'net'

    class CorePlusReadout(dj.Part, TestMixin, TrainMixin):
        definition = """
        -> master
        ---
        -> CoreConfig
        -> ReadoutConfig
        -> ShifterConfig
        -> ModulatorConfig
        -> TrainConfig
        -> DataConfig
        """

        @property
        def content(self):
            yield from (CoreConfig
                        * (ReadoutConfig - ReadoutConfig.SpatialTransformer2d - ReadoutConfig.SpatialXFeatures)
                        * ShifterConfig.StaticAffine2d
                        * ModulatorConfig.MLP
                        * (TrainConfig.Default & dict(batch_size=125))
                        * DataConfig.AreaLayer
                        & dict(normalize=1,
                               stimulus_type='stimulus.Frame',
                               exclude='images,responses')).fetch("KEY")

        @property
        def content_source(self):
            return (CoreConfig * ReadoutConfig * ShifterConfig * ModulatorConfig * TrainConfig * DataConfig).proj()


        def build_network(self, key, trainsets=None):
            """
            Builds a specified model
            Args:
                key:    key for CNNParameters used to load the parameter of the model. If None, (self & key) must
                        be non-empty so that key can be inferred.
                img_shape: image shape to figure out the size of the readouts
                n_neurons: dictionary with readout sizes (number of neurons)

            Returns:
                an uninitialized MultiCNN
            """
            
            key = dict((self & (key or dict())).fetch1(), **key)
            
            # --- load datasets
            if trainsets is None:
                train_key = TrainConfig().train_key(key)
                trainsets, _ = DataConfig().load_data(key, tier='train', **train_key)
            img_shape = list(trainsets.values())[0].img_shape
            n_neurons = OrderedDict([(k, v.n_neurons) for k, v in trainsets.items()])

            core = CoreConfig().build(img_shape[1], key)
            ro_in_shape = CorePlusReadout2d.get_readout_in_shape(core, img_shape)
            readout = ReadoutConfig().build(ro_in_shape, n_neurons, key)
            shifter = ShifterConfig().build(n_neurons, input_features=2, key=key)
            modulator = ModulatorConfig().build(n_neurons, input_features=3, key=key)

            # --- initialize
            return CorePlusReadout2d(core, readout, nonlinearity=Elu1(), shifter=shifter, modulator=modulator)


        @property
        def source_restriction(self):
            restr = []
            for source in  [CoreConfig(), ReadoutConfig(), ShifterConfig(), ModulatorConfig(), TrainConfig(), DataConfig()]:
                hname = '{}_hash'.format(source._config_type)
                print('---', 'Selecting {} '.format(source.__class__.__name__).ljust(76, '-'))
                hash, hashes = source.select_hashes()
                restr.append('{} in ("{}")'.format(hname, '", "'.join(hashes)))
            return dj.AndList(restr)

        def select_hashes(self):
            hashes = []
            restriction = dict()
            rel = self
            while restriction != '':
                old_restriction = restriction
                print(old_restriction)
                print(rel & old_restriction)
                restriction = input('Please enter a restriction [ENTER for exit]: ')
            hashes.extend((rel & old_restriction).fetch('{}_hash'.format(self._config_type)))
            return 'net_hash', hashes


    class CorePlusReadoutNoDownsample(dj.Part, TestMixin, TrainMixin):
        definition = """
        -> master
        ---
        (parent_hash) -> master.CorePlusReadout
        (new_ro_hash)-> ReadoutConfig
        """

        @property
        def content(self):
            yield from self.content_source.fetch("KEY")

        @property
        def content_source(self):
            parents = (NetworkConfig.CorePlusReadout
                       & ReadoutConfig.SpatialTransformerPyramid2d).proj(parent_hash='net_hash')
            return (parents * ReadoutConfig.SpatialTransformer2d.proj(new_ro_hash='ro_hash')).proj()

        def parent_key(self, key):
            pkey = dict(key)
            pkey['net_hash'] = pkey.pop('parent_hash')
            return NetworkConfig().net_key(pkey)

        def complete_with_parent(self, key):
            pkey = self.parent_key(key)
            del pkey['net_hash']
            return dict(key, **pkey)

        def train(self, key):
            return super().train(self.complete_with_parent(key))

        def build_network(self, key, trainsets=None):
            """
            Builds a specified model
            Args:
                key:    key for CNNParameters used to load the parameter of the model. If None, (self & key) must
                        be non-empty so that key can be inferred.
                img_shape: image shape to figure out the size of the readouts
                n_neurons: dictionary with readout sizes (number of neurons)

            Returns:
                an uninitialized MultiCNN
            """
            key = self.complete_with_parent(key)
            from .models import Model

            model = Model().load_network(self.parent_key(key), trainsets=trainsets)

            img_shape = list(trainsets.values())[0].img_shape
            n_neurons = OrderedDict([(k, v.n_neurons) for k, v in trainsets.items()])

            core = CoreConfig().build(img_shape[1], key)
            ro_in_shape = CorePlusReadout2d.get_readout_in_shape(core, img_shape)
            readout = ReadoutConfig().build(ro_in_shape, n_neurons, dict(ro_hash=key['new_ro_hash']))
            for rok in model.readout:
                readout[rok].grid = model.readout[rok].grid

            shifter = ShifterConfig().build(n_neurons, input_features=2, key=key)
            modulator = ModulatorConfig().build(n_neurons, input_features=3, key=key)
            # --- initialize
            return CorePlusReadout2d(core, readout, nonlinearity=Elu1(), shifter=shifter, modulator=modulator)

        @property
        def source_restriction(self):
            restr = []

            print('===', 'Selecting NetworkConfig '.ljust(76, '='))
            restr.append('parent_hash in ("{}")'.format('", "'.join((self.master.CorePlusReadout()
                                                & self.master.CorePlusReadout().source_restriction).fetch('net_hash'))))
            print('===', 'Selecting ReadoutConfig '.ljust(76, '='))
            restr.append('new_ro_hash in ("{}")'.format('", "'.join(ReadoutConfig().select_hashes()[1])))

            return dj.AndList(restr)

    def net_key(self, key):
        return dict(key, **self.parameters(key))

    def train(self, key, **kwargs):
        net_key = self.net_key(key)
        Network = getattr(self, net_key.pop('net_type'))
        return Network().train(net_key, **kwargs)

    def build_network(self, key, **kwargs):
        net_key = self.net_key(key)
        Network = getattr(self, net_key.pop('net_type'))
        return Network().build_network(net_key, **kwargs)

    def fill(self):
        type_name = self._config_type + '_type'
        hash_name = self._config_type + '_hash'
        configs = [getattr(self, member) for member in dir(self) if
                   isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part)]
        print('\n'.join(['({}) {}'.format(i, rel.__name__) for i, rel in enumerate(configs)]))
        choice = None
        choices = list(map(str, range(len(configs))))
        while choice not in choices:
            choice = input('Please select configuration: ')
        network_config = configs[int(choice)]()


        for key in network_config.content_source.proj() & network_config.source_restriction:
            key[type_name] = network_config.__class__.__name__
            key[hash_name] = key_hash(key)

            if not key in (network_config).proj():
                self.insert1(key, ignore_extra_fields=True)
                log.info('Inserting ' + repr(key))
                network_config.insert1(key, ignore_extra_fields=True)
            else:
                log.info('{} already defined'.format(key['net_hash']))
