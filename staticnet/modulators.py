from collections import OrderedDict

from torch import nn
from torch.nn.init import xavier_normal
import torch
from torch.nn import functional as F
import torch.nn.init as init
from attorch.regularizers import LaplaceL23d, LaplaceL2
from torch.autograd import Variable
from attorch.module import ModuleDict

from staticnet import logger as log


class MLP(nn.Module):
    def __init__(self, neurons, input_channels=3, hidden_channels=10, layers=2, **kwargs):
        super().__init__()
        log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        feat = [nn.Linear(input_channels, hidden_channels), nn.ReLU()]
        for _ in range(layers - 1):
            feat.extend([nn.Linear(hidden_channels, hidden_channels), nn.ReLU()])
        self.mlp = nn.Sequential(*feat)
        self.linear = nn.Linear(hidden_channels, neurons)

    def regularizer(self):
        return self.linear.weight.abs().mean()

    def initialize(self):
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def forward(self, input, readoutput=None):
        mod = torch.exp(self.linear(self.mlp(input)))

        if readoutput is None:
            log.info('Nothing to modulate. Returning modulation only')
            return mod
        else:
            return readoutput * mod


class StaticModulator(ModuleDict):
    _base_modulator = None

    def __init__(self, n_neurons, input_channels=3, hidden_channels=5,
                 layers=2, gamma_modulator=0, **kwargs):
        log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()
        self.gamma_modulator = gamma_modulator
        for k, n in n_neurons.items():
            if isinstance(input_channels, OrderedDict):
                ic = input_channels[k]
            else:
                ic = input_channels
            self.add_module(k, self._base_modulator(n, ic, hidden_channels, layers=layers))

    def initialize(self):
        log.info('Initializing ' + self.__class__.__name__)
        for k, mu in self.items():
            self[k].initialize()

    def regularizer(self, data_key):
        return self[data_key].regularizer() * self.gamma_modulator


class MLPModulator(StaticModulator):
    _base_modulator = MLP


def NoModulator(*args, **kwargs):
    """
    Dummy function to create an object that returns None
    Args:
        *args:   will be ignored
        *kwargs: will be ignored

    Returns:
        None
    """
    return None
