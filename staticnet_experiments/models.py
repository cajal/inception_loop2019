from collections import OrderedDict, namedtuple
from pprint import pformat
from neuro_data.static_images.data_schemas import StaticScan
from neuro_data.static_images.configs import DataConfig
from .utils import gitlog, set_seed
from . import logger as log
import datajoint as dj
from .utils import set_seed
from .configs import NetworkConfig, Seed, TrainConfig
import torch
from neuro_data.static_images.data_schemas import StaticMultiDataset
from .utils import correlation_closure, compute_predictions, compute_scores

# experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
# meso = dj.create_virtual_module('meso', 'pipeline_meso')

schema = dj.schema('neurostatic_models')

torch.backends.cudnn.benchmark = True

dj.config['external-static'] = dict(
    protocol='file',
    location='/external/neuro-static/')

@schema
class BadConfig(dj.Manual):
    definition = """
    # insert keys that won't work for reasons that are too tedious to anticipate
    -> StaticMultiDataset
    -> NetworkConfig
    -> Seed

    ---
    reason      : varchar(1024) # reason for being in this table
    """


class BadConfigException(Exception):
    pass

@schema
@gitlog
class Model(dj.Computed):
    definition = """
    -> StaticMultiDataset
    -> NetworkConfig
    -> Seed
    ---
    val_corr                  : float              # validation correlation (single trial)
    model                     : external-static    # stored model
    """

    @property
    def key_source(self):
        return super().key_source - BadConfig & 'seed > 1000'

    class TestScores(dj.Part):
        definition = """
        -> master
        -> StaticMultiDataset.Member
        ---
        neurons                  : int         # number of neurons
        pearson                  : float       # test correlation on single trial responses
        """

    class UnitTestScores(dj.Part):
        definition = """
        -> master.TestScores
        -> StaticScan.Unit
        ---
        pearson                  : float       # test correlation on single trial responses
        """

    def load_network(self, key=None, trainsets=None):
        if key is None:
            key = self.fetch1(dj.key)
        model = NetworkConfig().build_network(key, trainsets=trainsets)
        state_dict = (self & key).fetch1('model')
        state_dict = {k: torch.from_numpy(state_dict[k][0]) for k in state_dict.dtype.names}
        mod_state_dict = model.state_dict()
        for k in set(mod_state_dict) - set(state_dict):
            log.warning('Could not find paramater {} setting to initialization value'.format(repr(k)))
            state_dict[k] = mod_state_dict[k]
        model.load_state_dict(state_dict)
        return model

    def make(self, key):
        log.info('Populating ' + pformat(key, indent=10))
        # --- set seed
        seed = (Seed() & key).fetch1('seed')
        log.info('Setting seed to {}'.format(seed))
        set_seed(seed)
        try:
            updated_key, scores, unit_scores = NetworkConfig().train(key)
        except BadConfigException:
            return
        self.insert1(updated_key, ignore_extra_fields=True)
        git_key = self.log_git(key)
        log.info('Logging git key ' +  repr(git_key))
        self.TestScores().insert(scores, ignore_extra_fields=True)
        self.UnitTestScores().insert(unit_scores, ignore_extra_fields=True)

    def evaluate(self, key=None):
        if key is None:
            key = self.fetch1('KEY')


        model = self.load_network(key)
        model.eval()
        model.cuda()

        # get network configuration information
        net_key = NetworkConfig().net_key(key)
        train_key = TrainConfig().train_key(net_key)
        testsets, testloaders = DataConfig().load_data(net_key, tier='test', cuda=True, **train_key)

        scores, unit_scores = [], []
        for readout_key, testloader in testloaders.items():
            log.info('Computing test scores for ' + readout_key)

            y, y_hat = compute_predictions(testloader, model, readout_key)
            perf_scores = compute_scores(y, y_hat)

            member_key = (StaticMultiDataset.Member() & key & dict(name=readout_key)).fetch1(dj.key)
            member_key.update(key)

            unit_ids = testloader.dataset.neurons.unit_ids
            member_key['neurons'] = len(unit_ids)
            member_key['pearson'] = perf_scores.pearson.mean()

            scores.append(member_key)
            unit_scores.extend(
                [dict(member_key, unit_id=u, pearson=c) for u, c in zip(unit_ids, perf_scores.pearson)])
        return scores, unit_scores




