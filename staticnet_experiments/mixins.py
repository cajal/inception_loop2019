from functools import partial

import datajoint as dj
import numpy as np

from neuro_data.static_images.configs import DataConfig
from neuro_data.static_images.data_schemas import StaticMultiDataset
from . import logger as log
from .utils import correlation_closure, compute_predictions, compute_scores


class TrainMixin:
    def train(self, key):
        from .configs import TrainConfig
        log.info('Training ' + repr(key))
        # --- load data
        train_key = TrainConfig().train_key(key)
        trainsets, trainloaders = DataConfig().load_data(key, tier='train', cuda=True, **train_key)

        if len(trainsets) == 0:
            log.warning("Empty dataset. Adding this key into BadConfig")
            from .models import BadConfig, BadConfigException
            BadConfig.insert1(dict(key, reason='Empty dataset'), ignore_extra_fields=True)
            raise BadConfigException

        valsets, valloaders = DataConfig().load_data(key, tier='validation', cuda=True, **train_key,
                                                     key_order=trainsets)

        testsets, testloaders = DataConfig().load_data(key, tier='test', cuda=True, **train_key,
                                                       key_order=trainsets)

        for k, ts in trainsets.items():
            log.info('Trainingset {}\n{}'.format(k, repr(ts)))

        model = self.build_network(key, trainsets=trainsets)

        model = TrainConfig().train(key, model=model, trainloaders=trainloaders,
                                    valloaders=valloaders)

        # --- test
        stop_closure = partial(correlation_closure, loaders=valloaders)
        updated_key = dict(key,
                           val_corr=np.nanmean(stop_closure(model, avg=False)),
                           model={k: v.cpu().numpy() for k, v in model.state_dict().items()})

        scores, unit_scores = self.compute_test_score_tuples(key, testloaders, model)
        return updated_key, scores, unit_scores


class TestMixin:
    def compute_test_score_tuples(self, key, testloaders, model):
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