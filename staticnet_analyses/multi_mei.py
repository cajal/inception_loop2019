
import datajoint as dj
import torch
import numpy as np
from numpy.linalg import eigvals

from .utils import list_hash, key_hash, deepdraw, process, unprocess, SpatialTransformerPyramid2d, roll, create_gabor

from attorch.regularizers import Laplace
from scipy import ndimage
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from neuro_data.static_images.data_schemas import StaticMultiDataset, InputResponse, StaticScan, meso, stimulus, ConditionTier, Frame, experiment, process_frame, Preprocessing
from neuro_data.static_images.configs import DataConfig
from neuro_data.static_images.stats import Oracle

from staticnet_experiments.configs import CoreConfig, ReadoutConfig, ShifterConfig, ModulatorConfig, NetworkConfig
from staticnet_experiments.models import Model

from staticnet_experiments.utils import compute_scores

from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import convex_hull_image
from scipy.ndimage import find_objects, label

from numpy.linalg import inv, cholesky
import warnings
from skimage.morphology import erosion, square
imagenet = dj.create_virtual_module('imagenet', 'pipeline_imagenet')


try:
    import cv2
except:
    cv2 = None



def compute_predictions(loader, model, readout_key, eye_pos=None, behavior=None):
    y, y_hat = [], []
    for x_val, beh_val, eye_val, y_val in loader:
        if eye_pos is not None:
            eye_val = eye_pos
        if behavior is not None:
            beh_val = behavior
        y_mod = model(x_val, readout_key, eye_pos=eye_val, behavior=beh_val).data.cpu().numpy()
        y.append(y_val.cpu().numpy())
        y_hat.append(y_mod)
    return np.vstack(y), np.vstack(y_hat)


from itertools import count, product, chain
from tqdm import tqdm
from scipy import ndimage

from torch.utils.data import TensorDataset, DataLoader, Dataset




#meso = dj.create_virtual_module('meso', 'pipeline_meso')

schema = dj.schema('neurostatic_multi_mei')

dj.config['external-data'] = dict(
    protocol='file',
    location='/external/')

def best_model(m, aggr=None):
    """
    Given model `m`, returns the best scoring network for each combination of StaticMultiDataset and DataConfig
    :param m: potentially restricted models.Model instance
    :return: best network model for each combination of StaticMultiDataset and DataConfig
    """

    target = m * NetworkConfig.CorePlusReadout()
    aggr_target = StaticMultiDataset() * DataConfig()
    if aggr is not None:
        aggr_target = aggr_target * aggr
    return target * aggr_target.aggr(target, best_corr='max(val_corr)') & 'best_corr = val_corr'


def best_model_new(m, aggr=None):
    """ Given models m, find the best network configuration first (one with best average
    correlation across all seeds) and then find the best seed for that config. This is done
    for each combination of StaticMultiDataset and DataConfig.

    :param m: potentially restricted models.Model instance
    :param aggr: Whether to aggregate by any other factors in addition to
        StaticMultiDataset and DataConfig.

    :return: best network model for each combination of StaticMultiDataset and DataConfig
    """
    corrs_per_model = dj.U('group_id', 'net_hash').aggr(m, best_corr='MAX(val_corr)',
                                                        avg_corr='AVG(val_corr)') # per model but across seeds

    target = m * corrs_per_model * NetworkConfig.CorePlusReadout()
    aggr_target = StaticMultiDataset() * DataConfig()
    if aggr is not None:
        aggr_target = aggr_target * aggr

    best_model = aggr_target.aggr(target, best_avg='max(avg_corr)') # best model for each StaticMultiDataset * DataConfig combo

    return target * best_model & 'best_avg = avg_corr AND best_corr = val_corr'


def prepare_data(key, readout_key):
    """
    Given a key to uniquely identify a dataset and a readout key corresponding to a single component within the
    scan, returns information pertinent to generating MEIs

    Args:
        key: a key that can uniquely identify a single entry from StaticMultiDataset * DataConfig
        readout_key: a specific readout key

    Returns:
        trainset, img_shape, mu, mu_beh, mu_eye, s - where mu and s are mean and stdev of input images.
    """
    trainsets, _ = DataConfig().load_data(key)
    trainset = trainsets[readout_key]
    img_shape = trainset.img_shape
    mu = trainset.statistics['images'][trainset.stats_source]['mean'].value.item()
    s = trainset.statistics['images'][trainset.stats_source]['std'].value.item()
    mus = trainset.transformed_mean()
    mu_eye = mus.pupil_center[None, :].to('cuda')
    mu_beh = mus.behavior[None, :].to('cuda')
    return trainset, img_shape, mu, mu_beh, mu_eye, s





@schema
class ProcessedMonitorCalibration(dj.Computed):
    definition = """
    # process monitor calibrations
    -> experiment.MonitorCalibration
    """
    @property
    def key_source(self):
        # restrict to meso scans
        return experiment.MonitorCalibration & (experiment.Session & {'rig': '2P4'})

    @staticmethod
    def make_loss(x, y):
        def loss(params):
            amp, gamma, offset = params
            y_hat = amp * (x ** gamma) + offset
            return np.mean((y_hat - y) ** 2)

        return loss

    class GammaFit(dj.Part):
        definition = """
        # result of fitting fit lum = amplitude*pixel^gamma + offset
        -> master
        ---
        amplitude: float
        gamma:  float
        offset: float
        mse: float    # quality of fit
        """

        def get_fs(self, key=None):
            if key is None:
                key = {}
            amplitude, gamma, offset = (self & key).fetch1('amplitude', 'gamma', 'offset')

            def f(x):
                return amplitude * (x ** gamma) + offset

            def f_inv(y):
                return ((y - offset) / amplitude) ** (1 / gamma)

            return f, f_inv

        def get_interp(self, key=None):
            if key is None:
                key = {}
            px, lum = (experiment.MonitorCalibration & (self & key).proj()).fetch1('pixel_value', 'luminance')

            f = interp1d(px, lum, kind='cubic')
            f_inv = interp1d(lum, px, kind='cubic')
            return f, f_inv

    def make(self, key):
        self.insert1(key)

        pixel_value, luminance = (experiment.MonitorCalibration & key).fetch1('pixel_value', 'luminance')

        # hard coded threshold...I know...
        # well, there is no luminance measured, it appears
        if np.diff(luminance).mean() < 0.05:
            return

        loss = self.make_loss(pixel_value, luminance)
        best_fit, best_loss = None, np.Inf

        for i in range(200):
            fit = minimize(loss, np.random.randn(3), bounds=[(0, 1000), (0, 20), (0, 100)], )
            if fit.fun < best_loss:
                best_fit = fit
                best_loss = fit.fun

        amplitude, gamma, offset = best_fit.x
        key['amplitude'] = amplitude
        key['gamma'] = gamma
        key['offset'] = offset
        key['mse'] = best_fit.fun

        self.GammaFit().insert1(key)


@schema
class ClosestCalibration(dj.Computed):
    """
    For each scan, find the nearest timing calibration
    """
    definition = """
    -> experiment.Scan
    ---
    -> ProcessedMonitorCalibration.GammaFit.proj(calib_animal_id='animal_id', calib_session='session', calib_scan_idx='scan_idx')
    delta_time: float    # difference in time in hours
    """

    key_source = experiment.Scan & StaticMultiDataset.Member


    def make(self, key):
        target_scan = (experiment.Rig * experiment.Session * experiment.Scan) & key

        calib = experiment.Rig * experiment.Session * experiment.Scan() * ProcessedMonitorCalibration.GammaFit()
        calib_map = dict(calib_animal_id='animal_id', calib_session='session', calib_scan_idx='scan_idx')
        rel = target_scan * calib.proj(**calib_map, calib_ts='scan_ts')
        delta = rel.proj(delta_time='ABS(TIMESTAMPDIFF(SECOND, scan_ts, calib_ts))/3600')
        min_diff = target_scan.aggr(delta, min_diff='min(delta_time)')
        closest_calibration = (delta * min_diff & 'delta_time = min_diff').fetch()[0]

        self.insert1(closest_calibration, ignore_extra_fields=True)

    def get_calib(self, key=None):
        if key is None:
            key = {}
        key = (self & key).fetch1()
        key['animal_id'] = key['calib_animal_id']
        key['session'] = key['calib_session']
        key['scan_idx'] = key['calib_scan_idx']
        return key

    def get_interp(self, key=None):
        key = self.get_calib(key)
        return ProcessedMonitorCalibration.GammaFit().get_interp(key)

    def get_fs(self, key=None):
        key = self.get_calib(key)
        return ProcessedMonitorCalibration.GammaFit().get_fs(key)



@schema
class TargetModel(dj.Computed):
    """
    All models for which we want to generate MEIs
    """
    definition = """
    -> Model
    """

    def make(self, key):
        self.insert1(key)

    def load_model(self, key):
        return Model().load_model(key)

    def fill(self, group_id, dryrun=False):
        restr = {'group_id': group_id}
        core_types = [CoreConfig.Stacked2d, CoreConfig.Linear]
        # check if downsample should be made 0
        ro_types = [ReadoutConfig.ModifiedSpatialTransformerPyramid2d & 'downsample=1',
                    ReadoutConfig.ModifiedSpatialTransformerPyramid2d & 'downsample=0',
                    ReadoutConfig.SpatialTransformerPooled2d,
                    ReadoutConfig.SpatialTransformerPyramid2d & 'downsample=1',
                    ReadoutConfig.SpatialTransformerPyramid2d & 'downsample=0']
        shifter_types = [ShifterConfig.MLP, ShifterConfig.StaticAffine2d]
        mod_types = [ModulatorConfig.MLP]
        data_types = [DataConfig.AreaLayer & dict(stimulus_type='stimulus.Frame', exclude='images,responses')]

        conds = [dj.AndList(comb) for comb in chain(
            product(core_types, ro_types, shifter_types, mod_types, data_types)
        )]


        targets = []
        for cond in conds:
            keys = best_model(Model & (NetworkConfig.CorePlusReadout & cond) & restr).fetch('KEY')
            if len(keys):
              # work around for DataJoint bug #492
              targets.append(keys)

        if not dryrun:
            self.populate(targets)

        return targets


@schema
class TargetDataset(dj.Computed):
    """
    Register target dataset and all of their units
    """

    definition = """
    -> StaticMultiDataset
    -> DataConfig
    """

    class Unit(dj.Part):
        definition = """
        -> master
        readout_key : varchar(128)
        neuron_id  : smallint
        ---
        -> StaticMultiDataset.Member
        -> StaticScan.Unit
        brain_area : varchar(16)   # brain area
        layer:   varchar(16)       # layer
        """

    def make(self, key):
        # get the model
        self.insert1(key)

        # get the training set
        trainsets, _ = DataConfig().load_data(key, tier='train')

        for readout_key in trainsets.keys():
            ro_tuple = dict(key)
            ro_tuple['readout_key'] = readout_key
            member_key = (StaticMultiDataset.Member() & key & 'name="{}"'.format(readout_key)).fetch1('KEY')

            ro_tuple = dict(ro_tuple, **member_key)

            trainset = trainsets[readout_key]

            unit_ids = trainset.neurons.unit_ids
            areas = trainset.neurons.area
            layers = trainset.neurons.layer

            for neuron_id, unit_id, brain_area, layer in zip(count(), unit_ids, areas, layers):
                tuple = dict(ro_tuple)
                tuple['neuron_id'] = neuron_id
                tuple['unit_id'] = unit_id.item()
                tuple['brain_area'] = brain_area.item()
                tuple['layer'] = layer.item()
                self.Unit().insert1(tuple)


@schema
class UtilizedDataset(dj.Manual):
    """
    This table is filled with datasets that were used in actual closed loop experiments.
    """
    definition = """
    -> TargetDataset
    """

@schema
class OracleRankedUnit(dj.Computed):
    definition = """
    # ranked by the oracle score
    -> TargetDataset.Unit
    ---
    rank: int    # ranking
    oracle_corr: float   # oracle correlation
    top_ptile: float     # percentile from the top
    """
    key_source = TargetDataset()
    def make(self, key):
        keys, corr = (TargetDataset.Unit * Oracle.UnitScores & key).fetch('KEY', 'pearson', order_by='pearson DESC')
        N = len(keys)
        unit_keys = []
        for i, k, c in zip(count(), keys, corr):
            unit_key = dict(k)
            unit_key['rank'] = i
            unit_key['top_ptile'] = i / N
            unit_key['oracle_corr'] = c
            unit_keys.append(unit_key)
        self.insert(unit_keys, ignore_extra_fields=True)



@schema
class ModelGroup(dj.Computed):
    definition = """
    -> TargetDataset
    """

    @property
    def key_source(self):
        return TargetDataset & (NetworkConfig.CorePlusReadout * TargetModel)

    def make(self, key):
        best_cnn = best_model(Model & key & (NetworkConfig.CorePlusReadout * TargetModel & key & [CoreConfig.GaussianLaplace]))
        best_lin = best_model(Model & key & (NetworkConfig.CorePlusReadout * TargetModel & key & [CoreConfig.StackedLinearGaussianLaplace]))

        self.insert1(key)
        self.CNNModel().insert1(best_cnn.fetch1('KEY'))
        self.LinearModel().insert1(best_lin.fetch1('KEY'))
        #self.VGGModel().insert1(best_vgg)


    class CNNModel(dj.Part):
        definition = """
        -> master
        -> TargetModel
        """

    class LinearModel(dj.Part):
        definition = """
        -> master
        -> TargetModel
        """

    # class VGGModel(dj.Part):
    #     definition = """
    #     -> master
    #     -> CoreConfig.VGG
    #     -> TargetModel
    #     """


@schema
class HighUnitSelection(dj.Computed):
    definition = """
    -> TargetDataset.Unit
    ---
    hu_rank: int  # ranking
    """
    # Selection criteria
    # top 25 % oracle
    # top 25 % in fraction oracle performance on CNN and on Linear
    # top 50 neurons with largest performance diff CNN > Linear
    # excluding units that are closer than 25um in x,y plane


    key_source = ModelGroup()

    def make(self, key):

        g = (TargetDataset.Unit & key & HighOracleUnits)
        r = Oracle.UnitScores() & g

        cnn_scores = Model.UnitTestScores & (ModelGroup.CNNModel() & key)
        lin_scores = Model.UnitTestScores & (ModelGroup.LinearModel() & key)

        all_data = r * cnn_scores.proj(cnn_hash='net_hash', cnn_corr='pearson', cnn_seed='seed') * lin_scores.proj(
            lin_hash='net_hash', lin_corr='pearson', lin_seed='seed') * meso.ScanSet.UnitInfo()


        keys, oracle_corr, cnn_corr, lin_corr, posx, posy, posz = all_data.fetch('KEY', 'pearson', 'cnn_corr', 'lin_corr',
                                                                           'um_x', 'um_y', 'um_z')
        keys = np.array(keys)

        cnn_fraction = cnn_corr / oracle_corr
        lin_fraction = lin_corr / oracle_corr
        cnn_thr = np.percentile(cnn_fraction, 75)
        lin_thr = np.percentile(lin_fraction, 75)
        cnn_selection = cnn_fraction > cnn_thr
        lin_selection = lin_fraction > lin_thr
        pos = np.where(cnn_selection & lin_selection)[0]
        print('Overlap = %.2f %%' % (100 * len(pos) / (cnn_selection.sum() + lin_selection.sum() - len(pos))))
        print('%d units remaining' % len(pos))

        delta = (cnn_fraction - lin_fraction)[pos]

        sorted_ordering = pos[np.argsort(delta)[::-1]]

        px = posx[sorted_ordering]
        py = posy[sorted_ordering]
        pz = posz[sorted_ordering]
        good_keys = keys[sorted_ordering]
        xy_nearby = np.sqrt((px[:, None] - px) ** 2 + (py[:, None] - py) ** 2) < 25
        z_nearby = np.abs(pz[:, None] - pz) < 50
        nearby = xy_nearby & z_nearby

        checks = np.tril(nearby, k=-1)

        N = len(sorted_ordering)

        include = np.ones(N).astype(bool)

        #
        for i in range(N):
            if (checks[i] * include).sum() > 0:
                include[i] = False

        print('After distance check, {} units remain'.format(sum(include)))
        assert not np.any(checks[include][:, include]), 'Overlap still remains. Check the algorithm'

        final_keys = good_keys[include]

        for i, k in enumerate(final_keys):
            unit_key = (TargetDataset.Unit & k).fetch1('KEY')
            unit_key['hu_rank'] = i
            self.insert1(unit_key)


        #
        # g = (TargetDataset.Unit & key & HighOracleUnits)
        #
        # r = MesoNetAllOracle.UnitScores() & g
        #
        # common = ['group_id', 'data_hash', 'train_hash', 'animal_id', 'session', 'scan_idx', 'window', 'unit_id']
        # cnn_scores = rename(Encoder.UnitScores() & (ModelGroup.CNNModel() & g), prefix='cnn_', exclude=common)
        # lin_scores = rename(Encoder.UnitScores() & (ModelGroup.LinearModel() & g), prefix='lin_', exclude=common)
        # a = cnn_scores * lin_scores * r
        # keys, cnn_corr, lin_corr, oracle = a.fetch(dj.key, 'cnn_test_corr', 'lin_test_corr', 'pearson')
        #
        # cnn_fraction = cnn_corr / oracle
        # lin_fraction = lin_corr / oracle
        # cnn_thr = np.percentile(cnn_fraction, 25)
        # # lin_thr = np.percentile(lin_fraction, 50)
        # cnn_selection = cnn_fraction > cnn_thr
        # #lin_selection = lin_fraction > lin_thr
        # #pos = np.where(cnn_selection & lin_selection)[0]
        # pos = np.where(cnn_selection)[0]
        # #print('Overlap = %.2f %%' % (100 * len(pos) / (cnn_selection.sum() + lin_selection.sum() - len(pos))))
        # print('%d units remaining' % len(pos))
        #
        # delta = (cnn_fraction - lin_fraction)[pos]
        # delta_thr = np.percentile(delta, 75)
        # pos = pos[delta >= delta_thr]
        #
        # sub = [keys[p] for p in pos]
        #
        # print('Adding %d units' % len(sub))
        # self.insert(TargetDataset.Unit() & sub, ignore_extra_fields=True)

@schema
class CorrectedHighUnitSelection(dj.Computed):
    definition = """
    -> TargetDataset.Unit
    ---
    hu_rank: int  # ranking
    cnn_fraction_oracle: float # CNN model fraction oracle score
    lin_fraction_oracle: float # Linear model fraction oracle score
    """
    # Selection criteria
    # top 50 % oracle
    # top 30 % in fraction oracle performance on CNN and on Linear
    # top neurons with largest performance diff CNN > Linear
    # excluding units that are closer than 20um in x,y plane


    key_source = ModelGroup()

    def make(self, key):

        g = (TargetDataset.Unit & key & (OracleRankedUnit & 'top_ptile < 0.50'))
        r = Oracle.UnitScores() & g

        cnn_scores = Model.UnitTestScores & (ModelGroup.CNNModel() & key)
        lin_scores = Model.UnitTestScores & (ModelGroup.LinearModel() & key)

        all_data = r * cnn_scores.proj(cnn_hash='net_hash', cnn_corr='pearson', cnn_seed='seed') * lin_scores.proj(
            lin_hash='net_hash', lin_corr='pearson', lin_seed='seed') * meso.ScanSet.UnitInfo() * meso.ScanSet.Unit * meso.ScanInfo.Field

        # exclude neurons close to the edge of the field
        dist_thr = 10  # distance for exclusion
        n_before = len(all_data)
        all_data = (all_data & 'ABS(px_y - px_height / 2) < px_height / 2 - {} * px_height / um_height'.format(dist_thr) &
                    'ABS(px_x - px_width / 2) < px_width / 2 - {} * px_height / um_height'.format(dist_thr))
#        h, w, cx, cy = (meso.ScanInfo.Field & (TargetDataset.Unit & key) & 'field = 1').fetch1('um_height', 'um_width', 'x', 'y')
#        all_data = all_data & 'ABS(um_y - {}) < {}'.format(cy, h / 2 - dist_thr) & 'ABS(um_x - {}) < {}'.format(cx, w / 2 - dist_thr
        n_after = len(all_data)

        print('Excluded {} / {} neurons lying within {} of edge'.format(n_before - n_after, n_before, dist_thr))



        keys, oracle_corr, cnn_corr, lin_corr, posx, posy, posz = all_data.fetch('KEY', 'pearson', 'cnn_corr',
                                                                                 'lin_corr',
                                                                                 'um_x', 'um_y', 'um_z')
        keys = np.array(keys)

        cnn_fraction = cnn_corr / oracle_corr
        lin_fraction = lin_corr / oracle_corr
        cnn_thr = np.percentile(cnn_fraction, 70)
        lin_thr = np.percentile(lin_fraction, 70)
        cnn_selection = cnn_fraction > cnn_thr
        lin_selection = lin_fraction > lin_thr
        pos = np.where(cnn_selection & lin_selection)[0]
        print('Overlap = %.2f %%' % (100 * len(pos) / (cnn_selection.sum() + lin_selection.sum() - len(pos))))
        print('%d units remaining' % len(pos))

        delta = (cnn_fraction - lin_fraction)[pos]

        sorted_ordering = pos[np.argsort(delta)[::-1]]

        px = posx[sorted_ordering]
        py = posy[sorted_ordering]
        pz = posz[sorted_ordering]
        good_keys = keys[sorted_ordering]
        xyz_dist = np.sqrt((px[:, None] - px) ** 2 + (py[:, None] - py) ** 2 + (pz[:, None] - pz) ** 2)
        nearby = xyz_dist < 20

        checks = np.tril(nearby, k=-1)

        N = len(sorted_ordering)

        include = np.ones(N).astype(bool)

        #
        for i in range(N):
            if (checks[i] * include).sum() > 0:
                include[i] = False

        final_index = sorted_ordering[include]

        print('After distance check, {} units remain'.format(sum(include)))
        assert not np.any(checks[include][:, include]), 'Overlap still remains. Check the algorithm'

        final_keys = good_keys[include]
        cnn_f = cnn_fraction[final_index]
        lin_f = lin_fraction[final_index]

        for i, k, cf, lf in zip(count(), final_keys, cnn_f, lin_f):
            unit_key = (TargetDataset.Unit & k).fetch1('KEY')
            unit_key['hu_rank'] = i
            unit_key['cnn_fraction_oracle'] = cf
            unit_key['lin_fraction_oracle'] = lf
            self.insert1(unit_key)


        #
        # g = (TargetDataset.Unit & key & HighOracleUnits)
        #
        # r = MesoNetAllOracle.UnitScores() & g
        #
        # common = ['group_id', 'data_hash', 'train_hash', 'animal_id', 'session', 'scan_idx', 'window', 'unit_id']
        # cnn_scores = rename(Encoder.UnitScores() & (ModelGroup.CNNModel() & g), prefix='cnn_', exclude=common)
        # lin_scores = rename(Encoder.UnitScores() & (ModelGroup.LinearModel() & g), prefix='lin_', exclude=common)
        # a = cnn_scores * lin_scores * r
        # keys, cnn_corr, lin_corr, oracle = a.fetch(dj.key, 'cnn_test_corr', 'lin_test_corr', 'pearson')
        #
        # cnn_fraction = cnn_corr / oracle
        # lin_fraction = lin_corr / oracle
        # cnn_thr = np.percentile(cnn_fraction, 25)
        # # lin_thr = np.percentile(lin_fraction, 50)
        # cnn_selection = cnn_fraction > cnn_thr
        # #lin_selection = lin_fraction > lin_thr
        # #pos = np.where(cnn_selection & lin_selection)[0]
        # pos = np.where(cnn_selection)[0]
        # #print('Overlap = %.2f %%' % (100 * len(pos) / (cnn_selection.sum() + lin_selection.sum() - len(pos))))
        # print('%d units remaining' % len(pos))
        #
        # delta = (cnn_fraction - lin_fraction)[pos]
        # delta_thr = np.percentile(delta, 75)
        # pos = pos[delta >= delta_thr]
        #
        # sub = [keys[p] for p in pos]
        #
        # print('Adding %d units' % len(sub))
        # self.insert(TargetDataset.Unit() & sub, ignore_extra_fields=True)

@schema
class NonSpatialHighUnitSelection(dj.Computed):
    definition = """
    -> TargetDataset.Unit
    ---
    hu_rank: int  # ranking
    cnn_fraction_oracle: float # CNN model fraction oracle score
    lin_fraction_oracle: float # Linear model fraction oracle score
    """
    # Selection criteria
    # top 50 % oracle
    # top 30 % in fraction oracle performance on CNN and on Linear
    # top neurons with largest performance diff CNN > Linear
    # NOT excluding units that are closer than 20um in x,y plane


    key_source = ModelGroup()

    def make(self, key):

        g = (TargetDataset.Unit & key & (OracleRankedUnit & 'top_ptile < 0.50'))
        r = Oracle.UnitScores() & g

        cnn_scores = Model.UnitTestScores & (ModelGroup.CNNModel() & key)
        lin_scores = Model.UnitTestScores & (ModelGroup.LinearModel() & key)

        all_data = r * cnn_scores.proj(cnn_hash='net_hash', cnn_corr='pearson', cnn_seed='seed') * lin_scores.proj(
            lin_hash='net_hash', lin_corr='pearson', lin_seed='seed') * meso.ScanSet.UnitInfo()


        keys, oracle_corr, cnn_corr, lin_corr = all_data.fetch('KEY', 'pearson', 'cnn_corr', 'lin_corr')
        keys = np.array(keys)

        cnn_fraction = cnn_corr / oracle_corr
        lin_fraction = lin_corr / oracle_corr
        cnn_thr = np.percentile(cnn_fraction, 70)
        lin_thr = np.percentile(lin_fraction, 70)
        cnn_selection = cnn_fraction > cnn_thr
        lin_selection = lin_fraction > lin_thr
        pos = np.where(cnn_selection & lin_selection)[0]
        print('Overlap = %.2f %%' % (100 * len(pos) / (cnn_selection.sum() + lin_selection.sum() - len(pos))))
        print('%d units remaining' % len(pos))

        delta = (cnn_fraction - lin_fraction)[pos]

        sorted_ordering = pos[np.argsort(delta)[::-1]]

        good_keys = keys[sorted_ordering]
        cnn_f = cnn_fraction[sorted_ordering]
        lin_f = lin_fraction[sorted_ordering]

        for i, k, cf, lf in zip(count(), good_keys, cnn_f, lin_f):
            unit_key = (TargetDataset.Unit & k).fetch1('KEY')
            unit_key['hu_rank'] = i
            unit_key['cnn_fraction_oracle'] = cf
            unit_key['lin_fraction_oracle'] = lf
            self.insert1(unit_key)


def rand_images(img_shape=(1, 36, 64), n_images=1):
    """
    Generates n_images random images of shape img_shape
    Args:
        img_shape: shape of the image. Needs to at least specificy the [channel, height, width]. If larger than length
        3, only the last three dimensions are used to infer necessary image size information.
        n_images: number of random images to generate. Defaults to 1

    Returns:
        A numpy array representing a stack of random images, with dimension [n_images, channel, height, width].

    """
    return np.random.randn(n_images, *img_shape[-3:]).astype(np.float32)


def contrast_tuning(model, img, bias, scale, min_contrast=0.01, n=1000, linear=True, use_max_lim=False):
    mu = img.mean()
    delta = img - img.mean()
    vmax = delta.max()
    vmin = delta.min()

    min_pdist = delta[delta > 0].min()
    min_ndist = (-delta[delta < 0]).min()

    max_lim_gain = max((255 - mu) / min_pdist, mu / min_ndist)

    base_contrast = img.std()

    lim_contrast = 255 / (vmax - vmin) * base_contrast # maximum possible reachable contrast without clipping
    min_gain = min_contrast / base_contrast
    max_gain = min((255 - mu) / vmax, -mu / vmin)

    def run(x):
        with torch.no_grad():
            img = torch.Tensor(process(x[..., None], mu=bias, sigma=scale)[None, ...]).cuda()
            result = model(img)
        return result

    target = max_lim_gain if use_max_lim else max_gain

    if linear:
        gains = np.linspace(min_gain, target, n)
    else:
        gains = np.logspace(np.log10(min_gain), np.log10(target), n)
    vals = []
    cont = []

    for g in tqdm(gains):
        img = delta * g + mu
        img = np.clip(img, 0, 255)
        c = img.std()
        v = run(img).data.cpu().numpy()[0]
        cont.append(c)
        vals.append(v)

    vals = np.array(vals)
    cont = np.array(cont)

    return cont, vals, lim_contrast




# def adjust_contrast(img, contrast=-1, mu=-1, force=False, verbose=False, steps=1000):
#     current_contrast = img.std()
#
#     if contrast < 0:
#         gain = 1   # no adjustment of contrast
#     else:
#         gain = contrast / current_contrast
#
#     delta = img - img.mean()
#     if mu is None or mu < 0:
#         mu = img.mean()
#
#     min_pdist = delta[delta > 0].min()
#     min_ndist = (-delta[delta < 0]).min()
#
#     max_lim_gain = max((255 - mu) / min_pdist, mu / min_ndist)
#
#
#     vmax = delta.max()
#     vmin = delta.min()
#
#     max_gain = min((255 - mu) / vmax, -mu / vmin)
#     clipped = gain > max_gain
#     v = np.linspace(0, 50, 100)
#     if clipped and force:
#         if verbose:
#             print('Adjusting...')
#         cont = []
#         imgs = []
#         gains = np.logspace(np.log10(gain), np.log10(max_lim_gain), steps)
#         # for each gain, perform offset adjustment such that the mean is equal to the set value
#         for g in gains:
#             img = delta * g + mu
#             img = np.clip(img, 0, 255)
#             offset = img.mean() - mu
#             if offset < 0:
#                 offset = -offset
#                 mask = (255-img < v[:, None, None])
#                 nlow = mask.sum(axis=(1, 2))
#                 nhigh = img.size - nlow
#                 va = ((mask * (255-img)).sum(axis=(1, 2)) + v * nhigh) / (nlow + nhigh)
#                 pos = np.argmin(np.abs(va - offset))
#                 actual_offset = -v[pos]
#             else:
#                 mask = (img < v[:, None, None])
#                 nlow = mask.sum(axis=(1, 2))
#                 nhigh = img.size - nlow
#                 va = ((mask * img).sum(axis=(1, 2)) + v * nhigh) / (nlow + nhigh)
#                 pos = np.argmin(np.abs(va - offset))
#                 actual_offset = v[pos]
#             img = img - actual_offset
#             img = np.clip(img, 0, 255)
#             c = img.std()
#             cont.append(c)
#             imgs.append(img)
#             if c > contrast:
#                 break
#         loc = np.argmin(np.abs(np.array(cont) - contrast))
#         adj_img = imgs[loc]
#     else:
#         adj_img = delta * gain + mu
#         adj_img = np.clip(adj_img, 0, 255)
#     actual_contrast = adj_img.std()
#     return adj_img, clipped, actual_contrast

def adjust_contrast(img, contrast=-1, mu=-1, img_min=0, img_max=255, force=True, verbose=False, steps=5000):
    """
    Performs contrast adjustment of the image, being mindful of the image value bounds (e.g. [0, 255]). Given the bounds
    the normal shift and scale will not guarantee that the resulting image still has the desired mean luminance
    and contrast.
    Args:
        img: image to adjsut the contrast
        contrast: desired contrast - this is taken to be the RMS contrast
        mu: desired mean value of the final image
        img_min: the minimum pixel intensity allowed
        img_max: the maximum pixel intensity allowed
        force: if True, iterative approach is taken to produce an image with the desired stats. This will likely cause
        some pixels to saturate in the upper and lower bounds. If False, then image is scaled simply based on ratio of
        current and desired contrast, and then clipped. This likely results in an image that is higher in contrast
        than the original but not quite at the desired contrast and with some pixel information lost due to clipping.
        verbose: If True, prints out progress during iterative adjustment
        steps: If force=True, this sets the number of iterative steps to be used in adjusting the image. The larger the
        value, the closer the final image would approach the desired contrast.

    Returns:
        adjusted_image - a new image adjusted from the original such that the desired mean/contrast is achieved to the
            best of the configuration.
        clipped - Whether any clipping took place. If True, it indicates that some clipping of pixel intensities occured
            and thus some pixel information was lost.
        actual_contrast - the final contrast of the image reached


    """
    current_contrast = img.std()

    if contrast < 0:
        gain = 1   # no adjustment of contrast
    else:
        gain = contrast / current_contrast

    delta = img - img.mean()
    if mu is None or mu < 0: # no adjustment of mean
        mu = img.mean()

    min_pdist = delta[delta > 0].min()
    min_ndist = (-delta[delta < 0]).min()

    # point beyond which scaling would completely saturate out the image (e.g. all pixels would be completely black or
    # white)
    max_lim_gain = max((img_max - mu) / min_pdist, (mu - img_min) / min_ndist)


    vmax = delta.max()
    vmin = delta.min()

    # maximum gain that could be used without losing image information
    max_gain = min((img_max - mu) / vmax, (img_min-mu) / vmin)

    # if True, we already know that the desired contrast cannot be achieved without losing some pixel information
    # into the saturation regime
    clipped = gain > max_gain

    v = np.linspace(0, (img_max-img_min), steps) # candidates for mean adjustment
    if clipped and force:
        if verbose:
            print('Adjusting...')
        cont = []
        imgs = []
        gains = np.logspace(np.log10(gain), np.log10(max_lim_gain), steps)
        # for each gain, perform offset adjustment such that the mean is equal to the set value
        for g in gains:
            img = delta * g + mu
            img = np.clip(img, img_min, img_max)
            offset = img.mean() - mu # shift in clipped image mean caused by the clipping
            if offset < 0: # pixel values needs to be raised
                offset = -offset
                mask = (img_max-img < v[:, None, None])
                nlow = mask.sum(axis=(1, 2)) # pixels that are closer to the bound than v
                nhigh = img.size - nlow
                # calculate the actual shift in mean that can be achieved by shifting all pixels by v
                # then clipping
                va = ((mask * (img_max-img)).sum(axis=(1, 2)) + v * nhigh) / (nlow + nhigh)

                # find the best candidate offset that achieves closest to the desired shift in the mean
                pos = np.argmin(np.abs(va - offset))
                actual_offset = -v[pos]
            else:
                mask = (img-img_min < v[:, None, None])
                nlow = mask.sum(axis=(1, 2))
                nhigh = img.size - nlow
                # actual shift in mean that can be achieved by shifting all pixels by v
                va = ((mask * (img-img_min)).sum(axis=(1, 2)) + v * nhigh) / (nlow + nhigh)
                pos = np.argmin(np.abs(va - offset))
                actual_offset = v[pos]


            img = img - actual_offset
            img = np.clip(img, img_min, img_max)
            # actual contrast achieved with this scale and adjustment
            c = img.std()
            cont.append(c)
            imgs.append(img)
            if c > contrast:
                break
        loc = np.argmin(np.abs(np.array(cont) - contrast))
        adj_img = imgs[loc]
    else:
        adj_img = delta * gain + mu
        adj_img = np.clip(adj_img, img_min, img_max)
    actual_contrast = adj_img.std()
    return adj_img, clipped, actual_contrast


def adjust_contrast_with_mask(img, img_mask=None, contrast=-1, mu=-1, img_min=0, img_max=255, force=True, verbose=False,
                              mu_steps=500, gain_steps=500):
    """
    A version of the contrast adjustment that is mindful of the mask

    Performs contrast adjustment of the image, being mindful of the image value bounds (e.g. [0, 255]). Given the bounds
    the normal shift and scale will not guarantee that the resulting image still has the desired mean luminance
    and contrast.
    Args:
        img: image to adjsut the contrast
        contrast: desired contrast - this is taken to be the RMS contrast
        mu: desired mean value of the final image
        img_min: the minimum pixel intensity allowed
        img_max: the maximum pixel intensity allowed
        force: if True, iterative approach is taken to produce an image with the desired stats. This will likely cause
        some pixels to saturate in the upper and lower bounds. If False, then image is scaled simply based on ratio of
        current and desired contrast, and then clipped. This likely results in an image that is higher in contrast
        than the original but not quite at the desired contrast and with some pixel information lost due to clipping.
        verbose: If True, prints out progress during iterative adjustment
        steps: If force=True, this sets the number of iterative steps to be used in adjusting the image. The larger the
        value, the closer the final image would approach the desired contrast.

    Returns:
        adjusted_image - a new image adjusted from the original such that the desired mean/contrast is achieved to the
            best of the configuration.
        clipped - Whether any clipping took place. If True, it indicates that some clipping of pixel intensities occured
            and thus some pixel information was lost.
        actual_contrast - the final contrast of the image reached


    """
    if img_mask is None:
        img_mask = np.ones_like(img)

    def get_mu(x):
        return np.sum(img_mask * x) / np.sum(img_mask)

    def get_sigma(x):
        h, w = x.shape[-2:]
        avg = get_mu(x)
        return np.sqrt(np.sum(img_mask ** 2 * (x - avg) ** 2) / (h * w))

    adj_img = adj_img * img_mask + mu * (1 - img_mask)
    adj_img = np.clip(adj_img, img_min, img_max)
    mimg = img_mask * img
    test_img = np.clip(mimg - mimg.mean() + mu, img_min, img_max)
    current_contrast = test_img.std()
    if verbose:
        print('Initial contrast:', current_contrast)

    if contrast < 0:
        gain = 1  # no adjustment of contrast
    else:
        gain = contrast / current_contrast

    delta = (img - get_mu(img))  # * bin_mask # only consider deltas in mask region
    if mu is None or mu < 0:  # no adjustment of mean
        mu = get_mu(img)

    min_pdist = delta[delta > 0].min()
    min_ndist = (-delta[delta < 0]).min()

    # point beyond which scaling would completely saturate out the image (e.g. all pixels would be completely black or
    # white)
    max_lim_gain = min(max((img_max - mu) / min_pdist, (mu - img_min) / min_ndist), 100)

    vmax = (delta * img_mask).max()
    vmin = (delta * img_mask).min()

    # maximum gain that could be used without losing image information
    max_gain = min((img_max - mu) / vmax, (img_min - mu) / vmin)


    # if True, we already know that the desired contrast cannot be achieved without losing some pixel information
    # into the saturation regime
    clipped = gain > max_gain
    print('gains', gain , max_gain)

    v = np.linspace(0, (img_max - img_min), mu_steps)  # candidates for mean adjustment

    if clipped and force:
        if verbose:
            print('Adjusting...')
        cont = []
        imgs = []
        gains = np.logspace(np.log10(gain), np.log10(max_lim_gain), gain_steps)
        # for each gain, perform offset adjustment such that the mean is equal to the set value
        for g in tqdm(gains, disable=(not verbose)):
            print('')
            img = delta * g + mu
            img = np.clip(img, img_min, img_max)

            offset = mu - get_mu(img)  # shift in clipped image mean caused by the clipping
            if offset > 0:
                sign = 1
                edge = img_max
            else:
                sign = -1
                edge = img_min

            offset = sign * offset
            mask = (sign * (edge - img) < v[:, None, None])

            nlow = (mask * img_mask).sum(axis=(1, 2))  # effective number of pixels that are closer to the bound than v
            nhigh = img_mask.sum() - nlow

            # calculate the actual shift in mean that can be achieved by shifting all pixels by v
            # then clipping
            va = ((mask * img_mask * sign * (edge - img)).sum(axis=(1, 2)) + (
                        v[:, None, None] * img_mask * (1 - mask)).sum(axis=(1, 2))) / (nlow + nhigh)

            # find the best candidate offset that achieves closest to the desired shift in the mean
            pos = np.argmin(np.abs(va - offset))
            actual_offset = sign * v[pos]

            img = img + actual_offset
            img = np.clip(img, img_min, img_max)
            # actual contrast achieved with this scale and adjustment
            c = get_sigma(img)
            print('contrast now', c)
            cont.append(c)
            imgs.append(img)
            if c > contrast:
                break
        loc = np.argmin(np.abs(np.array(cont) - contrast))
        adj_img = imgs[loc]
    else:
        adj_img = delta * gain + mu

    adj_img = adj_img * img_mask + mu * (1 - img_mask)
    adj_img = np.clip(adj_img, img_min, img_max)
    actual_contrast = adj_img.std()
    return adj_img, clipped, actual_contrast


def adjust_img_stats(img, mu, sigma, img_min=0, img_max=255, mask=None, max_gain=6000, min_gain=0.001,
                           base_ratio=1.05, verbose=False):
    if mask is None:
        mask = np.ones_like(img)
    mimg = img * mask

    delta = img - mimg.sum() / mask.sum()

    def get_image(delta, offset=0):
        return np.clip((delta * mask) + mu + offset, img_min, img_max)

    sigma_i = get_image(delta).std()
    if sigma_i < 1e-8:
        warnings.warn('Zero standard deviation detected.')
        img = np.clip((delta * mask) + mu, img_min, img_max)  # flat image
        unmasked_img = np.clip(delta + mu, img_min, img_max)
    else:
        gain = sigma / sigma_i

        dir_sign = np.sign(np.log(gain))
        ratio = base_ratio ** dir_sign


        max_gain = min(max(img_max - mu, mu - img_min) / np.min(np.abs(delta)[mask > 0]), max_gain)
        min_gain = min_gain

        v = np.linspace(-30, 30, 1000)[:, None, None]

        imgs = []
        unmasked_imgs = []
        conts = []
        while True:
            if verbose:
                print('Trying gain', gain)
            adj_images = get_image(delta * gain, v)
            pos = np.argmin(np.abs(adj_images.mean(axis=(1, 2)) - mu))
            img = adj_images[pos]
            unmasked_img = np.clip(delta * gain + mu + v[pos], img_min, img_max)

            cont = img.std()
            if verbose:
                print('Got mean and contrast', img.mean(), cont)
            imgs.append(img)
            conts.append(cont)
            unmasked_imgs.append(unmasked_img)
            if (cont - sigma) * dir_sign > 0:
                break
            gain = gain * ratio
            if gain > max_gain or gain < min_gain:
                break

        pos = np.argmin(np.abs(np.array(conts) - sigma))
        img = imgs[pos]
        unmasked_img = unmasked_imgs[pos]
    if verbose:
        print('Selected version with mu={} and std={}'.format(img.mean(), img.std()))
    return img, unmasked_img


def normalize(img, radius=1):
    """
    Normalize image to L2 norm of radius
    """
    img = img - img.mean(-2, keepdim=True).mean(-1, keepdim=True)
    norms = img.pow(2).sum(-2, keepdim=True).sum(-1, keepdim=True).sqrt()
    return img / norms * radius


@schema
class MEIParameter(dj.Lookup):
    definition = """
    # parameters for the MEI generation

    mei_param_id        : varchar(64)  # id
    ---
    iter_n              : int   # number of iterations to run
    start_sigma         : float # starting sigma value
    end_sigma           : float # ending sigma value
    start_step_size     : float # starting step size
    end_step_size       : float # ending step size
    precond             : float # strength of gradient preconditioning filter falloff
    step_gain           : float # scaling of gradient steps
    jitter              : int   #size of translational jittering
    blur                : bool  # whether to apply bluring or not
    norm                : float # norm adjustment after step, negative to turn off
    train_norm          : float # norm adjustment during step, negative to turn off
    """

    @property
    def contents(self):
        yield from map(lambda x: (list_hash(x),) + x,
           (
               (1000, 1.5, 0.01, 3.0, 0.125, 0.1, 0.1, 0, True, -1, -1),
               (1000, 1.5, 0.01, 3.0, 0.125, 0.1, 0.1, 0, False, -1, 11.0),
           )
        )


def get_multi_model(key):
    # load the model
    key_seed = dict(key)
    key_seed.pop('seed')
    model_keys = (Model() & key_seed).fetch('KEY')

    models = []
    for mk in model_keys:
        model = (Model() & mk).load_network().to('cuda')
        model.eval()
        models.append(model)

    return models


def multi_model_wrapper(models):
    def compute(x, readout_key, eye_pos=None, behavior=None):
        resp = 0
        for m in models:
            resp = resp + m(x, readout_key, eye_pos=eye_pos, behavior=behavior)
        return resp / len(models)
    return compute


def get_adj_model(models, readout_key, neuron_id, mu_eye=None, pos=None, mu_beh=None):
    if pos is None:
        def adj_model(x):
            resp = 0
            for m in models:
                resp = resp + m(x, readout_key, eye_pos=mu_eye, behavior=mu_beh)[:, neuron_id]
            return resp / len(models)
    else:
        def adj_model(x):
            return models[pos](x, readout_key, eye_pos=mu_eye, behavior=mu_beh)[:, neuron_id]

    return adj_model



@schema
class MultiModelEvaluation(dj.Computed):
    definition = """
    -> TargetModel
    -> TargetDataset
    ---
    n_models: int         # number of models that are combined
    test_score: float    # average testset score
    """

    key_source = TargetModel * TargetDataset & NetworkConfig.CorePlusReadout

    class UnitScore(dj.Part):
        definition = """
        -> master
        -> TargetDataset.Unit
        ---
        unit_score: float   # correlation score
        oracle_score: float # oracle correlation
        fraction_oracle: float  # fraction oracle
        """

    def make(self, key):
        testsets, testloaders = DataConfig().load_data(key, tier='test', cuda=True, batch_size=30)

        models = get_multi_model(key)
        comb_model = multi_model_wrapper(models)

        scores, unit_scores = [], []
        oracle_units = Oracle.UnitScores * TargetDataset.Unit & key
        for readout_key, testloader in testloaders.items():
            y, y_hat = compute_predictions(testloader, comb_model, readout_key)
            perf_scores = compute_scores(y, y_hat)

            scores.append(perf_scores.pearson)
            unit_scores.extend(
                [dict(key, readout_key=readout_key, neuron_id=u, unit_score=c) for u, c in zip(count(), perf_scores.pearson)])

        for unit_key in unit_scores:
            unit_key['oracle_score'] = (oracle_units & unit_key).fetch1('pearson')
            unit_key['fraction_oracle'] = unit_key['unit_score'] / unit_key['oracle_score']

        key['n_models'] = len(models)
        key['test_score'] = np.concatenate(scores).mean()

        self.insert1(key)
        self.UnitScore.insert(unit_scores)


@schema
class FrozenMultiModelEvaluation(dj.Computed):
    """
    Version in which the multi-model is "frozen" -- it makes use of the mean eye position and behavioral state
    everywhere. This gives a fair comparison to Oracle
    """
    definition = """
    -> TargetModel
    -> TargetDataset
    ---
    n_models: int         # number of models that are combined
    test_score: float    # average testset score
    """

    key_source = TargetModel * TargetDataset & NetworkConfig.CorePlusReadout

    class UnitScore(dj.Part):
        definition = """
        -> master
        -> TargetDataset.Unit
        ---
        unit_score: float   # correlation score
        oracle_score: float # oracle correlation
        fraction_oracle: float  # fraction oracle
        """

    def make(self, key):
        testsets, testloaders = DataConfig().load_data(key, tier='test', cuda=True, batch_size=30)

        models = get_multi_model(key)
        comb_model = multi_model_wrapper(models)

        scores, unit_scores = [], []
        oracle_units = Oracle.UnitScores * TargetDataset.Unit & key
        for readout_key, testloader in testloaders.items():
            _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
            # override behavioral information with average behavioral values
            y, y_hat = compute_predictions(testloader, comb_model, readout_key, eye_pos=mu_eye, behavior=mu_beh)
            perf_scores = compute_scores(y, y_hat)

            scores.append(perf_scores.pearson)
            unit_scores.extend(
                [dict(key, readout_key=readout_key, neuron_id=u, unit_score=c) for u, c in zip(count(), perf_scores.pearson)])

        for unit_key in unit_scores:
            unit_key['oracle_score'] = (oracle_units & unit_key).fetch1('pearson')
            unit_key['fraction_oracle'] = unit_key['unit_score'] / unit_key['oracle_score']

        key['n_models'] = len(models)
        key['test_score'] = np.concatenate(scores).mean()

        self.insert1(key)
        self.UnitScore.insert(unit_scores)


@schema
class MEI(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    n_seeds             : int       # number of distinct seeded models used
    mei                 : longblob  # most exciting images
    activation          : float     # activation at the MEI
    monotonic           : bool      # does activity increase monotonically with contrast
    max_contrast        : float     # contrast at which maximum activity is achived
    max_activation      : float     # activation at the maximum contrast
    sat_contrast        : float     # contrast at which image would start saturating
    img_mean            : float     # mean luminance of the image
    lim_contrast        : float     # max reachable contrast without clipping
    """

    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return TargetModel() * MEIParameter() * TargetDataset.Unit & NetworkConfig.CorePlusReadout



    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))



        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        print('Working with images with mu={}, sigma={}'.format(bias, scale))

        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)


        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])
        norm = float(params['norm'])
        train_norm = float(params['train_norm'])

        octaves = [
            {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size': float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)

        # generate class visualization via octavewise gradient ascent
        gen_image = deepdraw(adj_model, gen_image, octaves, clip=True,
                             random_crop=False, blur=blur, jitter=jitter,
                             precond=precond, step_gain=step_gain,
                             bias=bias, scale=scale, norm=norm, train_norm=train_norm)

        mei = gen_image.squeeze()

        with torch.no_grad():
            img = torch.Tensor(process(gen_image, mu=bias, sigma=scale)[None, ...]).to('cuda')
            activation = adj_model(img).data.cpu().numpy()[0]

        cont, vals, lim_contrast = contrast_tuning(adj_model, mei, bias, scale)

        key['n_seeds'] = len(models)
        key['mei'] = mei
        key['activation'] = activation
        key['monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['max_activation'] = np.max(vals)
        key['max_contrast'] = cont[np.argmax(vals)]
        key['sat_contrast'] = np.max(cont)
        key['img_mean'] = mei.mean()
        key['lim_contrast'] = lim_contrast

        self.insert1(key)




def gauss2d(vx, vy, mu, cov):
    input_shape = vx.shape
    mu_x, mu_y = mu
    v = np.stack([vx.ravel() - mu_x, vy.ravel() - mu_y])
    cinv = inv(cholesky(cov))
    y = cinv @ v
    g = np.exp(-0.5 * (y * y).sum(axis=0))
    return g.reshape(input_shape)


def fit_gauss_envelope(img):
    """
    Given an image, finds a Gaussian fit to the image by treating the square of mean shifted image as the distribution.
    Args:
        img:

    Returns:

    """
    vx, vy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    rect = (img - img.mean()) ** 2
    pdf = rect / rect.sum()
    mu_x = (vx * pdf).sum()
    mu_y = (vy * pdf).sum()

    cov_xy = (vx * vy * pdf).sum() - mu_x * mu_y
    cov_xx = (vx ** 2 * pdf).sum() - mu_x ** 2
    cov_yy = (vy ** 2 * pdf).sum() - mu_y ** 2

    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

    g = gauss2d(vx, vy, (mu_x, mu_y), cov)
    mu = (mu_x, mu_y)
    return mu, cov, np.sqrt(g.reshape(img.shape))



@schema
class GaussianFactor(dj.Lookup):
    definition = """
    gaussian_factor_id: int
    ---
    gaussian_factor: float
    """

    # if you are wondering where this number came from, that's a great question! It was derived from fairly
    # ad-hoc procedure in which I matched the energy statistics of the masked matched natural images to that
    # of the MEI images
    contents = [(0, 1.875)]



@schema
class GaussianMask(dj.Computed):
    definition = """
    -> MEI
    -> GaussianFactor
    ---
    mei_mask: longblob    # Gaussian mei mask
    """

    def make(self, key):
        factor = (GaussianFactor & key).fetch1('gaussian_factor')
        img = (MEI & key).fetch1('mei')
        *_, mask = fit_gauss_envelope(img)
        key['mei_mask'] = mask ** factor  # this helps to counteract the overly wide Gaussian fit
        self.insert1(key)


@schema
class GaussianMaskFit(dj.Computed):
    definition = """
    -> MEI
    -> GaussianFactor
    ---
    mu_x: float           # center of mask in width dimension
    mu_y: float           # center of mask in height dimension
    cov_x: float        # var along x
    cov_y: float        # var along y
    cov_xy: float       # covariance of x and y 
    mei_mask: longblob    # Gaussian mei mask
    """

    def make(self, key):
        factor = (GaussianFactor & key).fetch1('gaussian_factor')
        img = (MEI & key).fetch1('mei')
        mu, cov, mask = fit_gauss_envelope(img)
        key['mu_x'] = mu[0]
        key['mu_y'] = mu[1]
        key['cov_x'] = cov[0, 0]
        key['cov_y'] = cov[1, 1]
        key['cov_xy'] = cov[0, 1]
        key['mei_mask'] = mask ** factor  # this helps to counteract the overly wide Gaussian fit
        self.insert1(key)




@schema
class MaskParameter(dj.Lookup):
    definition = """
    mask_param_id: int
    ---
    delta_thr: float  # threshold on the difference from mean
    size_thr:  float  # threshold for size of blobs
    expansion_sigma: float   # mask size expansion sigma
    expansion_thr: float     # mask size expansion threshold
    filter_sigma: float # sigma for final gaussian blur
    """
    contents = [(0, 16, 50, 3, 0.3, 2)]



def remove_small_area(mask, size_threshold=50):
    """
    Removes contiguous areas in a thresholded image that is smaller in the number of pixels than size_threshold.
    """
    mask_mod = mask.copy()
    label_im, nb_labels = label(mask_mod)
    for v in range(0, nb_labels+1):
        area = label_im == v
        s = np.sum(area)
        if s < size_threshold:
            mask_mod[area] = 0
    return mask_mod

@schema
class MEIMask(dj.Computed):
    definition = """
    -> MEI
    -> MaskParameter
    ---
    mei_mask: longblob   # float value mask for the longblob 
    """

    def make(self, key):
        delta_thr, size_thr, expansion_sigma, expansion_thr, filter_sigma = (MaskParameter & key).fetch1('delta_thr', 'size_thr',
                                                                                                         'expansion_sigma', 'expansion_thr',
                                                                                                         'filter_sigma')
        img = (MEI & key).fetch1('mei')
        img = img.copy(order='c')
        delta = img - img.mean()
        mask = np.abs(delta) > delta_thr
        # remove small lobes - likely an artifact
        mask = remove_small_area(mask, size_threshold=size_thr)
        # fill in the gap between lobes
        mask = convex_hull_image(mask)
        # expand the size of the mask
        mask = gaussian_filter(mask.astype(float), sigma=expansion_sigma) > expansion_thr
        # blur the edge, giving smooth transition
        mask = gaussian_filter(mask.astype(float), sigma=filter_sigma)
        key['mei_mask'] = mask
        self.insert1(key)


@schema
class ProcessedImagenet(dj.Computed):
    definition = """
    -> imagenet.Album.Single
    -> Preprocessing
    ---
    frame: external-data   # processed frame of imagenet 
    """

    key_source = imagenet.Album.Single * Preprocessing & 'preproc_id = 0 and collection_id = 4'

    def make(self, key):
        img = (stimulus.StaticImage.Image & key).fetch1('image')
        key['frame'] = process_frame(key, img)
        self.insert1(key)


@schema
class TightMaskParameter(dj.Lookup):
    """
    Parameter for "tight" mask on MEIs
    """
    definition = """
    tight_mask_param_id: int
    ---
    stdev_size_thr:  float  # fraction of standard dev threshold for size of blobs
    filter_sigma: float # sigma for final gaussian blur
    target_reduction_ratio: float  # reduction ratio to achieve for tightening the mask
    """
    contents = [(0, 1.0, 1.0, 0.9),
                (1, 1.0, 1.0, 0.95),
                (2, 1.0, 2.0, 0.9),
                (3, 0.8, 1.0, 0.9),
                (4, 1.0, 3.0, 0.9)]


@schema
class TightMEIMask(dj.Computed):
    definition = """
    -> MEI
    -> TightMaskParameter
    ---
    tight_mask: longblob   # mask for mei
    reduction_ratio:  float  # achieved reduction in activation from the baseline mask
    """

    def make(self, key):

        # get the MEI
        mei = (MEI() & key).fetch1('mei')

        # set in "c" contiguous
        img = mei.copy(order='c')

        stdev_size_thr, filter_sigma, target_reduction_ratio = (TightMaskParameter & key).fetch1('stdev_size_thr',
                                                                                                 'filter_sigma',
                                                                                                 'target_reduction_ratio')


        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        def get_activation(mei):
            with torch.no_grad():
                img = torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
                activation = adj_model(img).data.cpu().numpy()[0]
            return activation

        delta = img - img.mean()
        fluc = np.abs(delta)
        thr = np.std(fluc) * stdev_size_thr

        # original mask
        mask = convex_hull_image((fluc > thr).astype(float))
        fm = gaussian_filter(mask.astype(float), sigma=filter_sigma)
        masked_img = fm * img + (1 - fm) * img.mean()
        activation = base_line = get_activation(masked_img)

        print('Baseline:', base_line)
        count = 0
        while (activation > base_line * target_reduction_ratio):
            mask = erosion(mask, square(3))
            fm = gaussian_filter(mask.astype(float), sigma=filter_sigma)
            masked_img = fm * img + (1 - fm) * img.mean()
            activation  = get_activation(masked_img)
            print('Activation:', activation)
            count += 1

            if count > 100:
                print('This has been going on for too long! - aborting')
                raise ValueError('The activation does not reduce for the given setting')


        key['reduction_ratio'] = activation / base_line
        key['tight_mask'] = fm

        self.insert1(key)




@schema
class BestImageNet(dj.Computed):
    definition = """
    -> TargetModel
    -> TargetDataset.Unit
    ---
    -> stimulus.StaticImage.Image
    img_activation: float   # activation at the best unmasked image
    """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        target_scan = (StaticMultiDataset.Member & key).fetch1('KEY') # note that this only works for networks trained on a single scan StaticMultiDataset
        images, image_ids = (Frame * ConditionTier * stimulus.Frame & target_scan & 'image_class="imagenet"').fetch(
            'frame', 'image_id')

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # activations for each masked image

        img_activations = []
        for image in tqdm(images):
            image = np.atleast_3d(image)  # ensure channel dimension exist
            image = torch.tensor(process(image, mu=bias, sigma=scale)[None, ...], dtype=torch.float32,
                                 requires_grad=True, device='cuda')
            # --- Compute gradient receptive field at the image
            y = adj_model(image)
            img_activations.append(y.item())

        img_activations = np.array(img_activations)
        pos = np.argmax(img_activations)
        key['image_class'] = 'imagenet'
        key['image_id'] = image_ids[pos]
        key['img_activation'] = img_activations[pos]

        self.insert1(key)



# The following implementation contains a bug! Should not be used except for data preservation purpose!
@schema
class BestMaskedImageNet(dj.Computed):
    """
    This implementation contains a bug in which the masked region is turned completely black!!
    """
    definition = """
    -> GaussianMask
    ---
    -> stimulus.StaticImage.Image
    img_activation: float   # activation at the best masked image
    """


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn('Use of this table is deprecated. It is kept only for record keeping purpose')



    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        target_scan = (StaticMultiDataset.Member & key).fetch1('KEY')
        images, image_ids = (Frame * ConditionTier * stimulus.Frame & target_scan & 'image_class="imagenet"').fetch('frame', 'image_id')

        mask = (GaussianMask & key).fetch1('mei_mask')

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # activations for each masked image

        img_activations = []
        for image in tqdm(images):
            image = np.atleast_3d(image * mask) # ensure channel dimension exist
            image = torch.tensor(process(image, mu=bias, sigma=scale)[None, ...], dtype=torch.float32, requires_grad=True, device='cuda')
            # --- Compute gradient receptive field at the image
            y = adj_model(image)
            img_activations.append(y.item())

        img_activations = np.array(img_activations)
        pos = np.argmax(img_activations)
        key['image_class'] = 'imagenet'
        key['image_id'] = image_ids[pos]
        key['img_activation'] = img_activations[pos]

        self.insert1(key)




##### The following all use Gaussian mask

def mask_image(img, mask, background):
    """
    Applies the mask `mask` onto the `img`. The completely masked area is then
    replaced with the value `background`.


    Returns: masked image
    """
    filler = np.full_like(img, background)
    return img * mask + filler * (1-mask)



@schema
class BestGaussianMaskedImageNet(dj.Computed):
    """
    For each GaussianMask fitted for a target neuron, find the best natural image that yields the best response
    """
    definition = """
    -> GaussianMask
    ---
    -> stimulus.StaticImage.Image
    img_activation: float   # activation at the best masked image
    """


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        target_scan = (StaticMultiDataset.Member & key).fetch1('KEY')



        # get all images
        images, image_ids = (Frame * ConditionTier * stimulus.Frame & target_scan & 'image_class="imagenet"').fetch('frame', 'image_id')

        # get Gaussian mask to apply
        mask = (GaussianMask & key).fetch1('mei_mask')

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # activations for each masked image
        img_activations = []
        for image in tqdm(images): # by the way, this approach is not too efficient -- better would have been to batch images
            # apply the mask to the image
            image = np.atleast_3d(mask_image(image, mask, bias)) # ensure channel dimension exist
            # prepare as tensor
            image = torch.tensor(process(image, mu=bias, sigma=scale)[None, ...], dtype=torch.float32, requires_grad=True, device='cuda')
            # --- Compute gradient receptive field at the image
            y = adj_model(image)
            img_activations.append(y.item())

        img_activations = np.array(img_activations)
        # find out the best image to be used
        pos = np.argmax(img_activations)
        key['image_class'] = 'imagenet'
        key['image_id'] = image_ids[pos]
        key['img_activation'] = img_activations[pos]

        self.insert1(key)


@schema
class GaussianMaskedImageResponse(dj.Computed):
    """
    Summary responses of networks on masked and unmasked natural images selected to yield best "masked response"
    """

    definition = """
    -> BestGaussianMaskedImageNet
    ---
    masked_image_resp: float      # activation on the masked image
    original_image_resp: float    # activation on the unmasked image
    """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # use the best Gaussian masked ImageNet image
        image = (stimulus.Frame * Frame * ConditionTier * stimulus.StaticImage.Image * StaticMultiDataset.Member & (BestGaussianMaskedImageNet & key)).fetch1('frame')

        mask = (GaussianMask & key).fetch1('mei_mask')

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # activations for each masked image

        def evaluate_image(x):
            x = np.atleast_3d(x)
            x = torch.tensor(process(x, mu=bias, sigma=scale)[None, ...], dtype=torch.float32,
                             requires_grad=False, device='cuda')
            y = adj_model(x)
            return y.item()

        key['masked_image_resp'] = evaluate_image(mask_image(image, mask, bias))
        key['original_image_resp'] = evaluate_image(image)

        self.insert1(key)



@schema
class GaussianMaskedBestImageResponse(dj.Computed):
    """
    Response of masked vs unmaksed on **images with best unmasked resopsnes**
    """
    definition = """
    -> BestImageNet
    -> GaussianMask
    ---
    masked_image_resp: float      # activation on the masked image
    original_image_resp: float    # activation on the unmasked image
    """


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # select out the best unmasked ImageNet image
        image = (stimulus.Frame * Frame * ConditionTier * stimulus.StaticImage.Image * StaticMultiDataset.Member & (BestImageNet & key)).fetch1('frame')

        mask = (GaussianMask & key).fetch1('mei_mask')

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # activations for each masked image

        def evaluate_image(x):
            x = np.atleast_3d(x)
            x = torch.tensor(process(x, mu=bias, sigma=scale)[None, ...], dtype=torch.float32,
                             requires_grad=False, device='cuda')
            y = adj_model(x)
            return y.item()

        key['masked_image_resp'] = evaluate_image(mask_image(image, mask, bias))
        key['original_image_resp'] = evaluate_image(image)

        self.insert1(key)



@schema
class GaussianMaskedAllResponse(dj.Computed):
    definition = """
        -> GaussianMask
        ---
        original_img_activations: longblob   # activations on the original unmasked images
        masked_img_activations: longblob     # activations on the masked images
        image_ids: longblob                  # image ids
        """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        target_scan = (StaticMultiDataset.Member & key).fetch1('KEY')
        images, image_ids = (Frame * ConditionTier * stimulus.Frame & target_scan & 'image_class="imagenet"').fetch(
            'frame', 'image_id')

        mask = (GaussianMask & key).fetch1('mei_mask')

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # activations for each masked image

        def evaluate_image(x):
            x = np.atleast_3d(x)
            x = torch.tensor(process(x, mu=bias, sigma=scale)[None, ...], dtype=torch.float32,
                             requires_grad=False, device='cuda')
            y = adj_model(x)
            return y.item()

        original_img_activations = []
        masked_img_activations = []
        for image in tqdm(images):
            original_img_activations.append(evaluate_image(image))
            masked_img_activations.append(evaluate_image(mask_image(image, mask, bias)))

        key['original_img_activations'] = original_img_activations
        key['masked_img_activations'] = masked_img_activations
        key['image_ids'] = image_ids
        self.insert1(key)



@schema
class GaussianMaskedSelectedImage(dj.Computed):
    definition = """
    -> BestGaussianMaskedImageNet
    ---
    masked_imagenet_image: longblob   # masked image
    original_imagenet_image: longblob # original image
    """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # get Gauss mask
        mask = (GaussianMask & key).fetch1('mei_mask')
        # UGHHHH hard coding... need this to match the image as stored for ImageNet
        mask = ndimage.zoom(mask, zoom=4, mode='reflect')

        # select out the best Gaussian masked ImageNet image from stimulus.StaticImage.Image
        image = (stimulus.StaticImage.Image & (BestGaussianMaskedImageNet & key)).fetch1('image').astype('float')
        image = np.clip(image, 0, 255)


        # apply the mask
        masked_image = mask_image(image, mask, bias)
        masked_image = np.clip(masked_image, 0, 255)

        key['masked_imagenet_image'] = masked_image
        key['original_imagenet_image'] = image

        self.insert1(key)


## The following four use the "MEI mask" as defined in the `MEIMask` table


@schema
class BestMEIMaskedImageNet(dj.Computed):
    definition = """
    -> MEIMask
    ---
    -> stimulus.StaticImage.Image
    img_activation: float   # activation at the best masked image
    """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        target_scan = (StaticMultiDataset.Member & key).fetch1('KEY')
        images, image_ids = (Frame * ConditionTier * stimulus.Frame & target_scan & 'image_class="imagenet"').fetch('frame', 'image_id')

        mask = (MEIMask & key).fetch1('mei_mask')

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # activations for each masked image

        img_activations = []
        for image in tqdm(images):
            image = np.atleast_3d(mask_image(image, mask, bias)) # ensure channel dimension exist
            image = torch.tensor(process(image, mu=bias, sigma=scale)[None, ...], dtype=torch.float32, requires_grad=True, device='cuda')
            # --- Compute gradient receptive field at the image
            y = adj_model(image)
            img_activations.append(y.item())

        img_activations = np.array(img_activations)
        pos = np.argmax(img_activations)
        key['image_class'] = 'imagenet'
        key['image_id'] = image_ids[pos]
        key['img_activation'] = img_activations[pos]

        self.insert1(key)


@schema
class MEIMaskedImageResponse(dj.Computed):
    definition = """
    -> BestMEIMaskedImageNet
    ---
    masked_image_resp: float      # activation on the masked image
    original_image_resp: float    # activation on the unmasked image
    """


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        # use the best MEI masked ImageNet image
        image = (stimulus.Frame * Frame * ConditionTier * stimulus.StaticImage.Image * StaticMultiDataset.Member & (BestMEIMaskedImageNet & key)).fetch1('frame')

        mask = (MEIMask & key).fetch1('mei_mask')

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # activations for each masked image

        def evaluate_image(x):
            x = np.atleast_3d(x)
            x = torch.tensor(process(x, mu=bias, sigma=scale)[None, ...], dtype=torch.float32,
                             requires_grad=False, device='cuda')
            y = adj_model(x)
            return y.item()

        key['masked_image_resp'] = evaluate_image(mask_image(image, mask, bias))
        key['original_image_resp'] = evaluate_image(image)

        self.insert1(key)


@schema
class MEIMaskedBestImageResponse(dj.Computed):
    """
    Response of masked vs unmaksed on **images with best unmasked resopsnes**
    """
    definition = """
    -> BestImageNet
    -> MEIMask
    ---
    masked_image_resp: float      # activation on the masked image
    original_image_resp: float    # activation on the unmasked image
    """


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        image = (stimulus.Frame * Frame * ConditionTier * stimulus.StaticImage.Image * StaticMultiDataset.Member & (BestImageNet & key)).fetch1('frame')

        mask = (MEIMask & key).fetch1('mei_mask')

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # activations for each masked image

        def evaluate_image(x):
            x = np.atleast_3d(x)
            x = torch.tensor(process(x, mu=bias, sigma=scale)[None, ...], dtype=torch.float32,
                             requires_grad=False, device='cuda')
            y = adj_model(x)
            return y.item()

        key['masked_image_resp'] = evaluate_image(mask_image(image, mask, bias))
        key['original_image_resp'] = evaluate_image(image)

        self.insert1(key)


@schema
class MEIMaskedAllResponse(dj.Computed):
    definition = """
    -> MEIMask
    ---
    original_img_activations: longblob   # activations on the original unmasked images
    masked_img_activations: longblob     # activations on the masked images
    image_ids: longblob                  # image ids
    """


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        target_scan = (StaticMultiDataset.Member & key).fetch1('KEY')
        images, image_ids = (Frame * ConditionTier * stimulus.Frame & target_scan & 'image_class="imagenet"').fetch(
            'frame', 'image_id')

        mask = (MEIMask & key).fetch1('mei_mask')

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # activations for each masked image

        def evaluate_image(x):
            x = np.atleast_3d(x)
            x = torch.tensor(process(x, mu=bias, sigma=scale)[None, ...], dtype=torch.float32,
                             requires_grad=False, device='cuda')
            y = adj_model(x)
            return y.item()

        original_img_activations = []
        masked_img_activations = []
        for image in tqdm(images):
            original_img_activations.append(evaluate_image(image))
            masked_img_activations.append(evaluate_image(mask_image(image, mask, bias)))


        key['original_img_activations'] = original_img_activations
        key['masked_img_activations'] = masked_img_activations
        key['image_ids'] = image_ids
        self.insert1(key)


### Below we use the "tight" MEI mask and find the best masked ImageNet image
### for each target neuron.

@schema
class BestTightMaskedImageNet(dj.Computed):
    definition = """
    -> TightMEIMask
    ---
    -> stimulus.StaticImage.Image
    img_activation: float   # activation at the best masked image
    unmasked_activation: float  # activation of the unmasked best image
    """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        target_scan = (StaticMultiDataset.Member & key).fetch1('KEY')

        images, image_ids = (ProcessedImagenet & target_scan & 'collection_id = 4').fetch('frame', 'image_id')

        mask = (TightMEIMask & key).fetch1('tight_mask')

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # activations for each masked image

        img_activations = []

        def evaluate_image(x):
            x = np.atleast_3d(x)
            x = torch.tensor(process(x, mu=bias, sigma=scale)[None, ...], dtype=torch.float32,
                             requires_grad=False, device='cuda')
            y = adj_model(x)
            return y.item()

        # let sigma be proportional to the mask size
        sigma = scale * np.sqrt(mask.sum() / img_shape[-1] / img_shape[-2])
        for image in tqdm(images):
            adj_img, _ = adjust_img_stats(image, bias, sigma, mask=mask)
            img_activations.append(evaluate_image(adj_img))

        img_activations = np.array(img_activations)
        pos = np.nanargmax(img_activations)
        key['image_class'] = 'imagenet'
        key['image_id'] = image_ids[pos]
        key['img_activation'] = img_activations[pos]
        key['unmasked_activation'] = evaluate_image(images[pos])

        self.insert1(key)




@schema
class WrongMEI(dj.Computed):
    """
    This is WRONG in the sense that it uses the mean of the image as the standard deviation when
    generating the MEI. This reflects the previous bug in Normalizer where the images were standardized
    by the mean instead of std. Hence, when this "wrong" MEI generation is applied on those models,
    it actually generates MEI in a way that is consistent with the intensity range that the network
    was trained under.
    """
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    n_seeds             : int       # number of distinct seeded models used
    mei                 : longblob  # most exciting images
    activation          : float     # activation at the MEI
    monotonic           : tinyint      # does activity increase monotonically with contrast
    max_contrast        : float     # contrast at which maximum activity is achived
    max_activation      : float     # activation at the maximum contrast
    sat_contrast        : float     # contrast at which image would start saturating
    img_mean            : float     # mean luminance of the image
    lim_contrast        : float     # max reachable contrast without clipping
    """

    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return TargetModel() * MEIParameter() * TargetDataset.Unit & NetworkConfig.CorePlusReadout



    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))



        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # !!!!!INTENTIONALLY WRONG OVERRIDE!!!!!!
        scale = bias


        print('Working with images with mu={}, sigma={}'.format(bias, scale))

        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)


        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])
        norm = float(params['norm'])
        train_norm = float(params['train_norm'])

        octaves = [
            {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size': float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)

        # generate class visualization via octavewise gradient ascent
        gen_image = deepdraw(adj_model, gen_image, octaves, clip=True,
                             random_crop=False, blur=blur, jitter=jitter,
                             precond=precond, step_gain=step_gain,
                             bias=bias, scale=scale, norm=norm, train_norm=train_norm)

        mei = gen_image.squeeze()

        with torch.no_grad():
            img = torch.Tensor(process(gen_image, mu=bias, sigma=scale)[None, ...]).to('cuda')
            activation = adj_model(img).data.cpu().numpy()[0]

        cont, vals, lim_contrast = contrast_tuning(adj_model, mei, bias, scale)

        key['n_seeds'] = len(models)
        key['mei'] = mei
        key['activation'] = activation
        key['monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['max_activation'] = np.max(vals)
        key['max_contrast'] = cont[np.argmax(vals)]
        key['sat_contrast'] = np.max(cont)
        key['img_mean'] = mei.mean()
        key['lim_contrast'] = lim_contrast

        self.insert1(key)


@schema
class ImageConfig(dj.Lookup):
    definition = """
    img_config_id: int
    ---
    img_mean: float   # image mean to use. -1 would use original image mean.
    img_contrast: float   # image contrast to use. -1 would use original image contrast.
    force_stats: bool     # whether to make forcible adjustment on the stats
    """
    contents = [
        (0, 111.0, 16.0, True),
       # (1, 112.0, 6.0, False),
    ]



@schema
class MEIActivation(dj.Computed):
    definition = """
    -> MEI
    -> ImageConfig
    ---
    mei_activation: float   # activation on mei
    mei_clipped: bool       # whether image was clipped
    mei_contrast: float     # actual contrast of mei
    """

    def make(self, key):
        mei = (MEI() & key).fetch1('mei')

        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast', 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)


        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))


        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)


        with torch.no_grad():
            img = torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activation = adj_model(img).data.cpu().numpy()[0]

        key['mei_activation'] = activation
        key['mei_clipped'] = bool(clipped)
        key['mei_contrast'] = actual_contrast

        self.insert1(key)


@schema
class JitterConfig(dj.Lookup):
    definition = """
    jitter_config_id:   int
    ---
    jitter_size:   int
    """
    contents = [(0, 5)]


@schema
class JitterInPlace(dj.Computed):
    definition = """
    -> MEI
    -> ImageConfig
    -> JitterConfig
    ---
    jitter_activations: longblob      # activation resulting from jitter
    """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        mei = (MEI() & key).fetch1('mei')

        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)

        jitter_size = int((JitterConfig & key).fetch1('jitter_size'))

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        shift = list(enumerate(range(-jitter_size, jitter_size+1)))
        activations = np.empty((len(shift), len(shift)))

        with torch.no_grad():
            img = torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')

            for (iy, jitter_y), (ix, jitter_x) in product(shift, shift):
                jitter_y, jitter_x = int(jitter_y), int(jitter_x)
                jittered_img = roll(roll(img, jitter_y, -2), jitter_x, -1)
                activations[iy, ix] = adj_model(jittered_img).data.cpu().numpy()[0]

        key['jitter_activations'] = activations

        self.insert1(key)


@schema
class StartingImage(dj.Manual):
    definition = """
    image_class: varchar(32)   # class of image
    image_id: int              # unique id under a class
    ---
    image: longblob            # actual image
    tier: varchar(16)          # tier (e.g. train, test, valid)
    normalized: tinyint           # whether the image is pre-normalized
    description='': varchar(128) # additional description of the image
    """

@schema
class ImageStatistics(dj.Computed):
    """
    Image statistics in pixel values and in luminance based on interpolated values.
    """
    definition = """
    -> TargetDataset
    ---
    img_mean: float
    img_std: float
    lum_mean: float
    lum_std: float
    """

    def make(self, key):
        target_scan = (StaticMultiDataset.Member & key).fetch1('KEY')
        f, *_ = ClosestCalibration().get_interp(target_scan)
        images = (Frame * ConditionTier & target_scan).fetch('frame')
        images = np.stack(images)
        key['img_mean'] = images.mean()
        key['img_std'] = images.std(axis=(1, 2)).mean()
        key['lum_mean'] = f(images).mean()
        key['lum_std'] = f(images).std(axis=(1, 2)).mean()

        self.insert1(key)

@schema
class GammaImageStatistics(dj.Computed):
    """
    Image statistics in pixel values and in luminance based on fitted Gamma curves.
    """
    definition = """
    -> TargetDataset
    ---
    img_mean: float
    img_std: float
    lum_mean: float
    lum_std: float
    """

    def make(self, key):
        target_scan = (StaticMultiDataset.Member & key).fetch1('KEY')
        f, *_ = ClosestCalibration().get_fs(target_scan)
        images = (Frame * ConditionTier & target_scan).fetch('frame')
        images = np.stack(images)
        key['img_mean'] = images.mean()
        key['img_std'] = images.std(axis=(1, 2)).mean()
        key['lum_mean'] = f(images).mean()
        key['lum_std'] = f(images).std(axis=(1, 2)).mean()

        self.insert1(key)


@schema
class ImageGradResponse(dj.Computed):
    """
    Gradients at the bunch of natural images
    """
    definition = """
    -> TargetModel
    -> TargetDataset.Unit
    ---
    img_rfs:     external-data    # gradient at the image
    img_activations: longblob     # activation of the model at the image
    """

    @staticmethod
    def init_rf_image(stimulus_shape=(1, 36, 64)):
        return torch.zeros(1, *stimulus_shape, device='cuda', requires_grad=True)

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        target_scan = (StaticMultiDataset.Member & key).fetch1('KEY')
        images, tiers = (Frame * ConditionTier & target_scan).fetch('frame', 'tier')

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # compute gradient and activations for each image

        img_rfs = []
        img_activations = []
        for image in tqdm(images):
            image = np.atleast_3d(image) # ensure channel dimension exist
            image = torch.tensor(process(image, mu=bias, sigma=scale)[None, ...], dtype=torch.float32, requires_grad=True, device='cuda')
            # --- Compute gradient receptive field at the image
            y = adj_model(image)
            y.backward()
            img_rfs.append(image.grad.data.cpu().numpy().squeeze())
            img_activations.append(y.item())

        img_rfs = np.stack(img_rfs)

        key['img_rfs'] = img_rfs
        key['img_activations'] = img_activations
        self.insert1(key)


@schema
class Top100Eigen(dj.Computed):
    definition = """
    -> ImageGradResponse
    ---
    eigen_values: longblob  # eigen values
    """

    def make(self, key):
        z = (ImageGradResponse() & key).fetch1('img_rfs')
        z2 = z.reshape(z.shape[0], -1)
        cov = np.cov(z2.T)
        vals = np.real(eigvals(cov))[:100]
        key['eigen_values'] = vals
        self.insert1(key)


@schema
class AllEigen(dj.Computed):
    definition = """
        -> ImageGradResponse
        ---
        eigen_values: longblob  # eigen values
        """

    def make(self, key):
        z = (ImageGradResponse() & key).fetch1('img_rfs')
        z2 = z.reshape(z.shape[0], -1)
        cov = np.cov(z2.T)
        vals = np.real(eigvals(cov))
        key['eigen_values'] = vals
        self.insert1(key)


@schema
class GradRF(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    point_rf            : longblob  # single gradient RF
    rf                  : longblob  # most exciting images
    activation          : float     # activation at the MEI
    monotonic           : bool      # does activity increase monotonically with contrast
    max_activation      : float     # activation at the maximum contrast
    max_contrast        : float     # contrast at which maximum activity is archived
    sat_contrast        : float     # contrast at which image would start saturating
    """

    key_source =  TargetModel() * MEIParameter() * TargetDataset.Unit & NetworkConfig.CorePlusReadout

    @staticmethod
    def init_rf_image(stimulus_shape=(1, 36, 64)):
        return torch.zeros(1, *stimulus_shape, device='cuda', requires_grad=True)


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)


        # --- Compute gradient receptive field
        X = self.init_rf_image(img_shape[1:])
        y = adj_model(X)
        y.backward()
        point_rf = X.grad.data.cpu().numpy().squeeze()
        rf = X.grad.data

        def linear_model(x):
            return (x * rf).sum()

        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])
        norm = float(params['norm'])
        train_norm = float(params['train_norm'])

        octaves = [
         {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size':float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)

        # generate class visualization via octavewise gradient ascent
        gen_image = deepdraw(linear_model, gen_image, octaves, clip=True,
                             random_crop=False, blur=blur, jitter=jitter,
                             precond=precond, step_gain=step_gain,
                             bias=bias, scale=scale, norm=norm, train_norm=train_norm)

        with torch.no_grad():
            img = torch.Tensor(process(gen_image, mu=bias, sigma=scale)[None, ...]).to('cuda')
            activation = adj_model(img).data.cpu().numpy()[0]

        rf = gen_image.squeeze()
        cont, vals, lim_contrast = contrast_tuning(adj_model, rf, bias, scale)
        key['point_rf'] = point_rf
        key['monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['max_activation'] = np.max(vals)
        key['max_contrast'] = cont[np.argmax(vals)]
        key['sat_contrast'] = np.max(cont)
        key['rf'] = rf
        key['activation'] = activation
        self.insert1(key)


@schema
class RFJitterInPlace(dj.Computed):
    definition = """
    -> GradRF
    -> ImageConfig
    -> JitterConfig
    ---
    rf_jitter_activations: longblob      # activation resulting from jitter
    """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        rf = (GradRF() & key).fetch1('rf')

        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        rf, clipped, actual_contrast = adjust_contrast(rf, target_contrast, mu=target_mean, force=force_stats)

        jitter_size = int((JitterConfig & key).fetch1('jitter_size'))


        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)


        shift = list(enumerate(range(-jitter_size, jitter_size+1)))
        activations = np.empty((len(shift), len(shift)))

        with torch.no_grad():
            img = torch.Tensor(process(rf[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')

            for (iy, jitter_y), (ix, jitter_x) in product(shift, shift):
                jitter_y, jitter_x = int(jitter_y), int(jitter_x)
                jittered_img = roll(roll(img, jitter_y, -2), jitter_x, -1)
                activations[iy, ix] = adj_model(jittered_img).data.cpu().numpy()[0]

        key['rf_jitter_activations'] = activations

        self.insert1(key)


@schema
class ImageShifts(dj.Lookup):
    definition = """
    x_shift: int    # shift in the width dimension
    y_shift: int    # shift in the hieght dimension
    """
    contents = product([-1, 0, 1], [-1, 0, 1])

@schema
class ShiftedRF(dj.Computed):
    definition = """
       -> GradRF
       -> ImageConfig
       -> ImageShifts
       ---
       shifted_rf_activation: float      # activation resulting from shift
       shifted_rf: longblob               # copy of the shifted RF
       """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        rf = (GradRF() & key).fetch1('rf')

        # adjust the contrast and mean luminance of the image
        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        rf, clipped, actual_contrast = adjust_contrast(rf, target_contrast, mu=target_mean, force=force_stats)


        # shift the image
        x_shift, y_shift = key['x_shift'], key['y_shift']
        shifted_rf = np.roll(np.roll(rf, x_shift, 1), y_shift, 0)


        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(shifted_rf[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['shifted_rf_activation'] = activations
        key['shifted_rf'] = shifted_rf

        self.insert1(key)

@schema
class ShiftedMEI(dj.Computed):
    definition = """
       -> MEI
       -> ImageConfig
       -> ImageShifts
       ---
       shifted_mei_activation: float      # activation resulting from shift
       shifted_mei: longblob               # copy of the shifted RF
       """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        mei = (MEI() & key).fetch1('mei')

        # adjust the contrast and mean luminance of the image
        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)

        # shift the image
        x_shift, y_shift = key['x_shift'], key['y_shift']
        shifted_mei = np.roll(np.roll(mei, x_shift, 1), y_shift, 0)

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(shifted_mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['shifted_mei_activation'] = activations
        key['shifted_mei'] = shifted_mei

        self.insert1(key)


@schema
class ProcessedImageConfig(dj.Lookup):
    """
    Image configuration to be used by the processed image. Unlike ImageConfig, here
    the means and contrasts are given in actual luminance measure (cd/m^2).
    """
    definition = """
    img_config_id: int
    ---
    img_mean: float   # image mean to use. -1 would use original image mean.
    img_contrast: float   # image contrast to use. -1 would use original image contrast.
    zoom_factor: float    # factor by which image should be upscaled
    force_stats: bool     # whether to make forcible adjustment on the stats
    """
    contents = [
        (0, 10.0, 2.0, 4, True),
        (1, 16.0, 1.5, 4, True),
        (2, 13.0, 1.8, 4, True),
        (3, 16.0, 2.0, 4, True),
        (4, 10.0, 1.5, 4, True),
        (5, 10.0, 1.2, 4, True),
    ]

@schema
class ProcessedMaskedMEI(dj.Computed):
    definition = """
    -> MEIMask
    -> ProcessedImageConfig
    ---
    masked_mei_activation:  float    # activation on masked MEI
    masked_mei: longblob             # masked MEI
    """
    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']

        # get processing parameters
        target_mean, target_contrast, zoom_factor, force_stats = (ProcessedImageConfig() & key).fetch1('img_mean',
                                                                                              'img_contrast',
                                                                                              'zoom_factor',
                                                                                              'force_stats')


        # get MEI as was generated
        mei, mask = (MEI * MEIMask & key).fetch1('mei', 'mei_mask')

        original_shape = mei.shape

        # here mask is zoomed up separately to make condition equivalent on the natural image masking
        mask = ndimage.zoom(mask, zoom=zoom_factor, mode='reflect')


        # upscale the image
        mei = ndimage.zoom(mei, zoom=zoom_factor, mode='reflect')
        mei = np.clip(mei, 0, 255)

        # perform gamma transformation on the image to put into luminance
        f, f_inv = ClosestCalibration().get_interp(TargetDataset.Unit & key)

        lum_min, lum_max = f(0), f(255)
        lum_mei = f(mei)


        # adjust the contrast
        lum_masked_mei, clipped, actual_contrast = adjust_contrast_with_mask(lum_mei, mask, contrast=target_contrast,
                                                                             mu=target_mean, img_min=lum_min,
                                                                             img_max=lum_max, force=force_stats, verbose=True)

        # invert gamma transformation into image space
        masked_mei = np.clip(f_inv(lum_masked_mei), 0, 255)


        small_masked_mei = cv2.resize(masked_mei, original_shape, interpolation=cv2.INTER_AREA).astype(np.float32)


        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(small_masked_mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['masked_mei_activation'] = activations
        key['masked_mei'] = masked_mei

        self.insert1(key)

@schema
class ProcessedTightMaskedMEI(dj.Computed):
    definition = """
    -> TightMEIMask
    -> ProcessedImageConfig
    ---
    masked_mei_activation:  float    # activation on masked MEI
    masked_mei: longblob             # masked MEI
    masked_mei_mu: float             # mean luminance 
    masked_mei_contrast: float       # contrast of mei image
    """
    def make(self, key, verbose=False):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']

        # get processing parameters
        target_mean, target_contrast, zoom_factor, force_stats = (ProcessedImageConfig() & key).fetch1('img_mean',
                                                                                              'img_contrast',
                                                                                              'zoom_factor',
                                                                                              'force_stats')


        # get MEI as was generated
        mei, mask = (MEI * TightMEIMask & key).fetch1('mei', 'tight_mask')

        original_shape = mei.shape

        # here mask is zoomed up separately to make condition equivalent on the natural image masking
        mask = ndimage.zoom(mask, zoom=zoom_factor, mode='reflect')


        # upscale the image
        mei = ndimage.zoom(mei, zoom=zoom_factor, mode='reflect')
        mei = np.clip(mei, 0, 255)

        # perform gamma transformation on the image to put into luminance
        f, f_inv = ClosestCalibration().get_interp(TargetDataset.Unit & key)

        lum_min, lum_max = f(0), f(255)
        lum_mei = f(mei)




        lum_masked_mei, _ = adjust_img_stats(lum_mei, mu=target_mean, sigma=target_contrast, mask=mask, img_min=lum_min,
                                            img_max=lum_max, verbose=verbose)

        key['masked_mei_mu'] = lum_masked_mei.mean()
        key['masked_mei_contrast'] = lum_masked_mei.std()

        # invert gamma transformation into image space
        masked_mei = np.clip(f_inv(lum_masked_mei), 0, 255)


        small_masked_mei = cv2.resize(masked_mei, original_shape, interpolation=cv2.INTER_AREA).astype(np.float32)


        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(small_masked_mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['masked_mei_activation'] = activations
        key['masked_mei'] = masked_mei

        self.insert1(key)


@schema
class ProcessedTightMaskedImage(dj.Computed):
    definition = """
    -> BestTightMaskedImageNet
    -> ProcessedImageConfig
    ---
    masked_imagenet_activation:  float    # activation on masked imagenet
    masked_imagenet: longblob             # masked imagenet image
    masked_imagenet_mu: float             # mean luminance 
    masked_imagenet_contrast: float       # contrast of imagenet image
    """


    def make(self, key, verbose=False):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']

        # get processing parameters
        target_mean, target_contrast, zoom_factor, force_stats = (ProcessedImageConfig() & key).fetch1('img_mean',
                                                                                              'img_contrast',
                                                                                              'zoom_factor',
                                                                                              'force_stats')
        # get tight mask
        mask = (TightMEIMask & key).fetch1('tight_mask')

        original_shape = mask.shape

        mask = ndimage.zoom(mask, zoom=zoom_factor, mode='reflect')

        img = (stimulus.StaticImage.Image & (BestTightMaskedImageNet & key)).fetch1('image').astype('float')
        img = np.clip(img, 0, 255)



        # perform gamma transformation on the image to put into luminance
        f, f_inv = ClosestCalibration().get_interp(TargetDataset.Unit & key)

        lum_min, lum_max = f(0), f(255)
        lum_img = f(img)

        # adjust the contrast
        lum_masked_img, _ = adjust_img_stats(lum_img, mu=target_mean, sigma=target_contrast, mask=mask, img_min=lum_min,
                                            img_max=lum_max, verbose=verbose)

        key['masked_imagenet_mu'] = lum_masked_img.mean()
        key['masked_imagenet_contrast'] = lum_masked_img.std()

        # invert gamma transformation into image space
        masked_img = np.clip(f_inv(lum_masked_img), 0, 255)


        small_masked_img = cv2.resize(masked_img, original_shape, interpolation=cv2.INTER_AREA).astype(np.float32)


        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # compute the activation on the masked image
        with torch.no_grad():
            img = torch.Tensor(process(small_masked_img[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['masked_imagenet_activation'] = activations
        key['masked_imagenet'] = masked_img

        self.insert1(key)


@schema
class ProcessedTightUnmaskedImage(dj.Computed):
    definition = """
    -> BestTightMaskedImageNet
    -> ProcessedImageConfig
    ---
    unmasked_imagenet_activation:  float    # activation on masked imagenet
    unmasked_imagenet: longblob             # masked imagenet image
    unmasked_imagenet_mu: float             # mean luminance 
    unmasked_imagenet_contrast: float       # contrast of imagenet image
    """


    def make(self, key, verbose=False):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']

        # get processing parameters
        target_mean, target_contrast, zoom_factor, force_stats = (ProcessedImageConfig() & key).fetch1('img_mean',
                                                                                              'img_contrast',
                                                                                              'zoom_factor',
                                                                                              'force_stats')
        # get tight mask
        mask = (TightMEIMask & key).fetch1('tight_mask')

        original_shape = mask.shape

        mask = ndimage.zoom(mask, zoom=zoom_factor, mode='reflect')

        img = (stimulus.StaticImage.Image & (BestTightMaskedImageNet & key)).fetch1('image').astype('float')
        img = np.clip(img, 0, 255)



        # perform gamma transformation on the image to put into luminance
        f, f_inv = ClosestCalibration().get_interp(TargetDataset.Unit & key)

        lum_min, lum_max = f(0), f(255)
        lum_img = f(img)

        # adjust the contrast
        _, lum_unmasked_img = adjust_img_stats(lum_img, mu=target_mean, sigma=target_contrast, mask=mask, img_min=lum_min,
                                            img_max=lum_max, verbose=verbose)

        key['unmasked_imagenet_mu'] = lum_unmasked_img.mean()
        key['unmasked_imagenet_contrast'] = lum_unmasked_img.std()

        # invert gamma transformation into image space
        unmasked_img = np.clip(f_inv(lum_unmasked_img), 0, 255)


        small_unmasked_img = cv2.resize(unmasked_img, original_shape, interpolation=cv2.INTER_AREA).astype(np.float32)


        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # compute the activation on the masked image
        with torch.no_grad():
            img = torch.Tensor(process(small_unmasked_img[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['unmasked_imagenet_activation'] = activations
        key['unmasked_imagenet'] = unmasked_img

        self.insert1(key)



## Due to deprecation of BestMaskedImageNet, this class is also deprecated!
@schema
class ProcessedMaskedImage(dj.Computed):
    definition = """
    -> BestMaskedImageNet
    -> ProcessedImageConfig
    ---
    masked_imagenet_activation:  float    # activation on masked imagenet
    masked_imagenet: longblob             # masked imagenet image
    """


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']

        # get processing parameters
        target_mean, target_contrast, zoom_factor, force_stats = (ProcessedImageConfig() & key).fetch1('img_mean',
                                                                                              'img_contrast',
                                                                                              'zoom_factor',
                                                                                              'force_stats')
        # get Gauss mask
        mask = (GaussianMask & key).fetch1('mei_mask')
        original_shape = mask.shape

        mask = ndimage.zoom(mask, zoom=zoom_factor, mode='reflect')

        img = (stimulus.StaticImage.Image & (BestMaskedImageNet & key)).fetch1('image').astype('float')
        img = np.clip(img, 0, 255)






        # perform gamma transformation on the image to put into luminance
        f, f_inv = ClosestCalibration().get_interp(TargetDataset.Unit & key)

        lum_min, lum_max = f(0), f(255)
        lum_img = f(img)

        # adjust the contrast
        lum_masked_img, clipped, actual_contrast = adjust_contrast_with_mask(lum_img,  mask, contrast=target_contrast,
                                                                             mu=target_mean, img_min=lum_min,
                                                                             img_max=lum_max, force=force_stats, verbose=True)

        # invert gamma transformation into image space
        masked_img = np.clip(f_inv(lum_masked_img), 0, 255)


        small_masked_img = cv2.resize(masked_img, original_shape, interpolation=cv2.INTER_AREA).astype(np.float32)


        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(small_masked_img[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['masked_imagenet_activation'] = activations
        key['masked_imagenet'] = masked_img

        self.insert1(key)


@schema
class ProcessedShiftedMEI(dj.Computed):
    """
    Performs image rescaling and gamma correction before contrast/luminance matching.
    """
    definition = """
       -> MEI
       -> ProcessedImageConfig
       -> ImageShifts
       ---
       shifted_mei_activation: float      # activation resulting from shift
       shifted_mei: longblob               # copy of the shifted RF
       """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']

        # get processing parameters
        target_mean, target_contrast, zoom_factor, force_stats = (ProcessedImageConfig() & key).fetch1('img_mean',
                                                                                              'img_contrast',
                                                                                              'zoom_factor',
                                                                                              'force_stats')


        # get MEI as was generated
        mei = (MEI() & key).fetch1('mei')
        print(mei.max(), mei.min())

        original_shape = mei.shape

        # upscale the image
        mei = ndimage.zoom(mei, zoom=zoom_factor, mode='reflect')
        mei = np.clip(mei, 0, 255)

        # perform gamma transformation on the image to put into luminance
        f, f_inv = ClosestCalibration().get_interp(TargetDataset.Unit & key)

        lum_min, lum_max = f(0), f(255)
        lum_mei = f(mei)

        # adjust the contrast
        lum_mei, clipped, actual_contrast = adjust_contrast(lum_mei, target_contrast, mu=target_mean, img_min=lum_min,
                                                        img_max=lum_max, force=force_stats)


        # invert gamma transformation into image space
        mei = np.clip(f_inv(lum_mei), 0, 255)

        # shift the image
        x_shift, y_shift = int(zoom_factor * key['x_shift']), int(zoom_factor * key['y_shift'])
        shifted_mei = np.roll(np.roll(mei, x_shift, 1), y_shift, 0)

        small_shifted = cv2.resize(shifted_mei, original_shape, interpolation=cv2.INTER_AREA).astype(np.float32)


        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(small_shifted[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['shifted_mei_activation'] = activations
        key['shifted_mei'] = shifted_mei

        self.insert1(key)


# the following version makes use of the new adjust_img_stats and has been used for group_id = 29 and above
@schema
class ProcessedShiftedMEI2(dj.Computed):
    """
    Performs image rescaling and gamma correction before contrast/luminance matching.
    """
    definition = """
       -> MEI
       -> ProcessedImageConfig
       -> ImageShifts
       ---
       shifted_mei_activation: float      # activation resulting from shift
       shifted_mei: longblob               # copy of the shifted RF
       shifted_mei_mu: float              # actual image mean
       shifted_mei_contrast: float        # achieved contrast
       """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']

        # get processing parameters
        target_mean, target_contrast, zoom_factor, force_stats = (ProcessedImageConfig() & key).fetch1('img_mean',
                                                                                              'img_contrast',
                                                                                              'zoom_factor',
                                                                                              'force_stats')


        # get MEI as was generated
        mei = (MEI() & key).fetch1('mei')
        print(mei.max(), mei.min())

        original_shape = mei.shape

        # upscale the image
        mei = ndimage.zoom(mei, zoom=zoom_factor, mode='reflect')
        mei = np.clip(mei, 0, 255)

        # perform gamma transformation on the image to put into luminance
        f, f_inv = ClosestCalibration().get_interp(TargetDataset.Unit & key)

        lum_min, lum_max = f(0), f(255)
        lum_mei = f(mei)

        # adjust the contrast
        lum_mei, _ = adjust_img_stats(lum_mei,  mu=target_mean, sigma=target_contrast, img_min=lum_min, img_max=lum_max)

        key['shifted_mei_mu'] = lum_mei.mean()
        key['shifted_mei_contrast'] = lum_mei.std()

        # invert gamma transformation into image space
        mei = np.clip(f_inv(lum_mei), 0, 255)

        # shift the image
        x_shift, y_shift = int(zoom_factor * key['x_shift']), int(zoom_factor * key['y_shift'])
        shifted_mei = np.roll(np.roll(mei, x_shift, 1), y_shift, 0)

        small_shifted = cv2.resize(shifted_mei, original_shape, interpolation=cv2.INTER_AREA).astype(np.float32)


        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(small_shifted[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['shifted_mei_activation'] = activations
        key['shifted_mei'] = shifted_mei

        self.insert1(key)



@schema
class ModelType(dj.Lookup):
    definition = """
    model_type: varchar(16)   # type of model
    """
    contents = zip(['cnn', 'linear'])

@schema
class MEIConfusion(dj.Computed):
    definition = """
    -> MEIParameter
    -> TargetDataset.Unit
    -> TargetDataset.Unit.proj(src_neuron_id='neuron_id')
    -> ModelType
    -> ImageConfig
    ---
    confusion_activation: float   # activation of the target unit 
    """

    model_map = {
        'cnn': ModelGroup.CNNModel,
        'linear': ModelGroup.LinearModel
    }

    @property
    def key_source(self):
        units = TargetDataset.Unit & (CorrectedHighUnitSelection & 'hu_rank < 150')
        mei_param = dict(mei_param_id='150966b4691c643986d738a5f6656594')
        return units.proj(src_neuron_id='neuron_id') * MEIParameter * ImageConfig * ModelType & MEI.proj(
            src_neuron_id='neuron_id') & mei_param

    def make(self, key):
        target_neurons = (TargetDataset.Unit & (CorrectedHighUnitSelection & 'hu_rank < 150') & key).fetch('neuron_id')

        readout_key = key['readout_key']
        print('Confusion computation: Working on readout_key={}'.format(readout_key))

        # target model is always CNN
        model_key = (TargetModel & ModelGroup.CNNModel & key).fetch1('KEY')

        src_key = dict(key)
        src_key['neuron_id'] = src_key.pop('src_neuron_id')

        mei = (MEI() & self.model_map[key['model_type']] & src_key).fetch1('mei')

        # adjust the contrast and mean luminance of the image
        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(model_key)
        adj_model = get_adj_model(models, readout_key, target_neurons, mu_eye=mu_eye)

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]


        keys = [dict(neuron_id=nid, confusion_activation=activation, **key) for nid, activation in zip(target_neurons, activations)]
        self.insert(keys)


@schema
class MismatchedMEIConfusion(dj.Computed):
    """
    Using mean as std to capture the "mismatch" found during the training. Note that this still uses MEI that was
    generated "correctly" - using the correct image standard deviation during the generation procedure. For loop 1&2,
    this means that the MEIs were actually using values that were outside the possible image intensities.
    During this process here, the MEI confusion computation "correctly" accounts for the incorrect usage of the
    mean as standard deviation.
    """
    definition = """
    -> MEIParameter
    -> TargetDataset.Unit
    -> TargetDataset.Unit.proj(src_neuron_id='neuron_id')
    -> ModelType
    -> ImageConfig
    ---
    confusion_activation: float   # activation of the target unit 
    """

    model_map = {
        'cnn': ModelGroup.CNNModel,
        'linear': ModelGroup.LinearModel
    }

    @property
    def key_source(self):
        units = TargetDataset.Unit & (CorrectedHighUnitSelection & 'hu_rank < 150')
        mei_param = dict(mei_param_id='150966b4691c643986d738a5f6656594')
        return units.proj(src_neuron_id='neuron_id') * MEIParameter * ImageConfig * ModelType & MEI.proj(
            src_neuron_id='neuron_id') & mei_param

    def make(self, key):
        target_neurons = (TargetDataset.Unit & (CorrectedHighUnitSelection & 'hu_rank < 150') & key).fetch('neuron_id')

        readout_key = key['readout_key']
        print('Confusion computation: Working on readout_key={}'.format(readout_key))

        # target model is always CNN
        model_key = (TargetModel & ModelGroup.CNNModel & key).fetch1('KEY')

        src_key = dict(key)
        src_key['neuron_id'] = src_key.pop('src_neuron_id')

        mei = (MEI() & self.model_map[key['model_type']] & src_key).fetch1('mei')

        # adjust the contrast and mean luminance of the image
        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        ## !!!! Intentionally wrong step !!!!! ##
        scale = bias

        # load the model
        models = get_multi_model(model_key)
        adj_model = get_adj_model(models, readout_key, target_neurons, mu_eye=mu_eye)

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]


        keys = [dict(neuron_id=nid, confusion_activation=activation, **key) for nid, activation in zip(target_neurons, activations)]
        self.insert(keys)


@schema
class WrongMEIConfusion(dj.Computed):
    definition = """
    -> MEIParameter
    -> TargetDataset.Unit
    -> TargetDataset.Unit.proj(src_neuron_id='neuron_id')
    -> ModelType
    -> ImageConfig
    ---
    confusion_activation: float   # activation of the target unit 
    """

    model_map = {
        'cnn': ModelGroup.CNNModel,
        'linear': ModelGroup.LinearModel
    }

    @property
    def key_source(self):
        units = TargetDataset.Unit & (CorrectedHighUnitSelection & 'hu_rank < 150')
        mei_param = dict(mei_param_id='150966b4691c643986d738a5f6656594')
        return units.proj(src_neuron_id='neuron_id') * MEIParameter * ImageConfig * ModelType & WrongMEI.proj(
            src_neuron_id='neuron_id') & mei_param

    def make(self, key):
        target_neurons = (TargetDataset.Unit & (CorrectedHighUnitSelection & 'hu_rank < 150') & key).fetch('neuron_id')

        readout_key = key['readout_key']
        print('Confusion computation: Working on readout_key={}'.format(readout_key))

        # target model is always CNN
        model_key = (TargetModel & ModelGroup.CNNModel & key).fetch1('KEY')

        src_key = dict(key)
        src_key['neuron_id'] = src_key.pop('src_neuron_id')

        mei = (WrongMEI() & self.model_map[key['model_type']] & src_key).fetch1('mei')

        # adjust the contrast and mean luminance of the image
        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        ## !!!! Intentionally wrong step !!!!! ##
        scale = bias

        # load the model
        models = get_multi_model(model_key)
        adj_model = get_adj_model(models, readout_key, target_neurons, mu_eye=mu_eye)

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]


        keys = [dict(neuron_id=nid, confusion_activation=activation, **key) for nid, activation in zip(target_neurons, activations)]
        self.insert(keys)





@schema
class CrossShiftedMEIActivation(dj.Computed):
    definition = """
    -> MEI
    -> ImageConfig
    -> ImageShifts
    ---
    cross_shifted_mei_activation: float   # activation of CNN from shifted Lin model MEI
    """
    key_source = MEI * ImageConfig.proj() * ImageShifts & ModelGroup.CNNModel

    def make(self, key):
        # readout_key = key['readout_key']
        # neuron_id = key['neuron_id']
        # print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))
        #
        # mei = (MEI() & key).fetch1('mei')

        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        lin_key = (ModelGroup.LinearModel & (ModelGroup & (ModelGroup.CNNModel & key))).fetch1('KEY')

        key_lin = dict(key, **lin_key)

        mei = (MEI() & key_lin).fetch1('mei')

        # adjust the contrast and mean luminance of the image
        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)

        # shift the image
        x_shift, y_shift = key['x_shift'], key['y_shift']
        shifted_mei = np.roll(np.roll(mei, x_shift, 1), y_shift, 0)

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(shifted_mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['cross_shifted_mei_activation'] = activations

        self.insert1(key)


@schema
class CrossJitterInPlace(dj.Computed):
    definition = """
    -> MEI
    -> ImageConfig
    -> JitterConfig
    ---
    cross_jitter_activations: longblob      # activation resulting from jitter
    """

    key_source = MEI * ImageConfig.proj() * JitterConfig & ModelGroup.CNNModel()

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        lin_key = (ModelGroup.LinearModel & (ModelGroup & (ModelGroup.CNNModel & key))).fetch1('KEY')

        key_lin = dict(key, **lin_key)

        mei = (MEI() & key_lin).fetch1('mei')

        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)

        jitter_size = int((JitterConfig & key).fetch1('jitter_size'))

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)


        shift = list(enumerate(range(-jitter_size, jitter_size+1)))
        activations = np.empty((len(shift), len(shift)))

        with torch.no_grad():
            img = torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')

            for (iy, jitter_y), (ix, jitter_x) in product(shift, shift):
                jitter_y, jitter_x = int(jitter_y), int(jitter_x)
                jittered_img = roll(roll(img, jitter_y, -2), jitter_x, -1)
                activations[iy, ix] = adj_model(jittered_img).data.cpu().numpy()[0]

        key['cross_jitter_activations'] = activations

        self.insert1(key)






@schema
class SpotCenteredMEI(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> ImageConfig
    -> TargetDataset.Unit
    ---
    ctr_n_seeds             : int       # number of distinct seeded models used
    ctr_mei                 : longblob  # most exciting images
    ctr_activation          : float     # activation at the MEI
    """

    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return TargetModel() * MEIParameter() * ImageConfig() * TargetDataset.Unit & NetworkConfig.CorePlusReadout


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']

        # adjust the contrast and mean luminance of the image
        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))



        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        print('Working with images with mu={}, sigma={}'.format(bias, scale))

        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)


        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])
        norm = float(params['norm'])
        train_norm = float(params['train_norm'])

        octaves = [
            {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size': float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)

        with SpatialTransformerPyramid2d.center_readout():
            # generate class visualization via octavewise gradient ascent
            gen_image = deepdraw(adj_model, gen_image, octaves, clip=True,
                                 random_crop=False, blur=blur, jitter=jitter,
                                 precond=precond, step_gain=step_gain,
                                 bias=bias, scale=scale, norm=norm, train_norm=train_norm)

            mei = gen_image.squeeze()
            mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)

            with torch.no_grad():
                img = torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
                activation = adj_model(img).data.cpu().numpy()[0]


        key['ctr_n_seeds'] = len(models)
        key['ctr_mei'] = mei
        key['ctr_activation'] = activation

        self.insert1(key)


@schema
class ScaledCenteredMEI(dj.Computed):
    definition = """
    -> SpotCenteredMEI
    ---
    scaled_mei: longblob   # scaled centered MEI
    """

    def make(self, key):
        img = (SpotCenteredMEI & key).fetch1('ctr_mei')

        img = ndimage.zoom(img, zoom=4, mode='reflect')
        img = np.clip(img, 0, 255).astype('uint8')
        key['scaled_mei'] = img
        self.insert1(key)


@schema
class GaborRange(dj.Lookup):
    definition = """ # lists of gabor parameters to search over for the best gabor

    gaborrange_id:  int     # id for each range
    ---
    height:         int         # (px) image height
    width:          int         # (px) image width
    phases:         longblob    # (degree) angle at which to start the sinusoid
    wavelengths:    longblob    # (px) wavelength of the sinusoid (1 / spatial frequency)
    orientations:   longblob    # (degree) counterclockwise rotation to apply (0 is horizontal, 90 vertical)
    sigmas:         longblob    # (px) sigma of the gaussian mask used
    dys:            longblob    # (px/height) amount of translation in y (positive moves downwards)
    dxs:            longblob    # (px/width) amount of translation in x (positive moves right)
    """
    contents = [
        [1, 36, 64, [0, 90, 180, 270], [4, 7, 10, 15, 20], np.linspace(0, 180, 8, endpoint=False),
         [2, 3, 5, 7, 9], np.linspace(-0.3, 0.3, 7), np.linspace(-0.3, 0.3, 13)],
    ]


class GaborDataset(Dataset):
    """ Dataset that generates Gabors given some parameters with an specific mean and
    contrast.

    Arguments:
        height, width, phases, wavelengths, orientations, sigmas, dys, dxs: Parameters to
            generate a gabor image. See staticnet_analyses.utils.create_gabor()
        target_mean, target_contrast, force_stats: Desired mean and contrast of the output
            gabor.
    """

    def __init__(self, height, width, phases, wavelengths, orientations, sigmas, dys,
                 dxs, target_mean, target_contrast, force_stats, f=None, f_inv=None):
        if f is None or f_inv is None:
            f = f_inv = lambda x: x

        self.arg_names = ['height', 'width', 'phase', 'wavelength', 'orientation',
                          'sigma', 'dy', 'dx']
        self.args = list(product([height], [width], phases, wavelengths, orientations,
                                 sigmas, dys, dxs))

        # Gabors will have this mean and std (constrained to be in [0, 255] range
        self.target_mean = target_mean
        self.target_contrast = target_contrast
        self.force_stats = force_stats
        self.f = f
        self.f_inv = f_inv
        self.f_min = f(0)
        self.f_max = f(255)

    def __getitem__(self, item):
        gabor = create_gabor(**{k: v for k, v in zip(self.arg_names, self.args[item])})

        # Match the target mean and contrast
        # gabor = (gabor - gabor.mean()) / gabor.std()
        # generate gabor in the luminance space
        gabor, _ = adjust_img_stats(gabor, mu=self.target_mean, sigma=self.target_contrast, img_min=self.f_min,
                         img_max=self.f_max)
        # gabor = adjust_contrast(self.f(gabor), self.target_contrast, mu=self.target_mean,
        #                         force=self.force_stats, steps=self.steps, img_min=self.f_min, img_max=self.f_max)[0]

        # invert it back into the original space
        gabor = self.f_inv(gabor)

        return gabor.astype(np.float32)

    def __len__(self):
        return len(self.args)


@schema
class BestGabor(dj.Computed):
    definition = """ # find the most exciting gabor for each cell in this dataset
    -> TargetModel
    -> ProcessedImageConfig
    -> GaborRange
    -> TargetDataset
    ---
    num_gabors:     int             # number of gabors tried
    """

    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return TargetModel * ProcessedImageConfig * GaborRange * TargetDataset & NetworkConfig.CorePlusReadout

    class Unit(dj.Part):
        definition = """ # best gabor parameters per unit

        -> master
        -> TargetDataset.Unit
        ---
        best_activation:    float
        best_phase:         float
        best_wavelength:    float
        best_orientation:   float
        best_sigma:         float
        best_dy:            float
        best_dx:            float
        """

    def make(self, key):
        # Get gabor dataset
        gabor_params = (GaborRange & key).fetch1('height', 'width', 'phases',
                                                 'wavelengths', 'orientations', 'sigmas',
                                                 'dys', 'dxs')
        contrast_params = (ProcessedImageConfig & key).fetch1('img_mean', 'img_contrast',
                                                              'force_stats')

        # perform gamma transformation on the image to put into luminance
        f, f_inv = ClosestCalibration().get_interp(TargetDataset.Unit & key)
        dataset = GaborDataset(*gabor_params, *contrast_params, f=f, f_inv=f_inv)
        dataloader = DataLoader(dataset, batch_size=128, num_workers=11, shuffle=False)

        # Get model
        models = get_multi_model(key)

        # Iterate over readout keys (usually only one)
        readout_keys = (dj.U('readout_key') & (TargetDataset.Unit & key)).fetch('readout_key')
        for readout_key in readout_keys:
            # Get input statistics
            _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

            adj_model = get_adj_model(models, readout_key, neuron_id=slice(None),  # all cells
                                      mu_eye=mu_eye)

            # Evaluate all gabors
            activations = []
            with torch.no_grad():
                for i, gabors in tqdm(enumerate(dataloader)):
                    # norm = gabors
                    norm = (gabors - bias) / scale
                    img = torch.Tensor(norm[:, None, :, :]).to('cuda')
                    img_activations = adj_model(img).cpu().numpy()

                    activations.append(img_activations)
            activations = np.concatenate(activations)  # num_gabors x num_cells

            # Check we got all gabors and all cells
            if len(activations) != len(dataset):
                raise ValueError('Some gabor patches did not get processed')
            if activations.shape[1] != len(TargetDataset.Unit & key &
                                           {'readout_key': readout_key}):
                raise ValueError('Not all cells got processed')

            # Iterate best gabor per cell
            print('Inserting')
            self.insert1({**key, 'num_gabors': len(dataset)}, skip_duplicates=True)
            for neuron_id, neuron_activations in enumerate(activations.T):
                # Select best gabor
                best_idx = np.argmax(neuron_activations)
                best_activation = neuron_activations[best_idx]
                (best_phase, best_wavelength, best_orientation, best_sigma, best_dy,
                 best_dx) = dataset.args[best_idx][2:]

                # Insert
                self.Unit.insert1({**key, 'neuron_id': neuron_id,
                                   'readout_key': readout_key,
                                   'best_activation': best_activation,
                                   'best_phase': best_phase,
                                   'best_wavelength': best_wavelength,
                                   'best_orientation': best_orientation,
                                   'best_sigma': best_sigma, 'best_dy': best_dy,
                                   'best_dx': best_dx})


@schema
class ProcessedBestGabor(dj.Computed):
    definition = """
    -> BestGabor.Unit
    ---
    gabor_activation:  float    # activation on masked imagenet
    gabor: longblob             # masked imagenet image
    gabor_mu: float             # mean luminance 
    gabor_contrast: float       # contrast of imagenet image
    """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']

        # get processing parameters
        target_mean, target_contrast, zoom_factor, force_stats = (ProcessedImageConfig() & key).fetch1('img_mean',
                                                                                                       'img_contrast',
                                                                                                       'zoom_factor',
                                                                                                       'force_stats')

        # get best gabor parameters
        height, width = (GaborRange & key).fetch1('height', 'width')
        arg_names = ['phase', 'wavelength', 'orientation', 'sigma', 'dy', 'dx']
        attr_names = ['best_{}'.format(f) for f in arg_names]

        gabor_params = (BestGabor.Unit & key).fetch1(*attr_names)
        gabor = create_gabor(height, width, *gabor_params)
        original_shape = gabor.shape

        # upscale the image
        lum_gabor = ndimage.zoom(gabor, zoom=zoom_factor, mode='reflect')

        # perform gamma transformation on the image to put into luminance
        f, f_inv = ClosestCalibration().get_interp(TargetDataset.Unit & key)

        lum_min, lum_max = f(0), f(255)

        lum_gabor, _ = adjust_img_stats(lum_gabor, mu=target_mean, sigma=target_contrast, img_min=lum_min,
                                     img_max=lum_max, verbose=True)

        key['gabor_mu'] = lum_gabor.mean()
        key['gabor_contrast'] = lum_gabor.std()

        # invert gamma transformation into image space
        gabor = np.clip(f_inv(lum_gabor), 0, 255)

        small_gabor = cv2.resize(gabor, original_shape, interpolation=cv2.INTER_AREA).astype(np.float32)

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        # load the model
        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(small_gabor[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['gabor_activation'] = activations
        key['gabor'] = gabor

        self.insert1(key)


@schema
class GaborLimits(dj.Lookup):
    definition = """ # limits of some parameters search range to find the optimal gabor

    gaborlimits_id:     int         # id of this search range
    ---
    height:             int         # (px) height of image 
    width:              int         # (px) width of image
    lower_phase:        float
    upper_phase:       float
    lower_wavelength:   float
    upper_wavelength:   float
    lower_orientation:  float
    upper_orientation:  float
    lower_sigma:        float
    upper_sigma:        float
    lower_dy:           float
    upper_dy:           float
    lower_dx:           float
    upper_dx:           float
    """
    contents = [[1, 36, 64, 0, 360, 4, 20, 0, 180, 2, 9, -0.35, 0.35, -0.35, 0.35], ]


@schema
class OptimalGabor(dj.Computed):
    definition = """ # find parameters that produce an optimal gabor for this unit

    -> TargetModel
    -> ProcessedImageConfig
    -> TargetDataset.Unit
    -> GaborLimits
    ---
    best_gabor:         longblob    # best gabor image
    best_seed:          int         # random seed used to obtain the best gabor
    best_activation:    float       # activation at the best gabor image    
    best_phase:         float       # (degree) angle at which to start the sinusoid
    best_wavelength:    float       # (px) wavelength of the sinusoid (1 / spatial frequency)
    best_orientation:   float       # (degree) counterclockwise rotation to apply (0 is horizontal, 90 vertical)
    best_sigma:         float       # (px) sigma of the gaussian mask used
    best_dy:            float       # (px/height) amount of translation in y (positive moves downwards)
    best_dx:            float       # (px/width) amount of translation in x (positive moves right)
    """

    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return (TargetModel * ProcessedImageConfig.proj() * GaborLimits.proj() *
                TargetDataset.Unit & NetworkConfig.CorePlusReadout)

    def make(self, key):
        from scipy import optimize

        # Get some preprocessing params
        target_mean, target_contrast, force_stats = (ProcessedImageConfig & key).fetch1(
            'img_mean', 'img_contrast',  'force_stats')
        f, f_inv = (ClosestCalibration & (TargetDataset.Unit & key)).get_interp()

        # Get model
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, key['readout_key'])
        models = get_multi_model(key)
        adj_model = get_adj_model(models, key['readout_key'],
                                            neuron_id=key['neuron_id'], mu_eye=mu_eye)

        # Get optimization bounds per parameter
        params = (GaborLimits & key).fetch1()
        bounds = [(params['lower_phase'], params['upper_phase']),
                  (params['lower_wavelength'], params['upper_wavelength']),
                  (params['lower_orientation'], params['upper_orientation']),
                  (params['lower_sigma'], params['upper_sigma']),
                  (params['lower_dy'], params['upper_dy']),
                  (params['lower_dx'], params['upper_dx'])]

        # Write loss function to be optimized
        def neg_model_activation(params, bounds=bounds, height=params['height'],
                                 width=params['width'], target_mean=target_mean,
                                 target_contrast=target_contrast, train_mean=bias,
                                 train_std=scale, model=adj_model, f=f, finv=f_inv):
            # Get params
            params = [np.clip(p, l, u) for p, (l, u) in zip(params, bounds)]  # *
            phase, wavelength, orientation, sigma, dy, dx = params
            # * some local optimization methods in scipy.optimize receive parameter bounds
            # as arguments, however, empirically they seem to have lower performance than
            # those that do not (like Nelder-Mead which I use below). In general, gradient
            # based methods did worse than direct search ones.

            # Create gabor
            gabor = create_gabor(height=height, width=width, phase=phase,
                                 wavelength=wavelength, orientation=orientation,
                                 sigma=sigma, dy=dy, dx=dx)

            # # Preprocess it (directly in pixels)
            # gabor = multi_mei.adjust_img_stats(gabor, mu=target_mean, sigma=target_contrast)[0]

            # Express gabor in luminance, contrast match and then transform it back to pixels
            gabor = adjust_img_stats(gabor, mu=target_mean, sigma=target_contrast,
                                     img_min=f(0), img_max=f(255))[0]
            gabor = finv(gabor)

            # Compute activation
            with torch.no_grad():
                norm = (gabor - train_mean) / train_std
                img = torch.Tensor(norm[None, None, :, :]).to('cuda')
                activation = model(img).item()

            return -activation

        # Find best parameters (simulated annealing -> local search)
        best_activation = np.inf
        for seed in tqdm([1, 12, 123, 1234, 12345]):  # try 5 diff random seeds
            res = optimize.dual_annealing(neg_model_activation, bounds=bounds,
                                          no_local_search=True, maxiter=300, seed=seed)
            res = optimize.minimize(neg_model_activation, x0=res.x, method='Nelder-Mead')

            if res.fun < best_activation:
                best_activation = res.fun
                best_params = res.x
                best_seed = seed
        best_params = [np.clip(p, l, u) for p, (l, u) in zip(best_params, bounds)]

        # Create best gabor
        best_gabor = create_gabor(height=params['height'], width=params['width'],
                                  phase=best_params[0], wavelength=best_params[1],
                                  orientation=best_params[2], sigma=best_params[3],
                                  dy=best_params[4], dx=best_params[5])
        best_activation = -neg_model_activation(best_params)

        # Insert
        self.insert1({**key, 'best_gabor': best_gabor, 'best_seed': best_seed,
                      'best_activation': best_activation, 'best_phase': best_params[0],
                      'best_wavelength': best_params[1],
                      'best_orientation': best_params[2], 'best_sigma': best_params[3],
                      'best_dy': best_params[4], 'best_dx': best_params[5]})
