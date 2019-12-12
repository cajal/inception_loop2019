#!/usr/bin/env python3

import logging
from staticnet_experiments import models, configs
from staticnet_analyses import multi_mei, mei
from neuro_data.static_images import stats, data_schemas
import datajoint as dj
from neuro_data.static_images.data_schemas import stimulus, InputResponse
from scipy import ndimage
import numpy as np
from pandas import DataFrame
import pandas as pd
import warnings
stack = dj.create_virtual_module('stack', 'pipeline_stack')


## Configuration for the closed loop experiment

from neuro_data.static_images.data_schemas import *

schema = dj.schema('neurostatic_mei_closed_loop')

def fill_data(cond):
    data_schemas.StaticScan().populate(cond, reserve_jobs=True, order='random')
    data_schemas.ConditionTier().populate(cond,  reserve_jobs=True, order='random')
    data_schemas.Frame.populate('preproc_id=0', cond,  reserve_jobs=True, order='random')
    data_schemas.InputResponse().populate(cond, 'preproc_id=0',  reserve_jobs=True, order='random')
    data_schemas.Eye().populate(cond,  reserve_jobs=True, order='random')
    data_schemas.Treadmill().populate(cond,  reserve_jobs=True, order='random')
    data_schemas.StaticMultiDataset().fill()



model_restr = 'gauss_bias=0 and gauss_sigma=0.5'
candidates = [configs.CoreConfig.StackedLinearGaussianLaplace & model_restr & 'input_kern=15 and hidden_kern=7',
              configs.CoreConfig.GaussianLaplace & model_restr & 'input_kern=15 and hidden_kern=7']

model_candidates = (configs.NetworkConfig.CorePlusReadout
                    & (configs.TrainConfig.Default & dict(batch_size=60))
                    & candidates
                    & configs.ShifterConfig.MLP()
                    & configs.ReadoutConfig.SpatialTransformerPyramid2d()
                    & configs.ModulatorConfig.MLP()
                    & (configs.DataConfig.CorrectedAreaLayer() & 'stimulus_type="stimulus.Frame" and exclude="" '
                                                                 'and normalize_per_image=False'))

def get_condition(group_id):
    gid_restr = 'group_id = {}'.format(group_id)
    oracle_computed = stats.Oracle.UnitScores() & gid_restr
    return gid_restr, model_candidates & oracle_computed

def populate_models(group_id, **kwargs):
    models.Model().populate(*get_condition(group_id), **kwargs)


# # Fill stimulus table with MEI images
# def fill_stimulus(group_id):
#     gid_restr = dict(group_id=group_id)
#     restr = dj.AndList([mei.CorrectedHighUnitSelection & 'hu_rank < 150',
#                         'img_config_id=0 and mei_param_id = "443fce0cecfab7fd58963c1140aab57f"'])
#     cnn_meis = (mei.ShiftedMEI & mei.ModelGroup.CNNModel & gid_restr & restr).proj(cnn_hash='net_hash', cnn_seed='seed',
#                                                                                    cnn_mei='shifted_mei')
#     lin_meis = (mei.ShiftedMEI & mei.ModelGroup.LinearModel & gid_restr & restr).proj(lin_hash='net_hash',
#                                                                                       lin_seed='seed',
#                                                                                       lin_mei='shifted_mei')
#
#     combo = cnn_meis * lin_meis
#     n_good_comb = len(combo)
#     assert n_good_comb == len(cnn_meis) == len(lin_meis), 'Entries should match between CNN and Linear!'
#
#     keys = (cnn_meis * lin_meis * mei.CorrectedHighUnitSelection & 'hu_rank < 150').fetch('KEY',
#                                                                                 order_by='hu_rank, x_shift, y_shift')
#
#     for key in keys:
#         if stimulus.StaticImage.MEI() & key & 'image_class="cnn_mei"':
#             print('Entries found. Skipping...')
#             continue
#         else:
#             max_image_id = dj.U().aggr(stimulus.StaticImage.Image() & 'image_class="cnn_mei"',
#                                        cmax='max(image_id)').fetch1('cmax')
#             next_image_id = 0 if max_image_id is None else (max_image_id + 1)
#             print('Creating image ID {}'.format(next_image_id))
#
#             cnn_key, cnn_mei = (mei.ShiftedMEI & key & mei.ModelGroup.CNNModel).fetch1('KEY', 'shifted_mei')
#             lin_key, lin_mei = (mei.ShiftedMEI & key & mei.ModelGroup.LinearModel).fetch1('KEY', 'shifted_mei')
#
#             # upscale the MEIs by a factor of 4
#             cnn_mei = ndimage.zoom(cnn_mei, zoom=4, mode='reflect')
#             cnn_mei = np.clip(cnn_mei, 0, 255).astype('uint8')
#
#             lin_mei = ndimage.zoom(lin_mei, zoom=4, mode='reflect')
#             lin_mei = np.clip(lin_mei, 0, 255).astype('uint8')
#
#             new_mei_key = dict(image_id=next_image_id, image_class='cnn_mei', image=cnn_mei.astype('uint8'))
#             new_rf_key = dict(image_id=next_image_id, image_class='lin_rf', image=lin_mei.astype('uint8'))
#
#             cnn_key['image_id'] = next_image_id
#             cnn_key['image_class'] = 'cnn_mei'
#             lin_key['image_id'] = next_image_id
#             lin_key['image_class'] = 'lin_rf'
#
#             with dj.conn().transaction:
#                 stimulus.StaticImage.Image().insert([new_rf_key, new_mei_key])
#                 stimulus.StaticImage.MEI().insert1(cnn_key)
#                 stimulus.StaticImage.MEI().insert1(lin_key)


# Fill stimulus table with MEI images
def fill_multi_mei_stimulus(group_id, dryrun=False):

    gid_restr = dict(group_id=group_id)

    restr = dj.AndList([multi_mei.CorrectedHighUnitSelection & 'hu_rank < 150',
                        'img_config_id=0 and mei_param_id = "150966b4691c643986d738a5f6656594"'])
    data_config = configs.DataConfig.CorrectedAreaLayer & 'stimulus_type="stimulus.Frame" and exclude="" and normalize_per_image=False'

    cnn_meis = (multi_mei.ProcessedShiftedMEI2 & multi_mei.ModelGroup.CNNModel & data_config & gid_restr & restr).proj(cnn_hash='net_hash', cnn_seed='seed',
                                                                                   cnn_mei='shifted_mei')
    lin_meis = (multi_mei.ProcessedShiftedMEI2 & multi_mei.ModelGroup.LinearModel & data_config & gid_restr & restr).proj(lin_hash='net_hash',
                                                                                      lin_seed='seed',
                                                                                      lin_mei='shifted_mei')

    combo = cnn_meis * lin_meis
    n_good_comb = len(combo)
    assert n_good_comb == len(cnn_meis) == len(lin_meis), 'Entries should match between CNN and Linear!'

    keys = (cnn_meis * lin_meis * multi_mei.CorrectedHighUnitSelection).fetch('KEY',
                                                                                order_by='hu_rank, x_shift, y_shift')
    assert len(keys) == n_good_comb, 'Number of entries expanded!!'


    all_images = []
    all_multi_meis = []
    max_image_id = dj.U().aggr(stimulus.StaticImage.Image() & 'image_class="multi_cnn_mei"',
                               cmax='max(image_id)').fetch1('cmax')
    next_image_id = 0 if max_image_id is None else (max_image_id + 1)
    for key in keys:
        if stimulus.StaticImage.MultiMEI() & key & 'image_class="multi_cnn_mei"':
            print('\rEntries found. Skipping...', end='')
            continue
        else:
            print('Creating image ID {}'.format(next_image_id))
            cnn_key, cnn_mei = (multi_mei.ProcessedShiftedMEI2 & key & multi_mei.ModelGroup.CNNModel).fetch1('KEY', 'shifted_mei')
            lin_key, lin_mei = (multi_mei.ProcessedShiftedMEI2 & key & multi_mei.ModelGroup.LinearModel).fetch1('KEY', 'shifted_mei')

            if dryrun:
                f, f_inv = multi_mei.ClosestCalibration().get_interp(multi_mei.TargetDataset.Unit & key)
                lum_cnn = f(cnn_mei)
                lum_lin = f(lin_mei)
                target_mu, target_sigma = (multi_mei.ProcessedImageConfig & key).fetch1('img_mean', 'img_contrast')
                mu_cnn, sigma_cnn = lum_cnn.mean(), lum_cnn.std()
                mu_lin, sigma_lin = lum_lin.mean(), lum_lin.std()

                print('CNN MEI', mu_cnn, sigma_cnn)
                print('Linear MEI', mu_lin, sigma_lin)
                good_means = np.abs(1 - mu_cnn / target_mu) < 0.05 and np.abs(1 - mu_lin / target_mu) < 0.05
                good_sigmas = np.abs(1 - sigma_cnn / target_sigma) < 0.05 and np.abs(1 - sigma_lin / target_sigma) < 0.05

                if not (good_means and good_sigmas):
                    warnings.warn('Image statistic for {} not within acceptable limit...'.format(key))

            # clip and convert to uint8
            cnn_mei = np.clip(cnn_mei, 0, 255).astype('uint8')
            lin_mei = np.clip(lin_mei, 0, 255).astype('uint8')

            new_mei_key = dict(image_id=next_image_id, image_class='multi_cnn_mei', image=cnn_mei)
            new_rf_key = dict(image_id=next_image_id, image_class='multi_lin_rf', image=lin_mei)

            cnn_key['image_id'] = next_image_id
            cnn_key['image_class'] = 'multi_cnn_mei'
            cnn_key['src_table'] = 'multi_mei.ProcessedShiftedMEI2'
            lin_key['image_id'] = next_image_id
            lin_key['image_class'] = 'multi_lin_rf'
            lin_key['src_table'] = 'multi_mei.ProcessedShiftedMEI2'

            all_images.extend([new_rf_key, new_mei_key])
            all_multi_meis.extend([cnn_key, lin_key])

            next_image_id += 1

    if not dryrun:
        with dj.conn().transaction:
            print('Inserting...', flush=True)
            stimulus.StaticImage.Image().insert(all_images)
            stimulus.StaticImage.MultiMEI().insert(all_multi_meis)

def fill_multi_mei_stimulus_old(group_id, dryrun=False):

    gid_restr = dict(group_id=group_id)

    restr = dj.AndList([multi_mei.CorrectedHighUnitSelection & 'hu_rank < 150',
                        'img_config_id=0 and mei_param_id = "150966b4691c643986d738a5f6656594"'])
    data_config = configs.DataConfig.CorrectedAreaLayer & 'stimulus_type="stimulus.Frame" and exclude="" and normalize_per_image=False'

    cnn_meis = (multi_mei.ProcessedShiftedMEI & multi_mei.ModelGroup.CNNModel & data_config & gid_restr & restr).proj(cnn_hash='net_hash', cnn_seed='seed',
                                                                                   cnn_mei='shifted_mei')
    lin_meis = (multi_mei.ProcessedShiftedMEI & multi_mei.ModelGroup.LinearModel & data_config & gid_restr & restr).proj(lin_hash='net_hash',
                                                                                      lin_seed='seed',
                                                                                      lin_mei='shifted_mei')

    combo = cnn_meis * lin_meis
    n_good_comb = len(combo)
    assert n_good_comb == len(cnn_meis) == len(lin_meis), 'Entries should match between CNN and Linear!'

    keys = (cnn_meis * lin_meis * multi_mei.CorrectedHighUnitSelection).fetch('KEY',
                                                                                order_by='hu_rank, x_shift, y_shift')
    assert len(keys) == n_good_comb, 'Number of entries expanded!!'


    all_images = []
    all_multi_meis = []
    max_image_id = dj.U().aggr(stimulus.StaticImage.Image() & 'image_class="multi_cnn_mei"',
                               cmax='max(image_id)').fetch1('cmax')
    next_image_id = 0 if max_image_id is None else (max_image_id + 1)
    for key in keys:
        if stimulus.StaticImage.MultiMEI() & key & 'image_class="multi_cnn_mei"':
            print('\rEntries found. Skipping...', end='')
            continue
        else:
            print('Creating image ID {}'.format(next_image_id))
            cnn_key, cnn_mei = (multi_mei.ProcessedShiftedMEI & key & multi_mei.ModelGroup.CNNModel).fetch1('KEY', 'shifted_mei')
            lin_key, lin_mei = (multi_mei.ProcessedShiftedMEI & key & multi_mei.ModelGroup.LinearModel).fetch1('KEY', 'shifted_mei')

            # clip and convert to uint8
            cnn_mei = np.clip(cnn_mei, 0, 255).astype('uint8')
            lin_mei = np.clip(lin_mei, 0, 255).astype('uint8')

            new_mei_key = dict(image_id=next_image_id, image_class='multi_cnn_mei', image=cnn_mei)
            new_rf_key = dict(image_id=next_image_id, image_class='multi_lin_rf', image=lin_mei)

            cnn_key['image_id'] = next_image_id
            cnn_key['image_class'] = 'multi_cnn_mei'
            cnn_key['src_table'] = 'multi_mei.ProcessedShiftedMEI'
            lin_key['image_id'] = next_image_id
            lin_key['image_class'] = 'multi_lin_rf'
            lin_key['src_table'] = 'multi_mei.ProcessedShiftedMEI'

            all_images.extend([new_rf_key, new_mei_key])
            all_multi_meis.extend([cnn_key, lin_key])

            next_image_id += 1

    if not dryrun:
        with dj.conn().transaction:
            print('Inserting...', flush=True)
            stimulus.StaticImage.Image().insert(all_images)
            stimulus.StaticImage.MultiMEI().insert(all_multi_meis)




def fill_gabor_stimulus(group_id, dryrun=False):

    gid_restr = dict(group_id=group_id)

    restr = dj.AndList([multi_mei.CorrectedHighUnitSelection & 'hu_rank < 150',
                        'img_config_id=0 and gaborrange_id=1',
                        multi_mei.ModelGroup.CNNModel])
    data_config = configs.DataConfig.CorrectedAreaLayer & 'stimulus_type="stimulus.Frame" and exclude="" and normalize_per_image=False'


    gabors = multi_mei.ProcessedBestGabor & data_config & gid_restr & restr


    n_good_comb = len(gabors)

    keys, gabor_images = (gabors * multi_mei.CorrectedHighUnitSelection).fetch('KEY', 'gabor',
                                                                              order_by='hu_rank')
    assert len(keys) == n_good_comb, 'Number of entries expanded!!'

    all_images = []
    all_gabor_keys = []

    max_gabor_id = dj.U().aggr(stimulus.StaticImage.Image() & 'image_class="gabor"',
                               cmax='max(image_id)').fetch1('cmax')
    next_gabor_id = 0 if max_gabor_id is None else (max_gabor_id + 1)


    for key, gabor_image in zip(keys, gabor_images):
        f, f_inv = multi_mei.ClosestCalibration().get_interp(multi_mei.TargetDataset.Unit & key)
        lum_gabor = f(gabor_image)
        target_mu, target_sigma = (multi_mei.ProcessedImageConfig & key).fetch1('img_mean', 'img_contrast')
        mu_gabor, sigma_gabor = lum_gabor.mean(), lum_gabor.std()


        print('Gabor', mu_gabor, sigma_gabor)
        good_means = np.abs(1 - mu_gabor/target_mu) < 0.05
        good_sigmas = np.abs(1 - sigma_gabor/target_sigma) < 0.05

        if not (good_means and good_sigmas):
            warnings.warn('Image statistic for {} not within acceptable limit...'.format(key))

        if not stimulus.StaticImage.Gabor() & key:
            print('\rCreating Gabor entry with ID {}...'.format(next_gabor_id), end='')
            gabor_image = np.clip(gabor_image, 0, 255).astype('uint8')
            image_key = dict(image_id=next_gabor_id, image_class='gabor', image=gabor_image)
            all_images.append(image_key)

            gabor_key = dict(key)
            gabor_key['image_id'] = next_gabor_id
            gabor_key['image_class'] = 'gabor'
            gabor_key['src_table'] = 'multi_mei.ProcessedBestGabor'
            all_gabor_keys.append(gabor_key)
            next_gabor_id += 1
        else:
            print('\rEntries found for Gabor. Skipping...', end='')


    if not dryrun:
        with dj.conn().transaction:
            print('Inserting...', flush=True)
            stimulus.StaticImage.Image().insert(all_images)
            stimulus.StaticImage.Gabor().insert(all_gabor_keys, ignore_extra_fields=True)


def fill_tight_mei_vs_imagenet_stimulus(group_id, dryrun=False):

    gid_restr = dict(group_id=group_id)

    restr = dj.AndList([multi_mei.CorrectedHighUnitSelection & 'hu_rank < 150',
                        'img_config_id=0 and mei_param_id = "150966b4691c643986d738a5f6656594"',
                        multi_mei.ModelGroup.CNNModel,
                        dict(tight_mask_param_id=2)])
    data_config = configs.DataConfig.CorrectedAreaLayer & 'stimulus_type="stimulus.Frame" and exclude="" and normalize_per_image=False'


    masked_mei = multi_mei.ProcessedTightMaskedMEI & data_config & gid_restr & restr
    masked_imagenet = multi_mei.ProcessedTightMaskedImage & data_config & gid_restr & restr

    combo = masked_mei * masked_imagenet

    n_good_comb = len(combo)
    #assert n_good_comb == len(masked_mei) == len(masked_imagenet), 'Entries should match between MEI and ImageNet images!'

    keys, masked_meis, masked_imagenets = (masked_mei * masked_imagenet * multi_mei.CorrectedHighUnitSelection).fetch('KEY', 'masked_mei', 'masked_imagenet',
                                                                              order_by='hu_rank')
    assert len(keys) == n_good_comb, 'Number of entries expanded!!'

    all_images = []
    all_mei_keys = []
    all_imagenet_keys = []

    max_mei_id = dj.U().aggr(stimulus.StaticImage.Image() & 'image_class="masked_mei"',
                               cmax='max(image_id)').fetch1('cmax')
    next_mei_id = 0 if max_mei_id is None else (max_mei_id + 1)

    max_imagenet_id = dj.U().aggr(stimulus.StaticImage.Image() & 'image_class="masked_imagenet"',
                             cmax='max(image_id)').fetch1('cmax')
    next_imagenet_id = 0 if max_imagenet_id is None else (max_imagenet_id + 1)

    for key, masked_mei, masked_imagenet in zip(keys, masked_meis, masked_imagenets):
        f, f_inv = multi_mei.ClosestCalibration().get_interp(multi_mei.TargetDataset.Unit & key)
        lum_mei = f(masked_mei)
        lum_img = f(masked_imagenet)
        target_mu, target_sigma = (multi_mei.ProcessedImageConfig & key).fetch1('img_mean', 'img_contrast')
        mu_mei, sigma_mei = lum_mei.mean(), lum_mei.std()
        mu_img, sigma_img = lum_img.mean(), lum_img.std()


        print('MEI', mu_mei, sigma_mei)
        print('ImageNet', mu_img, sigma_img)
        good_means = np.abs(1 - mu_mei/target_mu) < 0.05 and np.abs(1 - mu_img/target_mu) < 0.05
        good_sigmas = np.abs(1 - sigma_mei/target_sigma) < 0.05 and np.abs(1 - sigma_img/target_sigma) < 0.05

        if not (good_means and good_sigmas):
            warnings.warn('Image statistic for {} not within acceptable limit...'.format(key))

        if not stimulus.StaticImage.TightMaskedMEI() & key:
            print('\rCreating masked MEI ID {}...'.format(next_mei_id), end='')
            masked_mei = np.clip(masked_mei, 0, 255).astype('uint8')
            mei_image_key = dict(image_id=next_mei_id, image_class='masked_mei', image=masked_mei)
            all_images.append(mei_image_key)

            mei_key = dict(key)
            mei_key['image_id'] = next_mei_id
            mei_key['image_class'] = 'masked_mei'
            mei_key['src_table'] = 'multi_mei.ProcessedTightMaskedMEI'
            all_mei_keys.append(mei_key)
            next_mei_id += 1
        else:
            print('\rEntries found for masked MEI. Skipping...', end='')

        if not stimulus.StaticImage.TightMaskedImageNet() & key:
            print('\rCreating masked ImageNet ID {}...'.format(next_imagenet_id), end='')
            masked_imagenet = np.clip(masked_imagenet, 0, 255).astype('uint8')
            imagenet_image_key = dict(image_id=next_imagenet_id, image_class='masked_imagenet', image=masked_imagenet)
            all_images.append(imagenet_image_key)

            imagenet_key = dict(key)
            imagenet_key['image_id'] = next_imagenet_id
            imagenet_key['image_class'] = 'masked_imagenet'
            imagenet_key['src_table'] = 'multi_mei.ProcessedTightMaskedImage'
            all_imagenet_keys.append(imagenet_key)
            next_imagenet_id += 1
        else:
            print('\rEntries found for masked ImageNet image. Skipping...', end='')


    if not dryrun:
        with dj.conn().transaction:
            print('Inserting...', flush=True)
            stimulus.StaticImage.Image().insert(all_images)
            stimulus.StaticImage.TightMaskedMEI().insert(all_mei_keys, ignore_extra_fields=True)
            stimulus.StaticImage.TightMaskedImageNet().insert(all_imagenet_keys, ignore_extra_fields=True)

def fill_tight_unmasked_imagenet_stimulus(group_id, dryrun=False):
    gid_restr = dict(group_id=group_id)

    restr = dj.AndList([multi_mei.CorrectedHighUnitSelection & 'hu_rank < 150',
                        'img_config_id=0 and mei_param_id = "150966b4691c643986d738a5f6656594"',
                        multi_mei.ModelGroup.CNNModel,
                        dict(tight_mask_param_id=2)])
    data_config = configs.DataConfig.CorrectedAreaLayer & 'stimulus_type="stimulus.Frame" and exclude="" and normalize_per_image=False'

    unmasked_imagenet = multi_mei.ProcessedTightUnmaskedImage & data_config & gid_restr & restr


    n_good_comb = len(unmasked_imagenet)

    keys, unmasked_images = (unmasked_imagenet * multi_mei.CorrectedHighUnitSelection).fetch('KEY', 'unmasked_imagenet',
                                                                              order_by='hu_rank')
    assert len(keys) == n_good_comb, 'Number of entries expanded!!'

    all_images = []
    all_keys = []

    max_unmasked_id = dj.U().aggr(stimulus.StaticImage.Image() & 'image_class="unmasked_imgnet"',
                               cmax='max(image_id)').fetch1('cmax')
    next_unmasked_id = 0 if max_unmasked_id is None else (unmasked_imagenet_id + 1)


    for key, unmasked_image in zip(keys, unmasked_images):
        f, f_inv = multi_mei.ClosestCalibration().get_interp(multi_mei.TargetDataset.Unit & key)
        lum_unmasked = f(unmasked_image)
        target_mu, target_sigma = (multi_mei.ProcessedImageConfig & key).fetch1('img_mean', 'img_contrast')
        mu_unmasked, sigma_unmasked = lum_unmasked.mean(), lum_unmasked.std()


        print('Unmasked Imagenet', mu_unmasked, sigma_unmasked)
        good_means = np.abs(1 - mu_unmasked/target_mu) < 0.05
        good_sigmas = np.abs(1 - sigma_unmasked/target_sigma) < 0.05

        if not (good_means and good_sigmas):
            warnings.warn('Image statistic for {} not within acceptable limit...'.format(key))

        if not stimulus.StaticImage.TightUnmaskedImageNet() & key:
            print('\rCreating Unmasked ImageNet entry with ID {}...'.format(next_unmasked_id), end='')
            unmasked_image = np.clip(unmasked_image, 0, 255).astype('uint8')
            image_key = dict(image_id=next_unmasked_id, image_class='unmasked_imgnet', image=unmasked_image)
            all_images.append(image_key)

            unmasked_key = dict(key)
            unmasked_key['image_id'] = next_unmasked_id
            unmasked_key['image_class'] = 'unmasked_imgnet'
            unmasked_key['src_table'] = 'multi_mei.ProcessedTightUnmaskedImage'
            all_keys.append(unmasked_key)
            next_unmasked_id += 1
        else:
            print('\rEntries found for Unmasked Imagenet. Skipping...', end='')


    if not dryrun:
        with dj.conn().transaction:
            print('Inserting...', flush=True)
            stimulus.StaticImage.Image().insert(all_images)
            stimulus.StaticImage.TightUnmaskedImageNet().insert(all_keys, ignore_extra_fields=True)


def fill_mei_vs_imagenet_stimulus(group_id, dryrun=False):

    gid_restr = dict(group_id=group_id)

    restr = dj.AndList([multi_mei.CorrectedHighUnitSelection & 'hu_rank < 150',
                        'img_config_id=0 and mei_param_id = "150966b4691c643986d738a5f6656594"',
                        multi_mei.ModelGroup.CNNModel,
                        dict(mask_param_id=0, gaussian_factor_id=0)])
    data_config = configs.DataConfig.CorrectedAreaLayer & 'stimulus_type="stimulus.Frame" and exclude="" and normalize_per_image=False'


    masked_mei = multi_mei.ProcessedMaskedMEI & data_config & gid_restr & restr
    masked_imagenet = multi_mei.ProcessedMaskedImage & data_config & gid_restr & restr

    combo = masked_mei * masked_imagenet

    n_good_comb = len(combo)
    #assert n_good_comb == len(masked_mei) == len(masked_imagenet), 'Entries should match between MEI and ImageNet images!'

    keys, masked_meis, masked_imagenets = (masked_mei * masked_imagenet * multi_mei.CorrectedHighUnitSelection).fetch('KEY', 'masked_mei', 'masked_imagenet',
                                                                              order_by='hu_rank')
    assert len(keys) == n_good_comb, 'Number of entries expanded!!'

    all_images = []
    all_mei_keys = []
    all_imagenet_keys = []

    max_mei_id = dj.U().aggr(stimulus.StaticImage.Image() & 'image_class="masked_mei"',
                               cmax='max(image_id)').fetch1('cmax')
    next_mei_id = 0 if max_mei_id is None else (max_mei_id + 1)

    max_imagenet_id = dj.U().aggr(stimulus.StaticImage.Image() & 'image_class="masked_imagenet"',
                             cmax='max(image_id)').fetch1('cmax')
    next_imagenet_id = 0 if max_imagenet_id is None else (max_imagenet_id + 1)

    for key, masked_mei, masked_imagenet in zip(keys, masked_meis, masked_imagenets):
        f, f_inv = multi_mei.ClosestCalibration().get_interp(multi_mei.TargetDataset.Unit & key)
        lum_mei = f(masked_mei)
        lum_img = f(masked_imagenet)
        target_mu, target_sigma = (multi_mei.ProcessedImageConfig & key).fetch1('img_mean', 'img_contrast')
        mu_mei, sigma_mei = lum_mei.mean(), lum_mei.std()
        mu_img, sigma_img = lum_img.mean(), lum_img.std()


        print('MEI', mu_mei, sigma_mei)
        print('ImageNet', mu_img, sigma_img)
        good_means = np.abs(1 - mu_mei/target_mu) < 0.05 and np.abs(1 - mu_img/target_mu) < 0.05
        good_sigmas = np.abs(1 - sigma_mei/target_sigma) < 0.05 and np.abs(1 - sigma_img/target_sigma) < 0.05

        if not (good_means and good_sigmas):
            warnings.warn('Image statistic for {} not within acceptable limit. Skipping...'.format(key))

        if not stimulus.StaticImage.MaskedMEI() & key:
            print('\rCreating masked MEI ID {}...'.format(next_mei_id), end='')
            masked_mei = np.clip(masked_mei, 0, 255).astype('uint8')
            mei_image_key = dict(image_id=next_mei_id, image_class='masked_mei', image=masked_mei)
            all_images.append(mei_image_key)

            mei_key = dict(key)
            mei_key['image_id'] = next_mei_id
            mei_key['image_class'] = 'masked_mei'
            mei_key['src_table'] = 'multi_mei.ProcessedMaskedMEI'
            all_mei_keys.append(mei_key)
            next_mei_id += 1
        else:
            print('\rEntries found for masked MEI. Skipping...', end='')

        if not stimulus.StaticImage.MaskedImageNet() & key:
            print('\rCreating masked ImageNet ID {}...'.format(next_imagenet_id), end='')
            masked_imagenet = np.clip(masked_imagenet, 0, 255).astype('uint8')
            imagenet_image_key = dict(image_id=next_imagenet_id, image_class='masked_imagenet', image=masked_imagenet)
            all_images.append(imagenet_image_key)

            imagenet_key = dict(key)
            imagenet_key['image_id'] = next_imagenet_id
            imagenet_key['image_class'] = 'masked_imagenet'
            imagenet_key['src_table'] = 'multi_mei.ProcessedMaskedImage'
            all_imagenet_keys.append(imagenet_key)
            next_imagenet_id += 1
        else:
            print('\rEntries found for masked ImageNet image. Skipping...', end='')


    if not dryrun:
        with dj.conn().transaction:
            print('Inserting...', flush=True)
            stimulus.StaticImage.Image().insert(all_images)
            stimulus.StaticImage.MaskedMEI().insert(all_mei_keys, ignore_extra_fields=True)
            stimulus.StaticImage.MaskedImageNet().insert(all_imagenet_keys, ignore_extra_fields=True)




def fill_gaussian_masked_vs_unmasked_imagenet_stimulus(group_id, dryrun=False):
    gid_restr = dict(group_id=group_id)

    restr = dj.AndList([multi_mei.CorrectedHighUnitSelection & 'hu_rank < 150',
                        'mei_param_id = "150966b4691c643986d738a5f6656594"',
                        multi_mei.ModelGroup.CNNModel,
                        dict(gaussian_factor_id=0)])
    data_config = configs.DataConfig.CorrectedAreaLayer & 'stimulus_type="stimulus.Frame" and exclude="" and normalize_per_image=False'


    keys, masked_imagenets, original_imagenets = ((multi_mei.GaussianMaskedSelectedImage & data_config & gid_restr & restr) * multi_mei.CorrectedHighUnitSelection).fetch('KEY', 'masked_imagenet_image', 'original_imagenet_image',
                                                                              order_by='hu_rank')


    all_images = []
    all_imagenet_keys = []


    max_id = dj.U().aggr(stimulus.StaticImage.Image() & 'image_class="gaussian_orig"',
                         cmax='max(image_id)').fetch1('cmax')
    max_masked_id = dj.U().aggr(stimulus.StaticImage.Image() & 'image_class="gaussian_masked"',
                                cmax='max(image_id)').fetch1('cmax')

    assert max_id == max_masked_id, 'ID mismatch between gaussian_orig and gaussian_masked!'

    next_id = 0 if max_id is None else (max_id + 1)


    for key, original_imagenet, masked_imagenet in zip(keys, original_imagenets, masked_imagenets):

        if not stimulus.StaticImage.RawMaskedImageNet() & key:
            print('\rCreating original ImageNet ID {}...'.format(next_id), end='')
            original_imagenet = np.clip(original_imagenet, 0, 255).astype('uint8')
            orig_image_key = dict(image_id=next_id, image_class='gaussian_orig', image=original_imagenet)
            all_images.append(orig_image_key)

            print('\rCreating Gaussian masked ImageNet ID {}...'.format(next_id), end='')
            masked_imagenet = np.clip(masked_imagenet, 0, 255).astype('uint8')
            masked_image_key = dict(image_id=next_id, image_class='gaussian_masked', image=masked_imagenet)
            all_images.append(masked_image_key)

            imagenet_key = dict(key)
            imagenet_key['image_id'] = next_id
            imagenet_key['image_class'] = 'gaussian_orig'
            imagenet_key['src_table'] = 'multi_mei.GaussianMaskedSelectedImage'
            all_imagenet_keys.append(imagenet_key)

            imagenet_key = dict(key)
            imagenet_key['image_id'] = next_id
            imagenet_key['image_class'] = 'gaussian_masked'
            imagenet_key['src_table'] = 'multi_mei.GaussianMaskedSelectedImage'
            all_imagenet_keys.append(imagenet_key)

            next_id += 1
        else:
            print('\rEntries found for masked MEI. Skipping...', end='')


    if not dryrun:
        with dj.conn().transaction:
            print('Inserting...', flush=True)
            stimulus.StaticImage.Image().insert(all_images)
            stimulus.StaticImage.RawMaskedImageNet().insert(all_imagenet_keys, ignore_extra_fields=True)






def old_fill_multi_mei_stimulus(group_id):
    """
    Caution! - this is an outdated copy that was used to fill images for the first and second mouse (group_id 14 and 15)
    Namely, it sources the image from multi_mei.ShiftedMEI which performs contrast/luminance matching on non-upscaled
    image and without gamma correction. The new version is provided above
    Args:
        group_id: target to generate images to be saved as part of the stimulus.StaticImage.
    Returns:

    """
    assert group_id in (14, 15), 'This function is outdated and should not be used except for data replication purpose for group_id 14 or 15'
    gid_restr = dict(group_id=group_id)
    restr = dj.AndList([multi_mei.CorrectedHighUnitSelection & 'hu_rank < 150',
                        'img_config_id=0 and mei_param_id = "150966b4691c643986d738a5f6656594"'])
    cnn_meis = (multi_mei.ShiftedMEI & multi_mei.ModelGroup.CNNModel & gid_restr & restr).proj(cnn_hash='net_hash', cnn_seed='seed',
                                                                                   cnn_mei='shifted_mei')
    lin_meis = (multi_mei.ShiftedMEI & multi_mei.ModelGroup.LinearModel & gid_restr & restr).proj(lin_hash='net_hash',
                                                                                      lin_seed='seed',
                                                                                      lin_mei='shifted_mei')

    combo = cnn_meis * lin_meis
    n_good_comb = len(combo)
    assert n_good_comb == len(cnn_meis) == len(lin_meis), 'Entries should match between CNN and Linear!'

    keys = (cnn_meis * lin_meis * multi_mei.CorrectedHighUnitSelection).fetch('KEY',
                                                                                order_by='hu_rank, x_shift, y_shift')
    assert len(keys) == n_good_comb, 'Number of entries expanded!!'

    for key in keys:
        if stimulus.StaticImage.MultiMEI() & key & 'image_class="multi_cnn_mei"':
            print('Entries found. Skipping...')
            continue
        else:
            max_image_id = dj.U().aggr(stimulus.StaticImage.Image() & 'image_class="multi_cnn_mei"',
                                       cmax='max(image_id)').fetch1('cmax')
            next_image_id = 0 if max_image_id is None else (max_image_id + 1)
            print('Creating image ID {}'.format(next_image_id))

            cnn_key, cnn_mei = (multi_mei.ShiftedMEI & key & multi_mei.ModelGroup.CNNModel).fetch1('KEY', 'shifted_mei')
            lin_key, lin_mei = (multi_mei.ShiftedMEI & key & multi_mei.ModelGroup.LinearModel).fetch1('KEY', 'shifted_mei')

            # upscale the MEIs by a factor of 4
            cnn_mei = ndimage.zoom(cnn_mei, zoom=4, mode='reflect')
            cnn_mei = np.clip(cnn_mei, 0, 255).astype('uint8')

            lin_mei = ndimage.zoom(lin_mei, zoom=4, mode='reflect')
            lin_mei = np.clip(lin_mei, 0, 255).astype('uint8')

            new_mei_key = dict(image_id=next_image_id, image_class='multi_cnn_mei', image=cnn_mei.astype('uint8'))
            new_rf_key = dict(image_id=next_image_id, image_class='multi_lin_rf', image=lin_mei.astype('uint8'))

            cnn_key['image_id'] = next_image_id
            cnn_key['image_class'] = 'multi_cnn_mei'
            lin_key['image_id'] = next_image_id
            lin_key['image_class'] = 'multi_lin_rf'

            with dj.conn().transaction:
                stimulus.StaticImage.Image().insert([new_rf_key, new_mei_key])
                stimulus.StaticImage.MultiMEI().insert1(cnn_key)
                stimulus.StaticImage.MultiMEI().insert1(lin_key)




def fill_eigen(restr=None):
    if restr is None:
        restr = {}
    multi_mei.ImageGradResponse.populate(restr, stimulus.StaticImage.MultiMEI(), reserve_jobs=True, order='random')
    multi_mei.Top100Eigen().populate(restr, reserve_jobs=True, order='random')
    multi_mei.AllEigen().populate(restr, reserve_jobs=True, order='random')


@schema
class MEISourceScan(dj.Computed):
    definition = """
    -> meso.ScanInfo
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> meso.ScanSet.Unit
        """

    @property
    def key_source(self):
        return meso.ScanInfo & (stimulus.StaticImage.MultiMEI * data_schemas.StaticMultiDataset.Member)

    def make(self, key):
        self.insert1(key)
        units = (meso.ScanSet.Unit & (
                    stimulus.StaticImage.MultiMEI * data_schemas.StaticMultiDataset.Member * multi_mei.TargetDataset.Unit & key)).fetch('KEY')
        self.Unit.insert(units)


@schema
class StimType(dj.Lookup):
    definition = """
    stim_type: varchar(16)  # name of the stim type
    ---
    stim_description: varchar(1024) # longer description about the stimulus
    """
    contents = [
        ('imagenet', 'Natural images used to train models and to generate MEI'),
        ('mei', 'Closed loop presentation of MEI'),
        ('noise', 'Noise images used to map RFs'),
        ('gabor', 'Closed loop comparison of MEIs vs Gabor'),
        ('natural', 'Closed loop comparison of Masked MEIs vs masked ImageNet images'),
        ('unmaskednatural', 'Closed loop comparison of Masked MEIs vs unmasked ImageNet images'),
    ]


@schema
class ClosedLoopScan(dj.Manual):
    definition = """
    # list of scans part of the closed loop
    -> experiment.Scan
    ---
    loop_group: int  # used to group related scans
    day: int        # day number in the closed loop starting with 1
    mei_source: tinyint  # whether it was used to generate MEI
    -> StimType
    """

scans = [
    # Loop group 0
    dict(animal_id=20505, session=2, scan_idx=24, loop_group=0, day=1, mei_source=1, stim_type="imagenet"),
    dict(animal_id=20505, session=3, scan_idx=7,  loop_group=0, day=1, mei_source=0, stim_type="imagenet"), # repeat imagenet day 1
    dict(animal_id=20505, session=5, scan_idx=26, loop_group=0, day=3, mei_source=0, stim_type="mei"), # MEI day 3
    dict(animal_id=20505, session=6, scan_idx=1,  loop_group=0, day=3, mei_source=0, stim_type="imagenet"), # imagenet day 3
    dict(animal_id=20505, session=7, scan_idx=23, loop_group=0, day=4, mei_source=0, stim_type="mei"), # MEI day 4
    dict(animal_id=20505, session=7, scan_idx=29, loop_group=0, day=4, mei_source=0, stim_type="imagenet"), # imagenet day 4
    # Loop group 1
    dict(animal_id=20457, session=5, scan_idx=9, loop_group=1, day=1, mei_source=1, stim_type="imagenet"),
    dict(animal_id=20457, session=5, scan_idx=17, loop_group=1, day=1, mei_source=0, stim_type="imagenet"),
    dict(animal_id=20457, session=5, scan_idx=27, loop_group=1, day=1, mei_source=0, stim_type="noise"),
    dict(animal_id=20457, session=7, scan_idx=4, loop_group=1, day=2, mei_source=0, stim_type="mei"),
    dict(animal_id=20457, session=7, scan_idx=10, loop_group=1, day=2, mei_source=0, stim_type="imagenet"),
    dict(animal_id=20457, session=7, scan_idx=16, loop_group=1, day=2, mei_source=0, stim_type="noise"),
    dict(animal_id=20457, session=8, scan_idx=9, loop_group=1, day=3, mei_source=0, stim_type="mei"),
    dict(animal_id=20457, session=8, scan_idx=12, loop_group=1, day=3, mei_source=0, stim_type="imagenet"),
    dict(animal_id=20457, session=8, scan_idx=22, loop_group=1, day=3, mei_source=0, stim_type="noise"),
    # Loop group 2
    dict(animal_id=20505, session=10, scan_idx=14, loop_group=2, day=1, mei_source=1, stim_type="imagenet"),
    dict(animal_id=20505, session=10, scan_idx=19, loop_group=2, day=1, mei_source=0, stim_type="imagenet"),
    dict(animal_id=20505, session=11, scan_idx=16, loop_group=2, day=2, mei_source=0, stim_type="imagenet"),
    dict(animal_id=20505, session=12, scan_idx=16, loop_group=2, day=3, mei_source=0, stim_type="mei"),
    dict(animal_id=20505, session=12, scan_idx=29, loop_group=2, day=3, mei_source=0, stim_type="imagenet"),
    dict(animal_id=20505, session=14, scan_idx=4, loop_group=2, day=4, mei_source=0, stim_type="mei"),
    dict(animal_id=20505, session=14, scan_idx=33, loop_group=2, day=4, mei_source=0, stim_type="imagenet"),
    # Loop group 3
    dict(animal_id=20210, session=4, scan_idx=11, loop_group=3, day=1, mei_source=1, stim_type="imagenet"),
    dict(animal_id=20210, session=5, scan_idx=16, loop_group=3, day=2, mei_source=0, stim_type="imagenet"),
    dict(animal_id=20210, session=7, scan_idx=10, loop_group=3, day=3, mei_source=0, stim_type="mei"),
    dict(animal_id=20210, session=7, scan_idx=14, loop_group=3, day=3, mei_source=0, stim_type="imagenet"),
    dict(animal_id=20210, session=8, scan_idx=17, loop_group=3, day=4, mei_source=0, stim_type="imagenet"),
    # Loop group 4
    dict(animal_id=20892, session=3, scan_idx=14, loop_group=4, day=1, mei_source=1, stim_type="imagenet"),
    dict(animal_id=20892, session=4, scan_idx=16, loop_group=4, day=2, mei_source=0, stim_type="imagenet"),
    dict(animal_id=20892, session=5, scan_idx=18, loop_group=4, day=3, mei_source=0, stim_type="mei"),
    dict(animal_id=20892, session=6, scan_idx=17, loop_group=4, day=4, mei_source=0, stim_type="mei"),
    dict(animal_id=20892, session=6, scan_idx=24, loop_group=4, day=4, mei_source=0, stim_type="imagenet"),
    # Loop group 5
    dict(animal_id=21067, session=9, scan_idx=17, loop_group=5, day=1, mei_source=1, stim_type="imagenet"),
    dict(animal_id=21067, session=10, scan_idx=14, loop_group=5, day=2, mei_source=0, stim_type="mei"),
    dict(animal_id=21067, session=10, scan_idx=18, loop_group=5, day=2, mei_source=0, stim_type="imagenet"),
    dict(animal_id=21067, session=11, scan_idx=12, loop_group=5, day=3, mei_source=0, stim_type="gabor"),
    dict(animal_id=21067, session=11, scan_idx=21, loop_group=5, day=3, mei_source=0, stim_type="imagenet"),
    dict(animal_id=21067, session=12, scan_idx=11, loop_group=5, day=4, mei_source=0, stim_type="natural"),
    dict(animal_id=21067, session=12, scan_idx=15, loop_group=5, day=4, mei_source=0, stim_type="imagenet"),
    dict(animal_id=21067, session=13, scan_idx=10, loop_group=5, day=5, mei_source=0, stim_type="unmaskednatural"),
    dict(animal_id=21067, session=13, scan_idx=14, loop_group=5, day=5, mei_source=0, stim_type="imagenet"),
]

ClosedLoopScan.insert(scans, skip_duplicates=True)


@schema
class StructureCellMatch(dj.Computed):
    definition = """
    # cell matches across scans
    -> experiment.Scan.proj(src_session='session', src_scan_idx='scan_idx')
    -> experiment.Scan
    ---
    """
    key_source = (ClosedLoopScan() & 'mei_source=1').proj(src_session='session', src_scan_idx='scan_idx',
                                                                      src_loop_group='loop_group') * \
                 (ClosedLoopScan() & 'mei_source != 1').proj('loop_group') & 'src_loop_group = loop_group'


    class AllMatch(dj.Part):
        definition = """
        -> master
        -> meso.ScanSet.Unit.proj(src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id')
        -> meso.ScanSet.Unit
        ---
        total_cts: int  # total structural scans used
        match_cts: int  # number of structural scan that suggested the match
        src_field: int # field of the source cell
        field: int     # field of the matched cell
        mean_iou: float   # average iou
        max_iou: float  # max iou
        mean_dist3d: float # average distance of matched cell to structural cell centroid
        """

    class BestMatch(dj.Part):
        definition = """
        -> master
        -> meso.ScanSet.Unit.proj(src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id')
        ---
        -> meso.ScanSet.Unit
        total_cts: int  # total structural scans used
        match_cts: int  # number of structural scan that suggested the match
        src_field: int # field of the source cell
        field: int     # field of the matched cell
        mean_iou: float   # average iou
        max_iou: float  # max iou
        mean_dist3d: float # average distance of matched cell to structural cell centroid
        """

    def make(self, key):
        self.insert1(key)
        src_key = dict(animal_id=key['animal_id'], session=key['src_session'], scan_idx=key['src_scan_idx'])
        target_neurons = multi_mei.TargetDataset.Unit * stimulus.StaticImage.MultiMEI & (StaticMultiDataset.Member & src_key)

        src_map = dict(src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id')
        src_units = (meso.ScanSet.Unit & target_neurons).proj(**src_map)
        matched_src_units = meso.Func2StructMatching.Match().proj('sunit_id', **src_map, src_iou='iou',
                                                                  src_field='field',
                                                                  src_distance3d='distance3d') & src_units

        n_max = len(dj.U('stack_session', 'stack_idx') & (meso.Func2StructMatching & key) & (
                    meso.Func2StructMatching & src_key))

        # only consider masks classified as "soma"
        soma_units = (meso.MaskClassification.Type() & 'type="soma"')
        pairs = (meso.ScanSet.Unit() & key & soma_units) * src_units

        keeps = ['src_field', 'field']
        stats = dict(match_cts='count(*)', mean_iou='AVG(iou)', max_iou='max(iou)', mean_dist3d='AVG(distance3d)')
        df = DataFrame((pairs.aggr(dj.U('sunit_id') * (meso.Func2StructMatching.Match() & key & soma_units) * matched_src_units,
                                   *keeps, **stats)).fetch())

        def get_top_selection(gp):
            selection = gp[(gp['match_cts'] == gp['match_cts'].max())]
            if len(selection) == 1:
                return selection
            selection = selection[selection['mean_iou'] == selection['mean_iou'].max()]
            if len(selection) == 1:
                return selection
            raise ValueError('Could not resolve down to a single entry')


        df['match_cts'] = pd.to_numeric(df['match_cts'])
        df['total_cts'] = n_max
        df['mean_iou'] = pd.to_numeric(df['mean_iou'])
        df['max_iou'] = pd.to_numeric(df['max_iou'])
        df['mean_dist3d'] = pd.to_numeric(df['mean_dist3d'])


        rec_array = df.to_records(index=False)
        self.AllMatch.insert(rec_array)

        entries = []
        for i, gp in df.groupby('src_unit_id'):
                entries.append(get_top_selection(gp))
        entries = pd.concat(entries)
        rec_array = entries.to_records(index=False)
        self.BestMatch.insert(rec_array)


src_map = dict(src_session='session', src_scan_idx='scan_idx', src_field='field')


def rename(d, **kwargs):
    return {kwargs[k] if k in kwargs else k: v for k, v in d.items() if k not in kwargs.values()}


@schema
class ProximityCellMatch(dj.Computed):
    definition = """
    -> meso.StackCoordinates.proj(src_session='session', src_scan_idx='scan_idx', src_field='field')
    -> meso.ScanInfo
    ---
    """

    class UnitMatch(dj.Part):
        definition = """
        -> master
        -> meso.StackCoordinates.UnitInfo.proj(src_session='session', src_scan_idx='scan_idx', src_field='field', src_unit_id='unit_id')
        -> meso.ScanInfo
        ---
        -> meso.ScanSet.Unit # matched unit id
        match_distance: float   # distance between matched units in um
        """

    @property
    def key_source(self):
        valid_combo = (ClosedLoopScan() & 'mei_source=1 and loop_group > 0').proj(src_session='session', src_scan_idx='scan_idx',
                                                               src_loop_group='loop_group') * \
                      (ClosedLoopScan() & 'mei_source != 1').proj('loop_group') & 'src_loop_group = loop_group'
        restr = meso.StackCoordinates.proj(**src_map) * meso.StackCoordinates() & valid_combo
        return meso.StackCoordinates.proj(**src_map) * meso.ScanInfo & restr

    def make(self, key):
        self.insert1(key)


        src_key = dict(animal_id=key['animal_id'], session=key['src_session'], scan_idx=key['src_scan_idx'])
        target_neurons = multi_mei.TargetDataset.Unit * stimulus.StaticImage.MultiMEI & (
                    StaticMultiDataset.Member & src_key)

        src_units = (meso.ScanSet.Unit & target_neurons)

        src_map = dict(src_session='session', src_scan_idx='scan_idx', src_field='field')
        src_key = rename(key, **src_map)
        src_unit_keys, src_x, src_y, src_z = (meso.StackCoordinates.UnitInfo() & src_key & src_units).fetch('KEY',
                                                                                                            'stack_x',
                                                                                                            'stack_y',
                                                                                                            'stack_z',
                                                                                                            order_by='unit_id')

        # only consider masks classified as "soma"
        soma_units = (meso.ScanSet.Unit() * meso.MaskClassification.Type() & 'type="soma"')
        dest_unit_keys, dest_x, dest_y, dest_z = (meso.StackCoordinates.UnitInfo() & key & soma_units).fetch('KEY',
                                                                                                             'stack_x',
                                                                                                             'stack_y',
                                                                                                             'stack_z',
                                                                                                             order_by='unit_id')
        dist = np.sqrt((src_x[:, None] - dest_x) ** 2 + (src_y[:, None] - dest_y) ** 2 + (src_z[:, None] - dest_z) ** 2)
        pos = dist.argmin(axis=1)
        min_dist = dist.min(axis=1)
        combs = []
        for src_unit, dest_unit, d in zip(src_unit_keys, [dest_unit_keys[k] for k in pos], min_dist):
            key_comb = dict(key, **dest_unit)
            key_comb['src_unit_id'] = src_unit['unit_id']
            key_comb['match_distance'] = d
            combs.append(key_comb)
        self.UnitMatch.insert(combs, ignore_extra_fields=True)


@schema
class BestProximityCellMatch(dj.Computed):
    definition = """
    -> meso.ScanSet.Unit.proj(src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id')
    -> meso.ScanInfo
    ---
    -> meso.ScanSet.Unit   # most frequently matched unit id
    match_freq:  int       # how many times it was matched
    total_stacks: int      # number of stacks used to match
    mean_distance: float   # average match distance (in um)
    """

    @property
    def key_source(self):
        src_scan = meso.ScanInfo.proj(src_session='session', src_scan_idx='scan_idx')
        dst_scan = meso.ScanInfo
        return src_scan * dst_scan * shared.SegmentationMethod & ProximityCellMatch()

    def make(self, key):
        df = pd.DataFrame((ProximityCellMatch.UnitMatch() & key).fetch())
        attrs = list(key) + ['src_unit_id']

        def get_info(df):
            vc = df['unit_id'].value_counts()
            vals = {}
            vals['unit_id'] = int(vc.index[0])
            vals['match_freq'] = vc.iloc[0]
            vals['total_stacks'] = len(df)
            vals['mean_distance'] = df.groupby('unit_id').mean().loc[vals['unit_id']]['match_distance']
            return pd.Series(vals)

        v = df.groupby(attrs).apply(get_info).reset_index()
        self.insert(v.to_records(index=False))


@schema
class MatchedUnits(dj.Computed):
    definition = """
    -> meso.ScanSet.Unit
    """
    key_source = meso.ScanSet.Unit & BestProximityCellMatch

    def make(self, key):
        self.insert1(key)


@schema
class ClosedLoopConfusion(dj.Computed):
    definition = """
    -> ClosedLoopScan
    """

    class UnitActivation(dj.Part):
        definition = """
        -> master
        -> multi_mei.ModelType
        image_unit_id: int   # image source unit id
        matched_unit_id: int # unit id of the matched neuron
        ---
        mu_resp: float      # average response across repeats
        sem_resp: float     # sem of response across repeats
        n_repeats: int      # number of repeats
        """
    @property
    def key_source(self):
        return ClosedLoopScan & 'loop_group > 0 and stim_type="mei"'

    def make(self, key):

        self.insert1(key)

        # find MEI source scan
        src_scan = (ClosedLoopScan & (
                    dj.U('loop_group') & (ClosedLoopScan & key)) & 'mei_source=True').fetch1('KEY')
        src_key = {'src_' + k: v for k, v in src_scan.items()}

        # find all units matching the source units
        src_units, target_units = (BestProximityCellMatch & src_key & key).fetch('src_unit_id', 'unit_id')
        unit_map = OrderedDict(zip(src_units, target_units))

        # load the data
        dataset = InputResponse().fetch1_data(key)

        # turn into a data frame
        df = DataFrame(dataset['responses'], columns=dataset['neurons']['unit_ids'])
        # convert into non binary string
        df['condition_hash'] = [x.decode() for x in list(dataset['condition_hashes'])]
        # assign repetition count
        df['rep_count'] = df.groupby(by='condition_hash').cumcount()
        # uniquely index the rows
        df = df.set_index(['condition_hash', 'rep_count'], verify_integrity=True)

        # map columns into target units
        df = pd.concat([df[v].rename(k) for k, v in unit_map.items()], axis=1)

        # get information about each stimulus condition
        deltas = multi_mei.CorrectedHighUnitSelection().proj(delta_fraction='cnn_fraction_oracle - lin_fraction_oracle')
        df_cond = DataFrame(
            (stimulus.Frame() * stimulus.StaticImage.MultiMEI * multi_mei.TargetDataset.Unit * deltas).proj(
                'condition_hash', 'image_class', 'unit_id', 'x_shift', 'y_shift', 'delta_fraction').fetch(
                order_by='image_class, unit_id, x_shift, y_shift'))
        df_cond = df_cond[['condition_hash', 'image_class', 'unit_id', 'x_shift', 'y_shift', 'delta_fraction']]
        df_cond = df_cond.set_index(['condition_hash'])

        # merge information
        df_all = df.join(df_cond)

        # reindex using image info and reorder rows according to unit_map
        df_all = df_all.reset_index()
        df_all = df_all.set_index(['image_class', 'unit_id', 'x_shift', 'y_shift', 'rep_count'])
        df_all.sort_index(axis=0, inplace=True)
        df_all = df_all.reindex(unit_map, level='unit_id')

        # only analyze the center, unshifted positions
        df_center = df_all[
            (df_all.index.get_level_values('x_shift') == 0) & (df_all.index.get_level_values('y_shift') == 0)]

        # split into MEI and RF responses

        mei_center = df_center[df_center.index.get_level_values('image_class') == 'multi_cnn_mei']
        mei_center.columns.name = 'matched_unit'
        rf_center = df_center[df_center.index.get_level_values('image_class') == 'multi_lin_rf']
        rf_center.columns.name = 'matched_unit'

        # compute mean and standard deviations across repeats
        g = mei_center.reset_index().groupby('unit_id', sort=False)
        mu_mei_center = g.mean()
        sem_mei_center = g.sem()
        mei_ct = g.condition_hash.count()
        mei_ct.name = 'rep_count'

        g = rf_center.reset_index().groupby('unit_id', sort=False)
        mu_rf_center = g.mean()
        sem_rf_center = g.sem()
        rf_ct = g.condition_hash.count()
        rf_ct.name = 'rep_count'

        unit_entries = list(unit_map)
        mei_confusion = mu_mei_center[unit_entries]
        mei_sem_confusion = sem_mei_center[unit_entries]
        rf_confusion = mu_rf_center[unit_entries]
        rf_sem_confusion = sem_rf_center[unit_entries]

        flat_mei_resp = pd.DataFrame({'mu_resp': mei_confusion.stack(), 'sem_resp': mei_sem_confusion.stack()})
        flat_mei_resp = flat_mei_resp.join(mei_ct)
        flat_rf_resp = pd.DataFrame({'mu_resp': rf_confusion.stack(), 'sem_resp': rf_sem_confusion.stack()})
        flat_rf_resp = flat_rf_resp.join(rf_ct)

        mei_entries = [dict(image_unit_id=row['unit_id'],
                            matched_unit_id=row['matched_unit'],
                            mu_resp=row['mu_resp'],
                            sem_resp=row['sem_resp'],
                            n_repeats=row['rep_count'],
                            model_type='cnn',
                            **key) for i, row in flat_mei_resp.reset_index().iterrows()]

        rf_entries = [dict(image_unit_id=row['unit_id'],
                           matched_unit_id=row['matched_unit'],
                           mu_resp=row['mu_resp'],
                           sem_resp=row['sem_resp'],
                           n_repeats=row['rep_count'],
                           model_type='linear',
                           **key) for i, row in flat_rf_resp.reset_index().iterrows()]

        self.UnitActivation.insert(mei_entries)
        self.UnitActivation.insert(rf_entries)


@schema
class ConfusionDiagonalEntry(dj.Computed):
    definition = """
    -> ClosedLoopScan
    matched_unit_id: int    # unit id of the matched neuron
    ---
    cnn_resp: float      # average response across repeats of CNN
    cnn_sem: float       # sem over CNN resp
    rf_resp: float       # average response across repeats of RF
    rf_sem:  float       # sem over RF resp
    diff_resp: float     # diff in response
    diff_rank: int       # rank of difference with 0 = largest CNN - RF
    t_diff: float        # Welch's t stat of difference
    diff_sem: float      # adjusted difference SEM based on Welch's t stat denominator
    t_rank: int          # rank of t-score based difference with 0 = largest t_diff
    """

    key_source = ClosedLoopConfusion()

    def make(self, key):
        ua = ClosedLoopConfusion.UnitActivation & key & 'image_unit_id = matched_unit_id'

        cnn_ua = (ua & 'model_type="cnn"').proj(t1='model_type', cnn_resp='mu_resp', cnn_sem='sem_resp')
        rf_ua = (ua & 'model_type="linear"').proj(t2='model_type', rf_resp='mu_resp', rf_sem='sem_resp')

        df = pd.DataFrame((cnn_ua * rf_ua).fetch())

        df['diff_resp'] = df['cnn_resp'] - df['rf_resp']

        df['diff_sem'] = np.sqrt(df['cnn_sem']**2 + df['rf_sem']**2)
        df['t_diff'] = df['diff_resp'] / df['diff_sem']

        df.sort_values(by='diff_resp', ascending=False, inplace=True)
        df['diff_rank'] = np.arange(len(df))

        df.sort_values(by='t_diff', ascending=False, inplace=True)
        df['t_rank'] = np.arange(len(df))

        self.insert(df.to_records(), ignore_extra_fields=True)


@schema
class NormalizedConfusion(dj.Computed):
    definition = """
    -> ClosedLoopScan
    """

    class UnitActivation(dj.Part):
        definition = """
        -> master
        -> multi_mei.ModelType
        image_unit_id: int   # image source unit id
        matched_unit_id: int # unit id of the matched neuron
        ---
        mu_resp: float      # average response across repeats
        var_resp: float     # var of response across repeats (ddof=0)
        n_repeats: int      # number of repeats
        """

    @property
    def key_source(self):
        return ClosedLoopScan & 'loop_group > 0 and stim_type="mei"'

    def make(self, key):

        self.insert1(key)

        # find MEI source scan
        src_scan = (ClosedLoopScan & (
                dj.U('loop_group') & (ClosedLoopScan & key)) & 'mei_source=True').fetch1('KEY')
        src_key = {'src_' + k: v for k, v in src_scan.items()}

        # find all units matching the source units
        src_units, target_units = (BestProximityCellMatch & src_key & key).fetch('src_unit_id', 'unit_id')
        unit_map = OrderedDict(zip(src_units, target_units))

        # load the data
        dataset = InputResponse().fetch1_data(key)

        # turn into a data frame
        df = DataFrame(dataset['responses'], columns=dataset['neurons']['unit_ids'])
        # normalize the response
        df = df / df.std(axis=0)
        # convert into non binary string
        df['condition_hash'] = [x.decode() for x in list(dataset['condition_hashes'])]
        # assign repetition count
        df['rep_count'] = df.groupby(by='condition_hash').cumcount()
        # uniquely index the rows
        df = df.set_index(['condition_hash', 'rep_count'], verify_integrity=True)

        # map columns into target units
        df = pd.concat([df[v].rename(k) for k, v in unit_map.items()], axis=1)

        # get information about each stimulus condition
        deltas = multi_mei.CorrectedHighUnitSelection().proj(delta_fraction='cnn_fraction_oracle - lin_fraction_oracle')
        df_cond = DataFrame(
            (stimulus.Frame() * stimulus.StaticImage.MultiMEI * multi_mei.TargetDataset.Unit * deltas).proj(
                'condition_hash', 'image_class', 'unit_id', 'x_shift', 'y_shift', 'delta_fraction').fetch(
                order_by='image_class, unit_id, x_shift, y_shift'))
        df_cond = df_cond[['condition_hash', 'image_class', 'unit_id', 'x_shift', 'y_shift', 'delta_fraction']]
        df_cond = df_cond.set_index(['condition_hash'])

        # add condition information into the response data
        df_all = df.join(df_cond)

        df_all = df_all.reset_index()
        df_all = df_all.set_index(['image_class', 'unit_id', 'x_shift', 'y_shift', 'rep_count'])
        df_all.sort_index(axis=0, inplace=True)
        df_all = df_all.reindex(unit_map, level='unit_id')

        # restrict to center responses
        df_center = df_all[
            (df_all.index.get_level_values('x_shift') == 0) & (df_all.index.get_level_values('y_shift') == 0)]

        df_center.columns.name = 'matched_unit'
        hashes = df_center['condition_hash']

        # get repetition counts
        rep_count = hashes.reset_index().groupby(['image_class', 'unit_id'], sort=False).condition_hash.count()
        rep_count.name = 'n_repeats'
        rep_count = pd.DataFrame(rep_count)

        # compute mean and uncorrected variance
        unit_entries = list(unit_map)
        df_center = df_center[unit_entries]
        g = df_center.reset_index().groupby(['image_class', 'unit_id'], sort=False)

        mu_center = g.mean()
        var_center = g.var(ddof=0)

        unit_entries = list(unit_map)
        mu_confusion = mu_center[unit_entries]
        var_confusion = var_center[unit_entries]

        flat_resp = pd.DataFrame({'mu_resp': mu_confusion.stack(), 'var_resp': var_confusion.stack()})
        flat_resp = flat_resp.reset_index().merge(rep_count.reset_index(), on=['image_class', 'unit_id'])

        class_map = {
            'multi_cnn_mei': 'cnn',
            'multi_lin_rf': 'linear'
        }
        flat_resp['model_type'] = flat_resp.image_class.map(class_map)

        for k, v in key.items():
            flat_resp[k] = v

        col_map = {
            'unit_id': 'image_unit_id',
            'matched_unit': 'matched_unit_id',
        }

        flat_resp = flat_resp.rename(col_map, axis=1)

        self.UnitActivation.insert(flat_resp.to_records(), ignore_extra_fields=True)

@schema
class SummaryConfusion(dj.Computed):
    definition = """
    loop_group: int   # loop group
    ---
    scan_count: int   # number of scans belonging to this group
    """

    @property
    def key_source(self):
        return dj.U('loop_group') & (ClosedLoopScan & NormalizedConfusion) & 'loop_group > 0'

    class UnitActivation(dj.Part):
        definition = """
        -> master
        -> multi_mei.ModelType
        image_unit_id: int   # image source unit id
        matched_unit_id: int # unit id of the matched neuron
        ---
        mu_resp: float      # average response across repeats
        sem_resp: float     # sem of response across repeats
        n_repeats: int      # number of repeats
        """

    def make(self, key):
        scans = (NormalizedConfusion & (ClosedLoopScan & {**key, 'stim_type': 'mei'})).fetch('KEY')
        unit_key = dict(key)
        key['scan_count'] = len(scans)
        self.insert1(key)

        df = pd.DataFrame((NormalizedConfusion.UnitActivation & scans).fetch())

        df['total_resp'] = df['mu_resp'] * df['n_repeats']
        df['total_x2'] = df['var_resp'] * df['n_repeats'] + (df['mu_resp'] ** 2) * df['n_repeats']
        summary = df.groupby(['model_type', 'image_unit_id', 'matched_unit_id'])[
            'n_repeats', 'total_resp', 'total_x2'].sum()

        summary['mu_resp'] = summary['total_resp'] / summary['n_repeats']
        summary['var_resp'] = (summary['total_x2'] - (summary['mu_resp'] ** 2) * summary['n_repeats']) / (
                    summary['n_repeats'] - 1)
        if any(summary['var_resp'] < 0):
            warnings.warn('Some numerical instability for zero variance units')
            summary['var_resp'] += 1e-12 # most of the time this is just a very small negative number
        summary['sem_resp'] = np.sqrt(summary['var_resp'] / summary['n_repeats'])

        summary = summary.reset_index()

        for k, v in unit_key.items():
            summary[k] = v

        self.UnitActivation.insert(summary.to_records(), ignore_extra_fields=True)
#
@schema
class SummaryConfusionDiagonalEntry(dj.Computed):
    definition = """
    -> SummaryConfusion
    matched_unit_id: int    # unit id of the matched neuron
    ---
    cnn_resp: float      # average response across repeats of CNN
    cnn_sem: float       # sem over CNN resp
    rf_resp: float       # average response across repeats of RF
    rf_sem:  float       # sem over RF resp
    diff_resp: float     # diff in response
    diff_rank: int       # rank of difference with 0 = largest CNN - RF
    t_diff: float        # Welch's t stat of difference
    diff_sem: float      # adjusted difference SEM based on Welch's t stat denominator
    t_rank: int          # rank of t-score based difference with 0 = largest t_diff
    """

    key_source = SummaryConfusion()

    def make(self, key):
        ua = SummaryConfusion.UnitActivation & key & 'image_unit_id = matched_unit_id'

        cnn_ua = (ua & 'model_type="cnn"').proj(t1='model_type', cnn_resp='mu_resp', cnn_sem='sem_resp')
        rf_ua = (ua & 'model_type="linear"').proj(t2='model_type', rf_resp='mu_resp', rf_sem='sem_resp')

        df = pd.DataFrame((cnn_ua * rf_ua).fetch())

        df['diff_resp'] = df['cnn_resp'] - df['rf_resp']

        df['diff_sem'] = np.sqrt(df['cnn_sem']**2 + df['rf_sem']**2)
        df['t_diff'] = df['diff_resp'] / df['diff_sem']

        df.sort_values(by='diff_resp', ascending=False, inplace=True)
        df['diff_rank'] = np.arange(len(df))

        df.sort_values(by='t_diff', ascending=False, inplace=True)
        df['t_rank'] = np.arange(len(df))

        self.insert(df.to_records(), ignore_extra_fields=True)
#

@schema
class ZScoreConfusion(dj.Computed):
    definition = """
    -> ClosedLoopScan
    """

    class UnitActivation(dj.Part):
        definition = """
        -> master
        -> multi_mei.ModelType
        image_unit_id: int   # image source unit id
        matched_unit_id: int # unit id of the matched neuron
        ---
        mu_resp: float      # average response across repeats in z-score
        var_resp: float     # var of response across repeats (ddof=0)
        n_repeats: int      # number of repeats
        """

    @property
    def key_source(self):
        return ClosedLoopScan & 'loop_group > 0 and stim_type="mei"'

    def make(self, key):

        self.insert1(key)

        # find MEI source scan
        src_scan = (ClosedLoopScan & (
                dj.U('loop_group') & (ClosedLoopScan & key)) & 'mei_source=True').fetch1('KEY')
        src_key = {'src_' + k: v for k, v in src_scan.items()}

        # find all units matching the source units
        src_units, target_units = (BestProximityCellMatch & src_key & key).fetch('src_unit_id', 'unit_id')
        unit_map = OrderedDict(zip(src_units, target_units))

        # load the data
        dataset = InputResponse().fetch1_data(key)

        # turn into a data frame
        df = DataFrame(dataset['responses'], columns=dataset['neurons']['unit_ids'])
        # normalize the response
        df = (df - df.mean(axis=0)) / df.std(axis=0)
        # convert into non binary string
        df['condition_hash'] = [x.decode() for x in list(dataset['condition_hashes'])]
        # assign repetition count
        df['rep_count'] = df.groupby(by='condition_hash').cumcount()
        # uniquely index the rows
        df = df.set_index(['condition_hash', 'rep_count'], verify_integrity=True)

        # map columns into target units
        df = pd.concat([df[v].rename(k) for k, v in unit_map.items()], axis=1)

        # get information about each stimulus condition
        deltas = multi_mei.CorrectedHighUnitSelection().proj(delta_fraction='cnn_fraction_oracle - lin_fraction_oracle')
        df_cond = DataFrame(
            (stimulus.Frame() * stimulus.StaticImage.MultiMEI * multi_mei.TargetDataset.Unit * deltas).proj(
                'condition_hash', 'image_class', 'unit_id', 'x_shift', 'y_shift', 'delta_fraction').fetch(
                order_by='image_class, unit_id, x_shift, y_shift'))
        df_cond = df_cond[['condition_hash', 'image_class', 'unit_id', 'x_shift', 'y_shift', 'delta_fraction']]
        df_cond = df_cond.set_index(['condition_hash'])

        # add condition information into the response data
        df_all = df.join(df_cond)

        df_all = df_all.reset_index()
        df_all = df_all.set_index(['image_class', 'unit_id', 'x_shift', 'y_shift', 'rep_count'])
        df_all.sort_index(axis=0, inplace=True)
        df_all = df_all.reindex(unit_map, level='unit_id')

        # restrict to center responses
        df_center = df_all[
            (df_all.index.get_level_values('x_shift') == 0) & (df_all.index.get_level_values('y_shift') == 0)]

        df_center.columns.name = 'matched_unit'
        hashes = df_center['condition_hash']

        # get repetition counts
        rep_count = hashes.reset_index().groupby(['image_class', 'unit_id'], sort=False).condition_hash.count()
        rep_count.name = 'n_repeats'
        rep_count = pd.DataFrame(rep_count)

        # compute mean and uncorrected variance
        unit_entries = list(unit_map)
        df_center = df_center[unit_entries]
        g = df_center.reset_index().groupby(['image_class', 'unit_id'], sort=False)

        mu_center = g.mean()
        var_center = g.var(ddof=0)

        unit_entries = list(unit_map)
        mu_confusion = mu_center[unit_entries]
        var_confusion = var_center[unit_entries]

        flat_resp = pd.DataFrame({'mu_resp': mu_confusion.stack(), 'var_resp': var_confusion.stack()})
        flat_resp = flat_resp.reset_index().merge(rep_count.reset_index(), on=['image_class', 'unit_id'])

        class_map = {
            'multi_cnn_mei': 'cnn',
            'multi_lin_rf': 'linear'
        }
        flat_resp['model_type'] = flat_resp.image_class.map(class_map)

        for k, v in key.items():
            flat_resp[k] = v

        col_map = {
            'unit_id': 'image_unit_id',
            'matched_unit': 'matched_unit_id',
        }

        flat_resp = flat_resp.rename(col_map, axis=1)

        self.UnitActivation.insert(flat_resp.to_records(), ignore_extra_fields=True)

@schema
class ZSummaryConfusion(dj.Computed):
    definition = """
    loop_group: int   # loop group
    ---
    scan_count: int   # number of scans belonging to this group
    """

    @property
    def key_source(self):
        return dj.U('loop_group') & (ClosedLoopScan & ZScoreConfusion) & 'loop_group > 0'

    class UnitActivation(dj.Part):
        definition = """
        -> master
        -> multi_mei.ModelType
        image_unit_id: int   # image source unit id
        matched_unit_id: int # unit id of the matched neuron
        ---
        mu_resp: float      # average response across repeats
        sem_resp: float     # sem of response across repeats
        n_repeats: int      # number of repeats
        """

    def make(self, key):
        scans = (ZScoreConfusion & (ClosedLoopScan & {**key, 'stim_type': 'mei'})).fetch('KEY')
        unit_key = dict(key)
        key['scan_count'] = len(scans)
        self.insert1(key)

        df = pd.DataFrame((ZScoreConfusion.UnitActivation & scans).fetch())

        df['total_resp'] = df['mu_resp'] * df['n_repeats']
        df['total_x2'] = df['var_resp'] * df['n_repeats'] + (df['mu_resp']**2) * df['n_repeats']
        summary = df.groupby(['model_type', 'image_unit_id', 'matched_unit_id'])[
            'n_repeats', 'total_resp', 'total_x2'].sum()

        summary['mu_resp'] = summary['total_resp'] / summary['n_repeats']
        summary['var_resp'] = (summary['total_x2'] - (summary['mu_resp']**2) * summary['n_repeats']) / (summary['n_repeats'] - 1)
        if any(summary['var_resp'] < 0):
            warnings.warn('Some numerical instability for zero variance units')
            summary['var_resp'] += 1e-12 # most of the time this is just a very small negative number
        summary['sem_resp'] = np.sqrt(summary['var_resp'] / summary['n_repeats'])

        summary = summary.reset_index()

        for k, v in unit_key.items():
            summary[k] = v

        self.UnitActivation.insert(summary.to_records(), ignore_extra_fields=True)
#
@schema
class ZSummaryConfusionDiagonalEntry(dj.Computed):
    definition = """
    -> ZSummaryConfusion
    matched_unit_id: int    # unit id of the matched neuron
    ---
    cnn_resp: float      # average response across repeats of CNN
    cnn_sem: float       # sem over CNN resp
    rf_resp: float       # average response across repeats of RF
    rf_sem:  float       # sem over RF resp
    diff_resp: float     # diff in response
    diff_rank: int       # rank of difference with 0 = largest CNN - RF
    t_diff: float        # Welch's t stat of difference
    diff_sem: float      # adjusted difference SEM based on Welch's t stat denominator
    t_rank: int          # rank of t-score based difference with 0 = largest t_diff
    """

    key_source = ZSummaryConfusion()

    def make(self, key):
        ua = ZSummaryConfusion.UnitActivation & key & 'image_unit_id = matched_unit_id'

        cnn_ua = (ua & 'model_type="cnn"').proj(t1='model_type', cnn_resp='mu_resp', cnn_sem='sem_resp')
        rf_ua = (ua & 'model_type="linear"').proj(t2='model_type', rf_resp='mu_resp', rf_sem='sem_resp')

        df = pd.DataFrame((cnn_ua * rf_ua).fetch())

        df['diff_resp'] = df['cnn_resp'] - df['rf_resp']

        df['diff_sem'] = np.sqrt(df['cnn_sem']**2 + df['rf_sem']**2)
        df['t_diff'] = df['diff_resp'] / df['diff_sem']

        df.sort_values(by='diff_resp', ascending=False, inplace=True)
        df['diff_rank'] = np.arange(len(df))

        df.sort_values(by='t_diff', ascending=False, inplace=True)
        df['t_rank'] = np.arange(len(df))

        self.insert(df.to_records(), ignore_extra_fields=True)
#

@schema
class CellMatchOracleCorrelation(dj.Computed):
    definition = """
    -> ClosedLoopScan
    ---
    units: longblob  # unit ids
    corr_matrix: longblob  # correlation matrix - rows are source and columns are targets (matched)
    """
    @property
    def key_source(self):
        return ClosedLoopScan & BestProximityCellMatch & 'stim_type="imagenet" AND day > 1 AND loop_group > 0'

    def make(self, key):
        src_key = (ClosedLoopScan & (dj.U('loop_group') & (ClosedLoopScan() & key)) & 'mei_source = True').fetch1('KEY')

        ## Get source scan info
        # load the data
        dataset = InputResponse().fetch1_data({**src_key, 'preproc_id': 0})
        target_units = (multi_mei.TargetDataset.Unit & (
                    stimulus.StaticImage.MultiMEI * data_schemas.StaticMultiDataset.Member & src_key)).fetch('unit_id',
                                                                                                             order_by='unit_id')

        # turn into a data frame
        df = DataFrame(dataset['responses'], columns=dataset['neurons']['unit_ids'])
        # normalize the response
        df = (df - df.mean(axis=0))/ df.std(axis=0)
        # convert into non binary string
        df['condition_hash'] = [x.decode() for x in list(dataset['condition_hashes'])]
        # assign repetition count
        df['rep_count'] = df.groupby(by='condition_hash').cumcount()
        condition_counts = df['condition_hash'].value_counts()
        condition_counts.name = 'counts'
        condition_counts = pd.DataFrame(condition_counts)
        condition_counts.index.name = 'condition_hash'
        # uniquely index the rows
        df = df.set_index(['condition_hash', 'rep_count'], verify_integrity=True)

        # get information about each stimulus condition
        df_cond = DataFrame(
            (stimulus.Frame() * stimulus.StaticImage.ImageNet).proj(
                'condition_hash', 'image_class', 'image_id').fetch(
                order_by='image_class, image_id'))
        df_cond = df_cond[['condition_hash', 'image_class', 'image_id']]
        df_cond = df_cond.set_index(['condition_hash'])

        # add condition information into the response data
        df_all = df.join(df_cond).join(condition_counts)

        # select condition with repeats
        df_all = df_all[df_all['counts'] > 4]

        df_all = df_all.reset_index()
        mu_resp = df_all.groupby('image_id').mean()

        mu_resp = mu_resp[target_units]
        mu_resp.columns.name = 'unit_id'
        src_resp = mu_resp.stack()
        src_resp.name = 'src_resp'

        ## Get matched unit responses
        # find all units matching the source units
        src_units, target_units = (BestProximityCellMatch & key).fetch('src_unit_id', 'unit_id', order_by='src_unit_id')
        unit_map = OrderedDict(zip(src_units, target_units))

        # load the data
        dataset = InputResponse().fetch1_data({**key, 'preproc_id': 0})

        # turn into a data frame
        df = DataFrame(dataset['responses'], columns=dataset['neurons']['unit_ids'])
        # normalize the response
        df = (df - df.mean(axis=0)) / df.std(axis=0)
        # convert into non binary string
        df['condition_hash'] = [x.decode() for x in list(dataset['condition_hashes'])]
        # assign repetition count
        df['rep_count'] = df.groupby(by='condition_hash').cumcount()
        condition_counts = df['condition_hash'].value_counts()
        condition_counts.name = 'counts'
        condition_counts = pd.DataFrame(condition_counts)
        condition_counts.index.name = 'condition_hash'
        # uniquely index the rows
        df = df.set_index(['condition_hash', 'rep_count'], verify_integrity=True)

        # get information about each stimulus condition
        df_cond = DataFrame(
            (stimulus.Frame() * stimulus.StaticImage.ImageNet).proj(
                'condition_hash', 'image_class', 'image_id').fetch(
                order_by='image_class, image_id'))
        df_cond = df_cond[['condition_hash', 'image_class', 'image_id']]
        df_cond = df_cond.set_index(['condition_hash'])

        # add condition information into the response data
        df_all = df.join(df_cond).join(condition_counts)

        # select condition with repeats
        df_all = df_all[df_all['counts'] > 4]

        df_all = df_all.reset_index()
        mu_resp = df_all.groupby('image_id').mean()

        # map columns into target units
        mu_resp = pd.concat([mu_resp[v].rename(k) for k, v in unit_map.items()], axis=1)

        # mu_resp = mu_resp[target_units]
        mu_resp.columns.name = 'unit_id'
        target_resp = mu_resp.stack()
        target_resp.name = 'target_resp'

        combined = pd.concat([src_resp, target_resp], axis=1)

        units = combined.reset_index()['unit_id']
        key['units'] = units.values

        src_mat = combined.unstack()['src_resp'].values
        target_mat = combined.unstack()['target_resp'].values

        n = src_mat.shape[0]
        r_src = (src_mat - src_mat.mean(axis=0)) / src_mat.std(axis=0) / np.sqrt(n)
        r_target = (target_mat - target_mat.mean(axis=0)) / target_mat.std(axis=0) / np.sqrt(n)

        corr = r_src.T @ r_target

        key['corr_matrix'] = corr

        self.insert1(key)




#
#
# @schema
# class CellMatch(dj.Manual):
#     definition = """
#     # cell matches across scans
#     -> meso.ScanSet.Unit.proj(src_sesion='session', src_scan_idx='scan_idx', src_unit_id='unit_id')
#     -> meso.ScanSet.Unit
#     ---
#     match_priority: int  # number to indicate the quality of the match. Smaller number the better (min=0)
#     src_field: int # field of the source cell
#     field: int     # field of the matched cell
#     proximity_match: tinyint   # whether matched by proximity
#     structural_match: tinyint  # whether matched by structural identity
#     match_note: varchar(255) # notes about the match
#     """
