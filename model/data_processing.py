import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset

import glob
import torch.nn as nn
import os
from pathlib import Path
import random

from global_vars import debugging_dict, season_collect



class MyDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return self.data_dict['x'].shape[0]

    def __getitem__(self, idx):
        return self.data_dict['x'][idx], self.data_dict['y'][idx], self.data_dict['cultivar_id'][idx], self.data_dict['freq'][idx]


class MyDatasetPhen(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return self.data_dict['x'].shape[0]

    def __getitem__(self, idx):
        return self.data_dict['x'][idx], self.data_dict['y'][idx], self.data_dict['cultivar_id'][idx], self.data_dict['labels'][idx]


def linear_interp(x, y, missing_indicator, show=False):
    y = np.array(y)
    if (not np.isnan(missing_indicator)):
        missing = np.where(y == missing_indicator)[0]
        not_missing = np.where(y != missing_indicator)[0]
    else:
        # special case for nan values
        missing = np.argwhere(np.isnan(y)).flatten()
        all_idx = np.arange(0, y.shape[0])
        not_missing = np.setdiff1d(all_idx, missing)

    interp = np.interp(x, not_missing, y[not_missing])

    if show == True:
        plt.figure(figsize=(16, 9))
        plt.title("Linear Interp. result where missing = " +
                  str(missing_indicator) + "  Values replaced: " + str(len(missing)))
        plt.plot(x, interp)

    return interp


def split_and_normalize_phenologies(_df, season_max_length, seasons, features,label, phenos_list = [], x_min = 0.0, x_max = 1.0, experiment=None, x_mean = None):
    x = []
    y = []
    for i, season in enumerate(seasons):
        #
        _x = (_df[features].loc[season, :]).to_numpy()

        _x = np.concatenate(
            (_x, np.zeros((season_max_length - len(season), len(features)))), axis=0)

        add_array = np.zeros((season_max_length - len(season), len(label)))
        add_array[:] = np.nan

        _y = _df.loc[season, :][label].to_numpy()
        _y = np.concatenate((_y, add_array), axis=0)

        x.append(_x)
        y.append(_y)

    x = np.array(x)
    y = np.array(y)

    norm_features_idx = np.arange(0, len(features))

    normalized = (x[:, :, norm_features_idx] - x_min) / (x_max - x_min)  # normalize
    x[:, :, norm_features_idx] = normalized

    return x, y


# this is the function where the train/test split actually is decided
def data_processing_simpler_phenologies(cultivar_name, cultivar_idx, args, true_collection):
    """
    RETURN:
        x_train,
        y_train,
        x_test,
        y_test,
        cultivar_label_train,
        cultivar_label_test,
    """

    df = args.cultivar_file_dict[cultivar_name] #get the df for this cultivar

    for column_name in ['MEAN_AT', 'MIN_AT', 'AVG_AT', 'MAX_AT']:
        df[column_name] = df[column_name].replace(-100, np.nan)

    total_idx = np.arange(df.shape[0]) #create a vector with ints from 0..df.shape[0]
    dormant_label = df['DORMANT_SEASON'].to_numpy() #just a giant vector of 0's and 1's
    first_dormant = np.where(dormant_label==1)[0][0]
    relevant_idx = total_idx[first_dormant:]
    dormant_seasons = [df[(df['SEASON']==season_name) & (df['DORMANT_SEASON']==1)].index.to_list() for season_name in list(df['SEASON'].unique())]
    dormant_seasons = [x for x in dormant_seasons if len(x)>0]
    temp_season=list()
    seasons=list()
    for idx in relevant_idx[:-1]:
        temp_season.append(idx)
        if dormant_label[idx]==0 and dormant_label[idx+1]==1:
            if df['SEASON'][idx] != '2007-2008': #remove 2007-2008 season, too much missing
                seasons.append(temp_season)
            if df['PHEN_PROGRESS'][idx] != -1: # remove seasons without all required phenologies
                seasons.append(temp_season)
            temp_season=list()

    #add the last season
    if seasons[-1][0]!=temp_season[0]: #the last season has no end boundary so append the season now if valid
        seasons.append(temp_season)

    #add last index of last
    if seasons[-1][-1]!=relevant_idx[-1]:
       seasons[-1].append(relevant_idx[-1])

    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB")
    #print(true_collection.keys())
    #amontonado = list()
    #print(true_collection)

    '''not sure why we have to filter here, we should have only processed for this cultivar anyway'''
    filtered = list(map(lambda x: (x[0][1], x[1]), filter(lambda x: x[0][0] == cultivar_name, true_collection.items())))
    filtered_seasons = list()
    filtered_labels = list()
    for x, y in filtered:
        filtered_labels.append(x)
        filtered_seasons.append(y)
    valid_seasons = filtered_seasons
    valid_seasons_idx = [x for season in valid_seasons for x in season]

    # NOTE AT THIS POINT:
    # valid_seasons contains a list of lists, where each inner list is a set of indices
    # and each list of indices is one season, and they're all contained here together
    # Also note that this function depends on the cultivar name itself, so it selects training
    # for each cultivar, which means a certain year, say 2018-2019, will be used for training for ALL cultivars

    season_max_length = args.season_max_len
    no_of_seasons = len(valid_seasons)

    # here's the part where we select the seasons
    cur_trial = int(args.trial[-1])
    if no_of_seasons >= 18: #not every cultivar is gonna have enough to do this, but the four main ones definitely will
        test_idx = [cur_trial*2, cur_trial*2 + 1]
    else: #all the rest can just have their training seasons repeated
        cur_trial = cur_trial % 3
        cur_trial_str = f"trial_{cur_trial}"
        if cur_trial_str == 'trial_0':
            test_idx = list([0,1]) #first two are used for testing
        elif cur_trial_str == 'trial_1':
            test_idx = list([(no_of_seasons // 2) - 1,(no_of_seasons // 2)]) #middle two
        elif cur_trial_str == 'trial_2':
            test_idx = list([no_of_seasons - 2, no_of_seasons - 1]) #last two
        elif cur_trial_str == 'random':
            test_idx = [random.randrange(no_of_seasons)]
        else:
            test_idx = [int(cur_trial_str)]
    train_seasons = list()
    test_seasons = list()
    train_season_labels = list()
    test_season_labels = list()

    for season_idx, season in enumerate(valid_seasons):
        season_collect.append(f"{cultivar_name}: {df['SEASON'][season[0]]}")
        if season_idx in test_idx:
            test_seasons.append(season)
            test_season_labels.append(filtered_labels[season_idx])
            print("Test:", df['SEASON'][season[0]])
        else:
            train_seasons.append(season)
            train_season_labels.append(filtered_labels[season_idx])
            print("Train:", df['SEASON'][season[0]])

    # here we shove all the seasons together into one big mono list
    valid_idx_train = [x for season in train_seasons for x in season]


    if args.training == True:
        x_min = df[args.features].iloc[valid_idx_train].min().to_numpy() #not sure why this has to be numpy but okay
        x_max = df[args.features].iloc[valid_idx_train].max().to_numpy()
        minmax = np.vstack((x_min, x_max))
        np.savetxt(args.output_folder + "/minmax.csv", minmax, delimiter = ",")
    else:
        minmax = np.genfromtxt(args.output_folder + "/minmax.csv", delimiter = ",")
        x_min = minmax[0, :]
        x_max = minmax[1, :]

    debugging_dict["pre_x"] = df[args.features].iloc[valid_seasons_idx] 


    #do interpolation AFTER you calculate the mean/std
    for feature_col in args.features:  # remove nan and do linear interp.
        df[feature_col] = df[feature_col].replace(np.nan, -100)
        df[feature_col] = linear_interp(np.arange(df.shape[0]), df[feature_col], -100, False)
        if df[feature_col].isna().sum() != 0:
            assert False

    x_train, y_train = split_and_normalize_phenologies(
        df, season_max_length, train_seasons, args.features, args.label, args.phenos_list, x_min, x_max, args.experiment)

    x_test, y_test = split_and_normalize_phenologies(
        df, season_max_length, test_seasons, args.features, args.label, args.phenos_list, x_min, x_max, args.experiment)


    cultivar_label_train = torch.ones(
        (x_train.shape[0], x_train.shape[1], 1))*cultivar_idx
    cultivar_label_test = torch.ones(
        (x_test.shape[0], x_test.shape[1], 1))*cultivar_idx
    return x_train, y_train, x_test, y_test, cultivar_label_train, cultivar_label_test, train_season_labels, test_season_labels



