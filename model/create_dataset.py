import numpy as np
import torch
from data_processing import *
from collections import Counter

def create_dataset_multiple_cultivars_phenologies(args):
    embedding_x_train_list, embedding_y_train_list, embedding_x_test_list, embedding_y_test_list, embedding_cultivar_label_train_list, embedding_cultivar_label_test_list = list(), list(), list(), list(), list(), list()
    season_max_lens = list()
    appended_labels_cultdict = dict()
    true_collection = dict() #pair the cultivar name with the season name into a tuple, and match it to the temp season

    for cf in args.proper_cultivars: #for every cultivar (that we told the program to run on)
        df = args.cultivar_file_dict[cf] #get dataframe for that cultivar

        for column_name in ['MEAN_AT', 'MIN_AT', 'AVG_AT', 'MAX_AT']:
            df[column_name] = df[column_name].replace(-100, np.nan)


        season_label = df['SEASON'].to_numpy()
        temp_season=list()
        seasons=list()

        appended_labels = list() #literally just a list of the names of the seasons that are used
        '''
        ex:
            ['1993-1994', '1994-1995', '1996-1997', '1997-1998', '2000-2001', '2001-2002', '2002-2003', '2003-2004', '2004-2005', '2005-2006', '2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2020-2021', '2022-2023', '2023-2024']
        '''

        for idx in range(len(season_label)-1): #want to assign each index in the df to its season
            temp_season.append(idx)
            if season_label[idx] != season_label[idx+1]: #at season boundaries append the season
                if season_label[idx] != '2007-2008': #remove 2007-2008 season, too much missing
                    if df['PHEN_PROGRESS'][temp_season[0]] != -1: # remove seasons without all required phenologies
                        seasons.append(temp_season) #at a good boundary, push the list of indices to the seasons list
                        true_collection[(cf, df['SEASON'][idx])] = temp_season
                        appended_labels.append(df['SEASON'][idx]) #record the actual season label in this list for some reason
                    else:
                        pass
                temp_season = list() #always clear the temp list after each boundary, whether we threw the indices out or kept them

        # push the final season (which has no ending border with the next season)
        last_index = len(season_label) - 1
        temp_season.append(last_index) #append final index
        if season_label[last_index] != '2007-2008': #remove 2007-2008 season, too much missing
            if df['PHEN_PROGRESS'][temp_season[0]] != -1: # remove seasons without all required phenologies (check the first day of the current season)
                seasons.append(temp_season)
                true_collection[(cf, df['SEASON'][last_index])] = temp_season #pair the cultivar name with the season name into a tuple, and match it to the temp season
                appended_labels.append(df['SEASON'][last_index])

        season_lens = [len(season) for season in seasons]
        season_max_lens.append(max(season_lens))
        appended_labels_cultdict[cf] = appended_labels

    train_full_labels = list()
    test_full_labels = list()

    args.season_max_len = max(season_max_lens)
    for cultivar_idx, cultivar in enumerate(args.cultivar_list):
        x_train, y_train, x_test, y_test, cultivar_label_train, cultivar_label_test, train_season_labels, test_season_labels = data_processing_simpler_phenologies(cultivar, cultivar_idx, args, true_collection)

        df = args.cultivar_file_dict[cultivar] #get the data for this particular cultivar

        for s in range(len(train_season_labels)):
            label = train_season_labels[s]
            this_season = df[df['SEASON'] == label]
            phen_progress = this_season['PHEN_PROGRESS'].to_numpy()
            phen_size = phen_progress.shape[0]
            for i in range(phen_size):
                if phen_progress[phen_size - i - 1] == np.float64(-1.0) or np.isnan(phen_progress[phen_size - i - 1]):
                    y_train[s, phen_size - i - 1, 0] = np.nan
                else:
                    break

        for s in range(len(test_season_labels)):
            label = test_season_labels[s]
            this_season = df[df['SEASON'] == label]
            phen_progress = this_season['PHEN_PROGRESS'].to_numpy()
            phen_size = phen_progress.shape[0]
            for i in range(phen_size):
                if phen_progress[phen_size - i - 1] == np.float64(-1.0) or np.isnan(phen_progress[phen_size - i - 1]):
                    y_test[s, phen_size - i - 1, 0] = np.nan
                else:
                    break

        embedding_x_train_list.append(x_train)
        embedding_x_test_list.append(x_test)
        embedding_y_train_list.append(y_train)
        embedding_y_test_list.append(y_test)
        embedding_cultivar_label_train_list.append(cultivar_label_train)
        embedding_cultivar_label_test_list.append(cultivar_label_test)

        train_full_labels.extend(map(lambda x: (cultivar, x), train_season_labels))
        test_full_labels.extend(map(lambda x: (cultivar, x), test_season_labels))

    train_dataset = {
        'x': torch.Tensor(np.concatenate(embedding_x_train_list)),
        'y': torch.Tensor(np.concatenate(embedding_y_train_list)),
        'cultivar_id': torch.squeeze(torch.Tensor(np.concatenate(embedding_cultivar_label_train_list)).long()),
        'labels': train_full_labels,
    }
    test_dataset = {
        'x': torch.Tensor(np.concatenate(embedding_x_test_list)),
        'y': torch.Tensor(np.concatenate(embedding_y_test_list)),
        'cultivar_id': torch.squeeze(torch.Tensor(np.concatenate(embedding_cultivar_label_test_list)).long()),
        'labels': test_full_labels,
    }
    cultivar_id_arr = train_dataset['cultivar_id']
    train_freq = dict(Counter(cultivar_id_arr[:,0].numpy()))
    train_freq = {key:1/value for key, value in train_freq.items()}
    train_freq_sum = sum(train_freq.values())
    train_freq = {key:value/train_freq_sum for key, value in train_freq.items()}
    train_freq_array = torch.Tensor([train_freq[key] for key in train_dataset['cultivar_id'][:,0].numpy()])
    train_freq_array = train_freq_array.unsqueeze(dim=1).repeat((1,cultivar_id_arr.shape[1]))
    test_freq_array = torch.zeros_like(test_dataset['cultivar_id'])
    train_dataset.update({'freq':train_freq_array})
    test_dataset.update({'freq':test_freq_array})

    return {'train':train_dataset, 'test':test_dataset}


