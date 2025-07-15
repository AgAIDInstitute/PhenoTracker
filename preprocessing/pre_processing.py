"""Main preprocessing file for the weather data to feed into the model."""

import os
from functools import reduce

import numpy as np
import pandas as pd
from pandas.core.generic import pickle



phenologies_selector = ["Budburst/Budbreak", "Full Bloom", "Veraison 50%"]


# generate the chilling temps for the season
def gen_chilling_temps(max_season_temps): #returns the chilling temps
    window_size = 7 #size of window to look for dips
    deviation_cap = 10 #how far can values deviate from dip center

    # built through the loop
    lowest = max_season_temps[0]
    chill_temps = []

    if len(max_season_temps) < window_size: #season is smaller than window size
        # pad out start
        for d in range(len(max_season_temps)):
            chill_temps.append(lowest)
    else:
        # pad out start
        for d in range(window_size):
            chill_temps.append(lowest)

        # find rest of values
        for d in range(window_size, len(max_season_temps)): #iterate through every day accounting for window size
            center = d - int(window_size / 2)

            if max_season_temps[center] >= chill_temps[d-1]: #propagate same value again
                chill_temps.append(chill_temps[-1])
                continue

            scan_range = max_season_temps[d-window_size+1:d+1]
            invalids = list(filter(lambda x: abs(x - max_season_temps[center]) > deviation_cap, scan_range))

            if len(invalids) == 0: #we have a valid scan
                chill_temps.append(max_season_temps[center])
            else: #invalid scan, propagate value
                chill_temps.append(chill_temps[-1])

    return chill_temps



def gen_progress(phenology_season): #returns progress
    """
    Generate the phenology progress and the stages.
    """

    progress = []
    stages = []

    try:
        phen_indices = [np.where(phenology_season == x)[0][0] for x in phenologies_selector]
        if len(phen_indices) != len(phenologies_selector):
            print("NOT ENOUGH PHENOLOGIES HERE")
    except:
        return None #something went wrong collecting the phenologies
    print(phen_indices)
    phen_indices.insert(0, 0)

    # push the progress values into the collected list
    stage_length_limit = 2
    for i in range(1, len(phen_indices)):
        cur_index = phen_indices[i]
        pre_index = phen_indices[i-1]
        phen_stage_len = cur_index - pre_index
        if phen_stage_len <= stage_length_limit:
            return None

        for j in range(phen_stage_len):
            progress.append(j / (phen_stage_len - 1))
            stages.append(i - 1)

    diff_size = len(phenology_season) - len(progress) #there will be a difference because the last phen stage is not the end of the season
    final_stage = len(phen_indices) - 1 #forgot we push 0 as the first one lol, yeah that explains everything
    for _ in range(diff_size):
        progress.append(-1)
        stages.append(final_stage)

    # updated stage vector concept
    stage_vectors = []
    for stage in stages:
        vect = [1.0 if x == stage else 0.0 for x in range(len(phenologies_selector)+1)]
        stage_vectors.append(vect)

    return progress, stages, stage_vectors


# data to extract and use
use_features = [
    "DATE",
    "SEASON",
    "DORMANT_SEASON",
    "PHENOLOGY",
    "PREDICTED_LTE10",
    "PREDICTED_LTE50",
    "PREDICTED_LTE90",

    # weather data
    'MEAN_AT', #air temperature
    'MIN_AT',
    'AVG_AT',
    'MAX_AT',
    'MIN_RH', #relative humidity
    'AVG_RH',
    'MAX_RH',
    'MIN_DEWPT',
    'AVG_DEWPT',
    'MAX_DEWPT',
    'P_INCHES',  # precipitation
    'WS_MPH',  # wind speed. if no sensor then value will be na
    'MAX_WS_MPH',

    # specific to the api
    #'MEAN_AT',
    #'AVG_RH',
    #'P_INCHES',
    #'WS_MPH',
    'WD_DEGREE',
    'LW_UNITY',
    #'SR_MJM2',
    'ST2',
    'ST8',
    'SM8_PCNT',
    'MSLP_HPA',
]


# features that will be shoved out
out_features = [
    "DATE",
    "DORMANT_SEASON", #the model requires this
    "PHENOLOGY",
    "PREDICTED_LTE10",
    "PREDICTED_LTE50",
    "PREDICTED_LTE90",

    # weather data
    'MEAN_AT', #air temperature
    'MIN_AT',
    'AVG_AT', #the model requires this
    'MAX_AT',
    'MIN_RH', #relative humidity
    'AVG_RH',
    'MAX_RH',
    'MIN_DEWPT',
    'AVG_DEWPT',
    'MAX_DEWPT',
    'P_INCHES',  # precipitation
    'WS_MPH',  # wind speed. if no sensor then value will be na
    'MAX_WS_MPH',

    # specific to the api
    #'MEAN_AT',
    #'AVG_RH',
    #'P_INCHES',
    #'WS_MPH',
    'WD_DEGREE',
    'LW_UNITY',
    #'SR_MJM2',
    'ST2',
    'ST8',
    'SM8_PCNT',
    'MSLP_HPA',
]


grape_files = filter(lambda x: x.startswith("ColdHardiness_Grape_Prosser_"), os.listdir("./input/weather"))

data_dict = {}

should_skip_cultivars = []

# process each cultivar
for file in grape_files:
    df = pd.read_csv("input/weather/" + file)
    print("reading: " + file)
    pre = df[use_features].to_numpy()

    s_names = pre[:, 1] #all season names (for this cultivar)
    _, idx = np.unique(s_names, return_index = True) #get the indices of the unique season names
    s_names_unique = s_names[np.sort(idx)] #get the unique season names in order
    phenologies = pre[:, 3] #get all phenologies for this cultivar

    # built during the next loop
    base_seasons = [] #list of seasons with all the phenologies we want (for this cultivar)
    base_seasons_indices = []
    mehr_seasons = [] #the next season for each of the previous seasons
    mehr_seasons_indices = []

    # iterate through all seasons, and their consecutive season
    for n in range(len(s_names_unique) - 1):
        s_name_this = s_names_unique[n]
        s_name_next = s_names_unique[n+1]
        season_indices_this = np.where(s_names == s_name_this)[0] #get this season indices
        season_indices_next = np.where(s_names == s_name_next)[0] #get next season indices

        base_seasons.append(s_name_this)
        base_seasons_indices.append(season_indices_this)
        mehr_seasons.append(s_name_next)
        mehr_seasons_indices.append(season_indices_next)

    dormant_season = pre[:, 2] #get all dormant season markers for this cultivar
    not_dormant_indices = np.where(dormant_season == 0)
    dormant_indices = np.where(dormant_season == 1)
    max_temps = pre[:, 9] #get the max temps

    # built during the next loop
    season_labels = [] #will be converted into a pandas column
    all_season_indices = [] #used to filter out rows
    all_chilling_temps = [] #conglomerate all the chilling temps for this cultivar
    all_phen_progress = []
    all_stages = []
    all_stage_vectors = []
    all_counters = [] #just keeps track of what day of the season we're in

    # notes, for any given season, the dormancy season is at the end and non-dormancy season is at the start

    # go through all collected seasons and generate columns
    for i in range(len(base_seasons)):
        base_name = base_seasons[i] #grab the dormancy season
        base_indices = base_seasons_indices[i]
        mehr_name = mehr_seasons[i] #grab the non-dormancy season
        mehr_indices = mehr_seasons_indices[i]

        # align the new indices based on the dormancy season
        base_half_indices = np.intersect1d(base_indices, dormant_indices)
        mehr_half_indices = np.intersect1d(mehr_indices, not_dormant_indices)
        new_season_indices = np.append(base_half_indices, mehr_half_indices) #IMPORTANT, we know that sept 7 is the first day of a new dormancy season
        new_season_indices += 24 #align first day to october 1st (september has 30 days)

        # now filter based on the present phenologies
        phenologies_unique_new = set(filter(lambda x: isinstance(x, str), np.take(phenologies, new_season_indices))) #unique phenologies for the new season
        if not all([x in phenologies_unique_new for x in phenologies_selector]): #if this season doesn't contain all required phenologies
            continue

        # chilling temps
        chilling_temps = gen_chilling_temps(np.take(max_temps, new_season_indices))

        # phenology progress
        gen_output = gen_progress(np.take(phenologies, new_season_indices))
        if gen_output == None: #if phenology things are missing, then skip them
            continue

        phenology_progress, stages, stage_vectors = gen_output

        # generate the counter
        season_counter = [i for i in range(1, len(phenology_progress) + 1)]

        # assuming everything went right, append these details too
        season_labels.extend([base_name] * new_season_indices.shape[0])
        all_season_indices.extend(new_season_indices)
        all_chilling_temps.extend(chilling_temps)
        all_phen_progress.extend(phenology_progress)
        all_stages.extend(stages)
        all_stage_vectors.extend(stage_vectors)
        all_counters.extend(season_counter)

    # ensure theres at least 3 seasons per cultivar (enough for at least one training season + 2 testing)
    cultivar = file[file.rfind("_")+1: file.find(".")]
    if len(set(season_labels)) < 4:
        print("Skip this cultivar")
        should_skip_cultivars.append(cultivar)

    # transform the dimensionality with numpy
    stage_vecs = np.array(all_stage_vectors).transpose() #one column per stage

    post_full = df[out_features]
    post = pd.DataFrame.take(post_full, all_season_indices, axis = 0)
    post.insert(0, "SEASON", season_labels)
    post.insert(2, "PHEN_PROGRESS", all_phen_progress)
    post.insert(3, "STAGE", all_stages)
    for i in range(stage_vecs.shape[0]):
        post.insert(i+4, f"STAGE_V_{i}", stage_vecs[i])
    post.insert(0, "COUNTER", all_counters)
    post.to_csv("output/progress/" + file)

    data_dict[cultivar] = post

with open("output/weather_dict.pkl", "wb") as file:
    pickle.dump(data_dict, file)

print("SKIP THESE CULTIVARS:")
for c in should_skip_cultivars:
    print(c)












