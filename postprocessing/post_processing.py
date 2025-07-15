"""
current monolithic post processing file
"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from functools import reduce
import pickle
import pandas as pd

from datetime import date, datetime, timedelta
from pathlib import Path

import math

import plotly.graph_objects as go
import altair as alt

import os
from PIL import Image




# UNIVERSAL
enable_printing = {
    "Progress": True,
    "Stages": True,
    "Event Violins": False, #deprecated
    "Weekly Event Violins": False, #deprecated
    "Scatterplots": True,
    "ROCs": True,
    "Prediction Timeline": True,
    "Prediction Timeline Errors": True,
}
good_cultivars = ["Cabernet Sauvignon", "Chardonnay", "Merlot", "Riesling"]
events = ["Budbreak", "Full Bloom", "Veraison 50%"]
model_colors = {"Parker": "crimson", "PhenoTracker": "forestgreen", "Zapata": "blue"}



def calculate_MAE(y: npt.NDArray[np.float64], pred_y: npt.NDArray[np.float64]) -> float:
    """Calculates the MAE for two numpy arrays."""
    # sum(|y_pred - y_true|) / n
    subtract = 0
    total = 0.0
    for i in range(len(y)):
        if np.isnan(pred_y[i]) or np.isnan(y[i]):
            subtract += 1
            continue
        total += abs(pred_y[i] - y[i])
    return total / (len(y) - subtract)



def calculate_RMSE(xs: npt.NDArray[np.float64]) -> float:
    """Calculates the RMSE for two one numpy diff array."""
    # sqrt(sum(diff**2) / n)
    squared = list(map(lambda x: x**2, filter(lambda x: not np.isnan(x), xs)))
    return np.sqrt(sum(squared) / len(squared))



def find_decision_boundaries(x: npt.NDArray[np.float64]) -> tuple[list[int], list[int]]:
    """Take in the stage prediction vectors for a season and find the boundaries."""
    # NOTE: x is a 2d array where each row has the same length as the number of stages
    stages = [] #the actual stage at each index
    boundaries = [] #the locations of all the boundaries (when the progress should be 1.0 and the day before a new stage)
    current_stage = 0 #the current stage, will never decrease and used to index each row in x

    for i, row in enumerate(x):
        if current_stage + 1 >= len(row): #we've exhausted all the stages, but we're not at the end of the list yet so the rest of the days will be this current stage
            while len(stages) < len(x):
                stages.append(current_stage) #fill out the rest of the days with this stage
            return stages, boundaries

        stages.append(current_stage) #append on every iteration, before incrementing the current stage if that'll happen today

        if row[current_stage+1] > row[current_stage]: #if the stage is going to switch over tomorrrow
            boundaries.append(i) #today is the last day of the current stage
            current_stage += 1

    return stages, boundaries



def generate_graphs_csvs():
    """Main monolithic function for reading output data and generating graphs and csvs."""
    # load input files
    with open("input/data_output.pkl", "rb") as file:
        """
        list[(cultivar: str, season: str, true: np.array(1, 366, 5), pred: np.array(1, 366, 5))]
        """
        phenotracker_data_raw: dict = pickle.load(file)

    phenotracker_data_train = list(map(lambda x: (x[0], x[1], x[2].reshape((366, 5)).transpose(), np.array([x[3], x[4], x[5], x[6], x[7]])), phenotracker_data_raw["training"]))
    phenotracker_data_valid = list(map(lambda x: (x[0], x[1], x[2].reshape((366, 5)).transpose(), np.array([x[3], x[4], x[5], x[6], x[7]])), phenotracker_data_raw["validation"]))

    """
    for true and pred, first is phenology progress and then each stage (one more stage than events, aka for e events there are e+1 stages)
    """
    phenotracker_data = dict() #size 120, none of the 3 trials intersect (20 cultivars * 2 training seasons * 3 trials)

    event_csv_rows: list = [["Cultivar", "Season", "Phenology", "Median", "25th", "75th", "% Below Median"]]
    weeks_csv_rows: list = [["Cultivar", "Season", "Phenology", "Week", "Median", "25th", "75th", "% Below Median"]]
    range_csv_rows: list = [["Cultivar", "Phenology", "Rounded Range", "True Range", "Rank Required", "Accuracy"]]
    ranks_csv_rows: list = [["Cultivar", "Season", "Phenology", "Rank", "Range", "Most Voted", "Truth", "Contains Truth"]]

    max_rank_search = 8









    # TRAINING, preprocess the training results first to generate such things as ranks
    season_counts = dict()
    cultivar_ranges = dict() #used for calculating optimal ranges for cultivars, might be deprecated
    true_dates_cultivars = dict() #used to store the true dates for each cultivar for later use (seasons are in order as received)
    true_dates_seasons = dict() #used to store true dates for cultivar season pairs
    for i, (cultivar, season, true, pred) in enumerate(phenotracker_data_train):

        # no repeats
        if (cultivar, season) in season_counts:
            season_counts[(cultivar, season)] += 1
            continue

        season_counts[(cultivar, season)] = 0

        # set a filter for only the important cultivars
        if cultivar in good_cultivars:
            pass
        else:
            continue

        print(f"{i+1}/{len(phenotracker_data_train)}: {cultivar}, {season}")

        phenotracker_data[(season, cultivar)] = (true, pred)
        stage_rows_true = true[1:].transpose() #just grab the stages and transform them for decision boundary reasons
        stage_rows_pred = pred[1:].transpose()

        stages_true, boundaries_true = find_decision_boundaries(stage_rows_true) #will be used for graphing the predicted stages
        stages_pred, boundaries_pred = find_decision_boundaries(stage_rows_pred) #will be used to calculate the predicted event dates

        true_dates_seasons[(cultivar, int(season[:4]))] = boundaries_true #IMPORTANT FOR COMPARING AGAINST OTHER MODELS

        boundaries_pred1 = [-1] #for organization purposes
        boundaries_pred1.extend(boundaries_pred)

        # calculate the previous event boundaries (so that we can predict the dates in combination with the progress)
        previous_boundary = []
        for i in range(len(pred[1])):
            for e in boundaries_pred1[::-1]:
                if i > e:
                    previous_boundary.append(e)
                    break

        # convert previous event boundaries into a predicted date by using the phenology progress
        predicted_dates = []
        for i, z in enumerate(zip(previous_boundary, pred[0])):
            prev_boundary, pred_prog = z
            days_since = i - prev_boundary #days since last boundary

            stage_length = int(days_since / float(pred_prog))
            pred_date = stage_length + prev_boundary
            predicted_dates.append(pred_date)

        # now split up this data for each individual event
        grouped_dates = [[] for _ in range(true.shape[0] - 1)] #one empty list for each stage
        for d, s in zip(predicted_dates, stages_pred):
            grouped_dates[s].append(d)

        # find the highest voted index, the next highest voted index, etc, continued in holistic section
        this_cultivar_ranks = []
        for i, group in enumerate(grouped_dates[:-1]):
            counts = dict()
            for x in group:
                if x in counts:
                    counts[x] += 1
                else:
                    counts[x] = 1
            sorted_counts = sorted(counts.items(), key = lambda x: x[1], reverse = True)

            true_day = boundaries_true[i]
            most_voted = sorted_counts[0][0]
            contained = [(False, 0)] #contain one by default for the loop to grab from
            for j in range(1, max_rank_search+1): #will expand up to the fifth largest voted day
                grabbed_day = sorted_counts[j][0] #grab the jth most voted day
                diff = abs(most_voted - grabbed_day) #find the range from the peak to the current ranking
                if most_voted - diff <= true_day and true_day <= most_voted + diff: #the true day is in the range
                    contained.append((True, diff))
                else:
                    contained.append((False or contained[-1][0], diff))
            contained = contained[1:] #chop off the default initial value
            this_cultivar_ranks.append(contained)
        cultivar_ranges[(cultivar, season)] = this_cultivar_ranks
        if cultivar in true_dates_cultivars:
            true_dates_cultivars[cultivar].append(boundaries_true)
        else:
            true_dates_cultivars[cultivar] = [boundaries_true]




    # change the format of the true_dates to be python dates
    true_dates_datetimes = {}
    for (cultivar, season), dates in true_dates_seasons.items():
        base_date = date(season, 10, 1) #october 1st of this year is the start
        adjusted_dates = list(map(lambda x: base_date + timedelta(x), dates))
        true_dates_datetimes[(cultivar, season)] = adjusted_dates

    # sort the true dates to equate the cultivar ranges
    true_dates_events = dict()
    for cultivar, v in true_dates_cultivars.items():
        collected = [[], [], []]
        for e in v:
            collected[0].append(e[0])
            collected[1].append(e[1])
            collected[2].append(e[2])
        true_dates_events[cultivar] = collected

    # for each cultivar, get the range results, sorted into the three events
    sorted_cultivar_ranges = dict()
    for k, v in cultivar_ranges.items():
        cultivar = k[0]
        if k[0] in sorted_cultivar_ranges:
            sorted_cultivar_ranges[k[0]][0].append(v[0])
            sorted_cultivar_ranges[k[0]][1].append(v[1])
            sorted_cultivar_ranges[k[0]][2].append(v[2])
        else:
            sorted_cultivar_ranges[k[0]] = [[v[0]], [v[1]], [v[2]]]

    range_required = dict() #for each cultivar, show three events and how many ranks were required for each season within those three events: (rank-2 (or -1 if failed), percentage, range aka radius)
    rank_required = dict()
    for cultivar, v in sorted_cultivar_ranges.items():
        cur_ranges = []
        this_cultivar_ranks = []
        for ie, e in enumerate(v): #look at each individual event for this cultivar

            # want to find the average range for each rank for this cultivar
            rank_averages = [sum(map(lambda x: x[i][1], e)) / len(e) for i in range(max_rank_search)]
            
            # want to find what percentage of each rank contained the true dates (aka was true)
            rank_percentages = [len(list(filter(None , map(lambda x: x[i][0], e)))) / len(e) for i in range(max_rank_search)]

            # find the first rank with 50% or more, and grab that average range size
            for i in range(max_rank_search):
                if rank_percentages[i] > 0.5:
                    cur_ranges.append((i, rank_percentages[i], rank_averages[i])) #NOTE, THIS i IS TWO LESS THAN REAL ANSWER
                    break
            else:
                print("Could not reach 50% for " + cultivar + " " + events[ie])
                exit()

            cur_rank_required = cur_ranges[-1][0] + 2
            accuracy = cur_ranges[-1][1]
            true_range = cur_ranges[-1][2] * 2
            rounded_range = math.ceil(true_range / 2) * 2 #this rounds to the next multiple of 2 (aka next even)
            range_csv_rows.append([cultivar, events[ie], rounded_range, true_range, cur_rank_required, accuracy]) #append to csv
            this_cultivar_ranks.append(cur_rank_required)

        range_required[cultivar] = cur_ranges
        rank_required[cultivar] = this_cultivar_ranks

    range_df = pd.DataFrame(range_csv_rows)
    range_df.to_csv("output_graphs/cultivar_ranks_training.csv")









    # VALIDATION, handle every single season individually
    print("VALIDATION")
    phenotracker_predicted_dates_seasons = dict() #storing the information gathered from phenotracker validation
    phenotracker_predicted_dates_cultivars = dict()
    for i, (cultivar, season, true, pred) in enumerate(phenotracker_data_valid):

        # set a filter for only the important cultivars
        if cultivar in good_cultivars:
            pass
        else:
            continue

        print(f"{i+1}/{len(phenotracker_data_valid)}: {cultivar}, {season}")

        actual_season_length = np.logical_not(np.isnan(true[0])).sum() #finds how many nans there are and use it to calculate the actual length of the season
        season = int(season[:4])
        phenotracker_data[(season, cultivar)] = (true, pred)
        stage_rows_true = true[1:].transpose() #just grab the stages and transform them for decision boundary reasons
        stage_rows_pred = pred[1:].transpose()

        stages_true, boundaries_true = find_decision_boundaries(stage_rows_true) #will be used for graphing the predicted stages
        stages_pred, boundaries_pred = find_decision_boundaries(stage_rows_pred) #will be used to calculate the predicted event dates

        boundaries_pred1 = [-1] #for organization purposes
        boundaries_pred1.extend(boundaries_pred)

        # calculate the previous event boundaries (so that we can predict the dates in combination with the progress)
        previous_boundary = []
        for i in range(len(pred[1])):
            for e in boundaries_pred1[::-1]:
                if i > e:
                    previous_boundary.append(e)
                    break

        # convert previous event boundaries into a predicted date by using the phenology progress
        predicted_dates = []
        for i, z in enumerate(zip(previous_boundary, pred[0])):
            prev_boundary, pred_prog = z
            days_since = i - prev_boundary #days since last boundary

            stage_length = int(days_since / float(pred_prog))
            pred_date = stage_length + prev_boundary
            predicted_dates.append(pred_date)

        # now split up this data for each individual event
        grouped_dates = [[] for _ in range(true.shape[0] - 1)] #one empty list for each stage
        for d, s in zip(predicted_dates, stages_pred):
            grouped_dates[s].append(d)

        # print prediction timeline for phenotracker
        if enable_printing["Prediction Timeline"]:
            max_deviation = 400
            neg_deviation = 0
            Path("./output_graphs/prediction_timeline").mkdir(parents=True, exist_ok=True) #ensure this folder exists
            plt.close()
            plt.clf()
            plt.figure(figsize = (6.4, 4.8), dpi = 100)
            plt.title(f"Prediction Timeline {cultivar} {season}")
            plt.xlabel("Date of Prediction")
            plt.ylabel("Predicted Date")
            accum_len = 0
            for i, group in enumerate(grouped_dates[:-1]):
                cleaned_top = list(map(lambda x: None if x > max_deviation else x, group))
                cleaned_bot = list(map(lambda x: None if x < neg_deviation else x, group))
                cleaned = list(map(lambda x: None if x > max_deviation or x < neg_deviation else x, group))

                top_xs = map(lambda x: x[0], filter(lambda x: x[1] == None, zip(range(len(cleaned_top)), cleaned_top)))
                bot_xs = map(lambda x: x[0], filter(lambda x: x[1] == None, zip(range(len(cleaned_bot)), cleaned_bot)))

                true_date = boundaries_true[i]

                # graph out of bounds
                for x in top_xs:
                    plt.scatter(accum_len + x, max_deviation, color = "red", s = 100, marker = "|")
                for x in bot_xs:
                    plt.scatter(accum_len + x, neg_deviation, color = "red", s = 100, marker = "|")

                # graph points
                plt.scatter(range(accum_len, accum_len + len(group)), cleaned, label=events[i])

                plt.axhline(y=true_date, color='black', linestyle='--', linewidth=1)

                accum_len += len(group)


            plt.legend(loc = "lower right")
            plt.savefig(f"output_graphs/prediction_timeline/prediction_timeline_{cultivar}_{season}.png")

        # store this for predicted timeline error, continued in holistic section
        if cultivar in phenotracker_predicted_dates_cultivars:
            phenotracker_predicted_dates_cultivars[cultivar].append((boundaries_true, grouped_dates))
        else:
            phenotracker_predicted_dates_cultivars[cultivar] = [(boundaries_true, grouped_dates)]

        # now split up each event into 4 week intervals
        weeked_dates = [[[], [], [], []] for _ in range(len(grouped_dates))]
        for j, e in enumerate(grouped_dates): #for each event
            event_length = len(e) #used for math things
            for i in range(4): #for each potential week
                ri = 3 - i
                if event_length >= (ri+1) * 7: #enough days for this week
                    weeked_dates[j][i].extend(e[event_length - (ri+1)*7 : event_length - ri*7]) #collect this particular week

        # now go back in and convert both grouped_dates and weeked_dates into differences rather than date predictions
        grouped_diffs = []
        for i, ds in enumerate(grouped_dates[:-1]): #should be three groups
            grouped_diffs.append(list(map(lambda x: x - boundaries_true[i], ds)))

        weeked_diffs = []
        for i, ws in enumerate(weeked_dates[:-1]): #ws will bring up the four weeks for this event, exclude that last one just like for grouped_diffs
            boundary = boundaries_true[i]
            this_event_diffs = []
            for j, w in enumerate(ws): #each week individually
                this_event_diffs.append(list(map(lambda x: x - boundary, w)))
            weeked_diffs.append(this_event_diffs)

        # find the highest voted index, the next highest voted index, etc, continued in holistic section, AND graph the predictions
        for i, group in enumerate(grouped_dates[:-1]):
            counts = dict()
            for x in group:
                if x in counts:
                    counts[x] += 1
                else:
                    counts[x] = 1
            sorted_counts = sorted(counts.items(), key = lambda x: x[1], reverse = True)

            true_day = boundaries_true[i] #true day for this event
            most_voted = sorted_counts[0][0] #most voted for this event

            use_rank = rank_required[cultivar][i] - 1 #decrease by 1 to compensate correctly
            grabbed_day = sorted_counts[use_rank][0] #grab the day for this rank, and remove the vote count
            cur_range = abs(grabbed_day - most_voted)
            contains = (most_voted - cur_range) <= true_day and true_day <= (most_voted + cur_range)

            '''ranks_csv_rows: list = [["Cultivar", "Season", "Phenology", "Rank", "Range", "Most Voted", "Truth", "Contains Truth",]]'''
            ranks_csv_rows.append((cultivar, season, events[i], use_rank+1, cur_range*2, most_voted, true_day, contains))
            this_tuple = ("Roza.2", season, cultivar, "PhenoTracker")
            if this_tuple in phenotracker_predicted_dates_seasons:
                phenotracker_predicted_dates_seasons[this_tuple][f"{events[i]} Most Voted"] = most_voted
                phenotracker_predicted_dates_seasons[this_tuple][f"{events[i]} Range"] = cur_range*2
            else:
                phenotracker_predicted_dates_seasons[this_tuple] = {f"{events[i]} Most Voted": most_voted, f"{events[i]} Range": cur_range*2}


        # print phenology progress
        print_length = len(true[0]) #print_length is used for graph printing bounds on the x axis
        Path("./output_graphs/progress").mkdir(parents=True, exist_ok=True) #ensure this folder exists
        if enable_printing["Progress"]:
            plt.close()
            plt.clf()
            plt.figure(figsize = (6.4, 4.8), dpi = 100)
            plt.title(f"Phenology Progress for {cultivar} {season}")
            plt.xlabel("Days Since October 1st")
            plt.ylabel("Percentage")
            plt.plot(range(print_length), true[0][:print_length], label = "True")
            plt.plot(range(print_length), pred[0][:print_length], label = "Predicted")
            plt.legend(loc = "upper left")
            plt.savefig(f"output_graphs/progress/progress_{cultivar}_{season}.png")

        # print stages
        print_length = len(true[0])
        Path("./output_graphs/stages").mkdir(parents=True, exist_ok=True) #ensure this folder exists
        if enable_printing["Stages"]:
            plt.close()
            plt.clf()
            plt.figure(figsize = (6.4, 4.8), dpi = 100)
            plt.title(f"Stage Probabilities for {cultivar} {season}")
            plt.xlabel("Days Since October 1st")
            plt.ylabel("Probability")
            plt.axvline(boundaries_true[0], color = "red", label = "True Event Dates") #just label one of them since they're all the same color
            plt.axvline(boundaries_true[1], color = "red")
            plt.axvline(boundaries_true[2], color = "red")
            plt.plot(range(print_length), pred[1][:print_length], label = "Budbreak")
            plt.plot(range(print_length), pred[2][:print_length], label = "Full Bloom")
            plt.plot(range(print_length), pred[3][:print_length], label = "Veraison 50%")
            plt.legend(loc = "center left")
            plt.savefig(f"output_graphs/stages/stages_{cultivar}_{season}.png")

        # combine phenology progress and stages
        print_length = len(true[0])
        Path("./output_graphs/progress_stage_combined").mkdir(parents=True, exist_ok=True) #ensure this folder exists
        if enable_printing["Progress"] and enable_printing["Stages"]:
            from PIL import Image

            # Load the image
            left = Image.open(f"output_graphs/progress/progress_{cultivar}_{season}.png")
            right = Image.open(f"output_graphs/stages/stages_{cultivar}_{season}.png")

            # Get original dimensions
            width, height = left.size

            # Create a new blank image with double height
            combined = Image.new('RGB', (width*2, height))

            # Paste the original image twice (top and bottom)
            combined.paste(left, (0, 0))         # top
            combined.paste(right, (width, 0))    # bottom

            # Save or show
            combined.save(f"output_graphs/progress_stage_combined/progress_stage_{cultivar}_{season}.png")


        # print violins for all events, and generate the csv
        print_length = len(true[0])
        Path("./output_graphs/event_violins").mkdir(parents=True, exist_ok=True) #ensure this folder exists
        violin_ys = [0.2, 0.5, 0.8] #three violins
        violin_height = 0.05 #actually half of the size
        colors = ["blue", "orange", "green", "red"] #BOTTOM TO TOP
        if enable_printing["Event Violins"]:
            plt.close()
            plt.clf()
            plt.figure(figsize = (6.4, 4.8), dpi = 100)
            plt.axvline(0, color = "black", linewidth = 0.5) #easier to see 0
        
        for i, e in enumerate(grouped_diffs): #go through each event
            ea = np.array(e)
            ea = ea[np.abs(ea) < 50.0] #remove outliers more than 50 off
            ea_so = ea.copy()
            ea_so.sort()
            q25 = ea_so[int(len(ea_so) * 0.25)]
            q75 = ea_so[int(len(ea_so) * 0.75)]
            if len(set(ea)) == 1:
                ea = np.concatenate((ea, np.array([ea[0] - 0.02, ea[0] + 0.02, ea[0] + 0.02, ea[0] - 0.02])))
            if enable_printing["Event Violins"]:
                plt.violinplot(ea, positions = [0.5 * i], orientation = "horizontal", showextrema = False) #TODO: if all numbers are the same, no violin is displayed

            # add to the csv list
            median_idx = len(ea_so) // 2
            median = ea_so[median_idx]
            skew = median_idx / len(ea_so)
            event_csv_rows.append([cultivar, season, events[i], median, q25, q75, skew])

        if enable_printing["Event Violins"]:
            plt.title(f"Event Predictions for {cultivar} {season}")
            plt.xlabel("Error (in Days) and Quartiles")
            plt.ylabel("     Budbreak         Full Bloom     Veraison 50%")
            plt.yticks([])
            plt.savefig(f"output_graphs/event_violins/event_violins_{cultivar}_{season}.png")

        # print violins for all weeks of each event, and generate the csv
        print_length = len(true[0])
        Path("./output_graphs/event_violins_weekly").mkdir(parents=True, exist_ok=True) #ensure this folder exists
        violin_ys = [0.16, 0.39, 0.61, 0.84] #four violins
        violin_height = 0.05 #actually half of the size
        colors = ["blue", "orange", "green", "red"] #BOTTOM TO TOP
        for i, e in enumerate(weeked_diffs): #go through each event
            if enable_printing["Weekly Event Violins"]:
                plt.close()
                plt.clf()
                plt.figure(figsize = (6.4, 4.8), dpi = 100)
                plt.axvline(0, color = "black", linewidth = 0.5) #easier to see 0

            for j, w in enumerate(e): #go through each week
                wa = np.array(w)
                wa = wa[np.abs(wa) < 50.0] #remove outliers
                wa_so = wa.copy()
                wa_so.sort()
                q25 = wa_so[int(len(wa_so) * 0.25)]
                q75 = wa_so[int(len(wa_so) * 0.75)]
                if len(set(wa)) == 1: #if all the points are exactly the same, then make a few modifications or else no violin will be displayed
                    wa = np.concatenate((wa, np.array([wa[0] - 0.02, wa[0] + 0.02, wa[0] + 0.02, wa[0] - 0.02])))
                if enable_printing["Weekly Event Violins"]:
                    plt.axvline(q25, color = colors[j], ymin = violin_ys[j] - violin_height, ymax = violin_ys[j] + violin_height)
                    plt.axvline(q75, color = colors[j], ymin = violin_ys[j] - violin_height, ymax = violin_ys[j] + violin_height)
                    plt.violinplot(wa, positions = [0.5 * j], orientation = "horizontal", showextrema = False)

                # add to the csv list
                median_idx = len(wa_so) // 2
                median = wa_so[median_idx]
                skew = median_idx / len(wa_so)
                weeks_csv_rows.append([cultivar, season, events[i], -j, median, q25, q75, skew])

            if enable_printing["Weekly Event Violins"]:
                plt.title(f"Weekly Event Predictions for {cultivar} {season} ({events[i]})")
                plt.xlabel("Error (in Days) and Quartiles")
                plt.ylabel("Weeks Before Event\n  4 - 3           3 - 2           2 - 1           1 - 0")
                plt.yticks([])
                plt.savefig(f"output_graphs/event_violins_weekly/event_violins_weekly_{cultivar}_{season}_{events[i]}.png")






    # HOLISTIC SECTION

    event_df = pd.DataFrame(event_csv_rows)
    weeks_df = pd.DataFrame(weeks_csv_rows)
    event_df.to_csv("output_graphs/cultivar_events.csv")
    weeks_df.to_csv("output_graphs/cultivar_events_weekly.csv")

    ranks_df = pd.DataFrame(ranks_csv_rows)
    ranks_df.to_csv("output_graphs/cultivar_ranks_validation.csv")

    # print the percentage of trues in containing the true date in the validation data
    booled_ranks = list(map(lambda x: x[7], ranks_csv_rows[1:]))
    trues = list(filter(None, booled_ranks))
    print(f"Validation contained true: {len(trues)} / {len(booled_ranks)}: {len(trues) / len(booled_ranks)}")

    # graph the prediction timeline errors
    if enable_printing["Prediction Timeline Errors"]:
        max_deviation = 40

        Path("./output_graphs/prediction_timeline_errors").mkdir(parents=True, exist_ok=True) #ensure this folder exists
        for cultivar, block in phenotracker_predicted_dates_cultivars.items():
            # want to grab all seasons, but go through the three events iteratively
            use_colors = ["blue", "orange", "green"]
            for i in range(3):
                tuples = list(map(lambda x: (x[0][i], x[1][i]), block))
                errors = [list(map(lambda x: x - t[0], t[1])) for t in tuples]

                plt.close()
                plt.clf()
                #plt.figure(figsize = (6.4, 4.8), dpi = 100)
                plt.figure(figsize = (6.4, 2.4), dpi = 100)
                plt.title(f"Prediction Timeline Errors {cultivar}")
                plt.xlabel("bottom")
                plt.ylabel(events[i] + "\nError in Days")

                for err in errors:
                    clean_top = list(map(lambda x: None if x > max_deviation else x, err))
                    clean_bot = list(map(lambda x: None if x < 0-max_deviation else x, err))
                    clean = list(map(lambda x: None if abs(x) > max_deviation else x, err))

                    top_xs = map(lambda x: x[0], filter(lambda x: x[1] == None, zip(range(len(clean_top)), clean_top)))
                    bot_xs = map(lambda x: x[0], filter(lambda x: x[1] == None, zip(range(len(clean_bot)), clean_bot)))

                    # graph out of bounds
                    for x in top_xs:
                        plt.scatter(0 - x, max_deviation, color = "red", s = 10, marker = "|")
                    for x in bot_xs:
                        plt.scatter(0 - x, -max_deviation, color = "red", s = 10, marker = "|")

                    # graph points
                    plt.scatter(range(1 - len(clean), 1), clean, color = use_colors[i], s = 2)

                plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
                plt.xlim(left = -225, right = 3)

                plt.savefig(f"output_graphs/prediction_timeline_errors/prediction_timeline_errors_{cultivar}_{i}.png")
                

            # okay now let's combine all three images
            from PIL import Image


            ## Load the image
            image0 = Image.open(f"output_graphs/prediction_timeline_errors/prediction_timeline_errors_{cultivar}_0.png")
            image1 = Image.open(f"output_graphs/prediction_timeline_errors/prediction_timeline_errors_{cultivar}_1.png")
            image2 = Image.open(f"output_graphs/prediction_timeline_errors/prediction_timeline_errors_{cultivar}_2.png")

            crop_top_0 = 25        # crop top of bottom image  NOTE: 28 is a perfect shave
            crop_top_1 = 25        # crop top of middle image
            crop_bottom_1 = 25     # crop bottom of middle image
            crop_bottom_2 = 25     # crop bottom of top image

            width, height = image0.size  # assume all same size

            cropped_1 = image0.crop((0, crop_top_0, width, height))                     # bottom
            cropped_2 = image1.crop((0, crop_top_1, width, height - crop_bottom_1))     # middle
            cropped_3 = image2.crop((0, 0, width, height - crop_bottom_2))              # top

            total_height = cropped_1.height + cropped_2.height + cropped_3.height
            combined = Image.new("RGB", (width, total_height))

            y_offset = 0
            for img in [cropped_3, cropped_2, cropped_1]:
                combined.paste(img, (0, y_offset))
                y_offset += img.height

            ## Save or show
            combined.save(f"output_graphs/prediction_timeline_errors/prediction_timeline_errors_{cultivar}.png")

        # now combine all four

        image_paths = [f"output_graphs/prediction_timeline_errors/prediction_timeline_errors_{cultivar}.png" for cultivar in good_cultivars]
        crop_px = 0 #this amount to keep the labels

        from PIL import Image
        images = [Image.open(p) for p in image_paths]

        # Assume all images are the same size
        img_w, img_h = images[0].size

        # Crop all images equally on all sides
        def crop_uniform(img, crop):
            return img.crop((
                crop,           # left
                crop,           # top
                img_w - crop,   # right
                img_h - crop    # bottom
            ))

        cropped_images = [crop_uniform(img, crop_px) for img in images]

        # Get new dimensions
        new_w, new_h = cropped_images[0].size

        # Create a blank canvas for the 2x2 grid
        grid_img = Image.new('RGB', (new_w * 2, new_h * 2))

        # Paste into the grid
        grid_img.paste(cropped_images[0], (0, 0))
        grid_img.paste(cropped_images[1], (new_w, 0))
        grid_img.paste(cropped_images[2], (0, new_h))
        grid_img.paste(cropped_images[3], (new_w, new_h))

        # Save the final image
        grid_img.save("output_graphs/prediction_timeline_errors/prediction_timeline_errors_all.png")







    # OKAY NOW WE COMPARE AGAINST THE OTHER MODELS
    with open("input/other_models_dump.pkl", "rb") as file:
        other_models_dump: dict = pickle.load(file)
    all_models = other_models_dump | phenotracker_predicted_dates_seasons #slows us down but more convenient code

    models_diffs_csv_rows = []
    models_predictions = []
    range_contains_accum = {"Zapata": [[], [], [], [], [], []], "PhenoTracker": [[], [], [], [], [], []]}
    models_diffs_rmses = dict()
    for tup, info in all_models.items():
        (station, season, cultivar, model) = tup

        # fix some things
        if cultivar == "White Riesling":
            cultivar = "Riesling"
        if model in ["Zapata", "Parker"]:
            season = int(season) - 1
        else:
            season = int(season)

        true_dates = true_dates_datetimes[(cultivar, season)]


        match model:
            case "Zapata":
                predicted_dates = []
                predicted_dates.append(datetime.strptime(info["Bud Break Mean"], "%Y-%m-%d").date())
                predicted_dates.append(datetime.strptime(info["Full Bloom Mean"], "%Y-%m-%d").date())
                predicted_dates.append(datetime.strptime(info["Veraison Mean"], "%Y-%m-%d").date())
                predicted_dates.append(datetime.strptime(info["Bud Break Start"], "%Y-%m-%d").date())
                predicted_dates.append(datetime.strptime(info["Full Bloom Start"], "%Y-%m-%d").date())
                predicted_dates.append(datetime.strptime(info["Veraison Start"], "%Y-%m-%d").date())
                predicted_dates.append(datetime.strptime(info["Bud Break End"], "%Y-%m-%d").date())
                predicted_dates.append(datetime.strptime(info["Full Bloom End"], "%Y-%m-%d").date())
                predicted_dates.append(datetime.strptime(info["Veraison End"], "%Y-%m-%d").date())
                diffs = []
                diffs.append((predicted_dates[0] - true_dates[0]).days)
                diffs.append((predicted_dates[1] - true_dates[1]).days)
                diffs.append((predicted_dates[2] - true_dates[2]).days)
                diffs.append((predicted_dates[3] - true_dates[0]).days)
                diffs.append((predicted_dates[4] - true_dates[1]).days)
                diffs.append((predicted_dates[5] - true_dates[2]).days)
                diffs.append((predicted_dates[6] - true_dates[0]).days)
                diffs.append((predicted_dates[7] - true_dates[1]).days)
                diffs.append((predicted_dates[8] - true_dates[2]).days)

                range_contains_accum["Zapata"][0].append(predicted_dates[3] <= true_dates[0] <= predicted_dates[6])
                range_contains_accum["Zapata"][1].append(predicted_dates[4] <= true_dates[1] <= predicted_dates[7])
                range_contains_accum["Zapata"][2].append(predicted_dates[5] <= true_dates[2] <= predicted_dates[8])
                range_contains_accum["Zapata"][3].append((predicted_dates[6] - predicted_dates[3]).days)
                range_contains_accum["Zapata"][4].append((predicted_dates[7] - predicted_dates[4]).days)
                range_contains_accum["Zapata"][5].append((predicted_dates[8] - predicted_dates[5]).days)

                best_predictions = []
                best_diffs = []
                if abs(diffs[0]) <= abs(diffs[3]) and abs(diffs[0]) <= abs(diffs[6]):
                    best_budbreak = "Mean"
                    best_predictions.append(predicted_dates[0])
                    best_diffs.append(diffs[0])
                elif abs(diffs[3]) <= abs(diffs[0]) and abs(diffs[3]) <= abs(diffs[6]):
                    best_budbreak = "Start"
                    best_predictions.append(predicted_dates[3])
                    best_diffs.append(diffs[3])
                else:
                    best_budbreak = "End"
                    best_predictions.append(predicted_dates[6])
                    best_diffs.append(diffs[6])

                if abs(diffs[1]) <= abs(diffs[4]) and abs(diffs[1]) <= abs(diffs[7]):
                    best_bloom = "Mean"
                    best_predictions.append(predicted_dates[1])
                    best_diffs.append(diffs[1])
                elif abs(diffs[4]) <= abs(diffs[1]) and abs(diffs[4]) <= abs(diffs[7]):
                    best_bloom = "Start"
                    best_predictions.append(predicted_dates[4])
                    best_diffs.append(diffs[4])
                else:
                    best_bloom = "End"
                    best_predictions.append(predicted_dates[7])
                    best_diffs.append(diffs[7])

                if abs(diffs[2]) <= abs(diffs[5]) and abs(diffs[2]) <= abs(diffs[8]):
                    best_veraison = "Mean"
                    best_predictions.append(predicted_dates[2])
                    best_diffs.append(diffs[2])
                elif abs(diffs[5]) <= abs(diffs[2]) and abs(diffs[5]) <= abs(diffs[8]):
                    best_veraison = "Start"
                    best_predictions.append(predicted_dates[5])
                    best_diffs.append(diffs[5])
                else:
                    best_veraison = "End"
                    best_predictions.append(predicted_dates[8])
                    best_diffs.append(diffs[8])

                
                models_diffs_csv_rows.append([cultivar, season, events[0], model, diffs[0], predicted_dates[0], true_dates[0]])
                models_diffs_csv_rows.append([cultivar, season, events[1], model, diffs[1], predicted_dates[1], true_dates[1]])
                models_diffs_csv_rows.append([cultivar, season, events[2], model, diffs[2], predicted_dates[2], true_dates[2]])
                models_predictions.append((cultivar, season, events[0], model, predicted_dates[0]))
                models_predictions.append((cultivar, season, events[1], model, predicted_dates[1]))
                models_predictions.append((cultivar, season, events[2], model, predicted_dates[2]))

                for i in range(3):
                    rmse_tuple = (cultivar, events[i], model)
                    if rmse_tuple in models_diffs_rmses:
                        models_diffs_rmses[rmse_tuple].append(diffs[i])
                    else:
                        models_diffs_rmses[rmse_tuple] = [diffs[i]]
            case "Parker":
                predicted_dates = [None]
                predicted_dates.append(datetime.strptime(info["Full Bloom Start"], "%Y-%m-%d").date())
                predicted_dates.append(datetime.strptime(info["Veraison Start"], "%Y-%m-%d").date())
                diffs = [None]
                diffs.append((predicted_dates[1] - true_dates[1]).days)
                diffs.append((predicted_dates[2] - true_dates[2]).days)
                models_diffs_csv_rows.append([cultivar, season, events[1], model, diffs[1], predicted_dates[1], true_dates[1]])
                models_diffs_csv_rows.append([cultivar, season, events[2], model, diffs[2], predicted_dates[2], true_dates[2]])
                models_predictions.append((cultivar, season, events[1], model, predicted_dates[1]))
                models_predictions.append((cultivar, season, events[2], model, predicted_dates[2]))

                for i in range(1, 3):
                    rmse_tuple = (cultivar, events[i], model)
                    if rmse_tuple in models_diffs_rmses:
                        models_diffs_rmses[rmse_tuple].append(diffs[i])
                    else:
                        models_diffs_rmses[rmse_tuple] = [diffs[i]]
            case "PhenoTracker":
                base_date = date(season, 10, 1) #october 1st of this year is the start
                predicted_dates = []
                predicted_dates.append(base_date + timedelta(info["Budbreak Most Voted"]))
                predicted_dates.append(base_date + timedelta(info["Full Bloom Most Voted"]))
                predicted_dates.append(base_date + timedelta(info["Veraison 50% Most Voted"]))
                diffs = []
                diffs.append((predicted_dates[0] - true_dates[0]).days)
                diffs.append((predicted_dates[1] - true_dates[1]).days)
                diffs.append((predicted_dates[2] - true_dates[2]).days)
                models_diffs_csv_rows.append([cultivar, season, events[0], model, diffs[0], predicted_dates[0], true_dates[0]])
                models_diffs_csv_rows.append([cultivar, season, events[1], model, diffs[1], predicted_dates[1], true_dates[1]])
                models_diffs_csv_rows.append([cultivar, season, events[2], model, diffs[2], predicted_dates[2], true_dates[2]])
                models_predictions.append((cultivar, season, events[0], model, predicted_dates[0]))
                models_predictions.append((cultivar, season, events[1], model, predicted_dates[1]))
                models_predictions.append((cultivar, season, events[2], model, predicted_dates[2]))

                range_contains_accum["PhenoTracker"][0].append(diffs[0] <= info[f"{events[0]} Range"])
                range_contains_accum["PhenoTracker"][1].append(diffs[1] <= info[f"{events[1]} Range"])
                range_contains_accum["PhenoTracker"][2].append(diffs[2] <= info[f"{events[2]} Range"])
                range_contains_accum["PhenoTracker"][3].append(info[f"{events[0]} Range"])
                range_contains_accum["PhenoTracker"][4].append(info[f"{events[1]} Range"])
                range_contains_accum["PhenoTracker"][5].append(info[f"{events[2]} Range"])

                for i in range(3):
                    rmse_tuple = (cultivar, events[i], model)
                    if rmse_tuple in models_diffs_rmses:
                        models_diffs_rmses[rmse_tuple].append(diffs[i])
                    else:
                        models_diffs_rmses[rmse_tuple] = [diffs[i]]

    # calculate the number of times the range is sufficient for balcarcel and zapata (fake range)
    models_contains_csv = []
    for model, event_vals in range_contains_accum.items():
        models_contains_csv.append([model, events[0], len(list(filter(None, event_vals[0]))) / len(event_vals[0]), sum(event_vals[3]) / len(event_vals[3])])
        models_contains_csv.append([model, events[1], len(list(filter(None, event_vals[1]))) / len(event_vals[1]), sum(event_vals[4]) / len(event_vals[4])])
        models_contains_csv.append([model, events[2], len(list(filter(None, event_vals[2]))) / len(event_vals[2]), sum(event_vals[5]) / len(event_vals[5])])
    models_contains_csv.insert(0, ["Model", "Event", "Percent in Range", "Average Range Size"])
    pd.DataFrame(models_contains_csv).to_csv("output_graphs/models_contains.csv")

    models_diffs_csv_rows.sort(key = lambda x: x[2])
    models_diffs_csv_rows.sort(key = lambda x: x[1])
    models_diffs_csv_rows.sort(key = lambda x: x[0])
    models_diffs_csv_rows.insert(0, ["Cultivar", "Season", "Event", "Model", "Diff", "Predicted Day", "True Day"])
    diffs_df = pd.DataFrame(models_diffs_csv_rows)
    diffs_df.to_csv("output_graphs/models_diffs.csv")

    # sort the model predictions by cultivar, since we will make one scatterplot per cultivar
    models_predictions_by_cultivar = {x:[] for x in good_cultivars}
    for row in models_predictions:
        models_predictions_by_cultivar[row[0]].append(row[1:])

    # rmses
    models_rmses_csv = []
    for tup, lis in models_diffs_rmses.items():
        models_rmses_csv.append([tup[0], tup[1], tup[2], calculate_RMSE(lis)])
    models_rmses_csv.sort(key = lambda x: x[1])
    models_rmses_csv.sort(key = lambda x: x[0])
    models_rmses_csv.insert(0, ["Cultivar", "Event", "Model", "RMSE"])
    pd.DataFrame(models_rmses_csv).to_csv("output_graphs/models_diffs_rmses.csv")


    # make the scatterplots (one per cultivar)
    if enable_printing["Scatterplots"]:
        image_paths = []
        for cultivar, lists in models_predictions_by_cultivar.items():
            plt.close()
            plt.clf()
            plt.figure(figsize = (6.4, 6.4), dpi = 100)
            plt.xlabel("Predicted (DOY)")
            plt.ylabel("Observed (DOY)")
            max_ = 0
            min_ = 366
            xs = {x:[] for x in model_colors.keys()} #just a little cheat
            ys = {x:[] for x in model_colors.keys()}
            for (season, event, model, pred) in lists:
                event_i = events.index(event)
                true = true_dates_datetimes[(cultivar, season)][event_i].timetuple().tm_yday
                pred = pred.timetuple().tm_yday
                xs[model].append(pred)
                ys[model].append(true)
                if max(true, pred) > max_:
                    max_ = max(true, pred)
                if min(true, pred) < min_:
                    min_ = min(true, pred)
            for model, color in model_colors.items():
                plt.scatter(xs[model], ys[model], label=f'{model}', color=color)
            plt.plot([min_, max_], [min_, max_], color='black', linestyle='--', label='1:1')
            plt.axis("equal")
            plt.legend()
            plt.grid(True)
            plt.title(f"Model Predictions for {cultivar}")
            plt.savefig(f"output_graphs/scatterplots/scatterplot_{cultivar}.png")
            image_paths.append(f"output_graphs/scatterplots/scatterplot_{cultivar}.png")

        crop_px = 20 #this amount to keep the labels

        from PIL import Image
        images = [Image.open(p) for p in image_paths]

        # Assume all images are the same size
        img_w, img_h = images[0].size

        # Crop all images equally on all sides
        def crop_uniform(img, crop):
            return img.crop((
                crop,           # left
                crop,           # top
                img_w - crop,   # right
                img_h - crop    # bottom
            ))

        cropped_images = [crop_uniform(img, crop_px) for img in images]

        # Get new dimensions
        new_w, new_h = cropped_images[0].size

        # Create a blank canvas for the 2x2 grid
        grid_img = Image.new('RGB', (new_w * 2, new_h * 2))

        # Paste into the grid
        grid_img.paste(cropped_images[0], (0, 0))
        grid_img.paste(cropped_images[1], (new_w, 0))
        grid_img.paste(cropped_images[2], (0, new_h))
        grid_img.paste(cropped_images[3], (new_w, new_h))

        # Save the final image
        grid_img.save("output_graphs/scatterplots/scatterplot_all.png")



    # make the ROC curves (one per event, graphing diff on the bottom and compound the seasons up), one line per model
    # find the actual lowest mean diff per event
    event_groupings = {x:[] for x in events}
    event_model_diffs = {(e, m):[] for m in model_colors.keys() for e in events}
    event_model_diffs.pop(("Budbreak", "Parker")) #absolute diffs

    for (cultivar, season, event, model, pred) in models_predictions:
        diff = abs((pred - true_dates_datetimes[(cultivar, season)][events.index(event)]).days)
        event_groupings[event].append((cultivar, season, model, pred, diff))
        event_model_diffs[(event, model)].append(diff)

    # intermediate values
    event_model_diff_averages_ugly = dict(map(lambda x: (x[0], sum(x[1]) / len(x[1])), event_model_diffs.items()))
    diff_results = sorted(list(event_model_diff_averages_ugly.items()), key = lambda x: x[1])
    event_model_diff_averages = {x:[] for x in model_colors.keys()} #maybe redundant, tells us the average diffs for each model for each event, so we know how to order them on the graph


    sorted_event_groupings = dict(map(lambda x: (x[0], sorted(sorted(x[1], key = lambda x: x[0]), key = lambda x: x[1])), event_groupings.items())) #ensure all seasons and cultivars are sorted (season is major, cultivar is minor)
    for event in events:
        clean = list(map(lambda x: (x[0][1], x[1]), filter(lambda x: x[0][0] == event, diff_results)))
        cur_diff_averages = {x:0.0 for x in model_colors.keys()} #tells us how to sort the models visually on the graph
        for model, val in clean:
            event_model_diff_averages[model].append(val)
            cur_diff_averages[model] = val
        if event == "Budbreak":
            cur_diff_averages.pop("Parker")
        sorted_models_by_diff = sorted(cur_diff_averages.items(), key=lambda x: x[1]) #USED FOR DISPLAYING

        # now for this event we have the order of performance, now graph two things for each event (compound and non-compound)
        # use the sorted_event_groupings for access to all of this easily

        # each model is separated into its own basket, and only includes this particular event
        split_event_groupings = {x:[] for x in model_colors.keys()} #for this event, each model is assigned a list of all its diffs and preds, sorted majorly by sesaon and minorly by cultivar
        for (cultivar, season, model, pred, diff) in sorted_event_groupings[event]:
            split_event_groupings[model].append((cultivar, season, pred, diff))
            pass

        if enable_printing["ROCs"]:
            split_event_groupings_clean = dict(map(lambda x: (x[0], list(map(lambda x: x[3], x[1]))), split_event_groupings.items())) #each season's diffs, as sorted above
            if event == "Budbreak":
                split_event_groupings_clean.pop("Parker")
            print(split_event_groupings_clean)
            split_event_groupings_accum = dict() #each season's diffs compounded over time
            for model, lis in split_event_groupings_clean.items():
                running_total = 0
                new_counts = []
                for v in lis:
                    running_total += v
                    new_counts.append(running_total)
                split_event_groupings_accum[model] = new_counts
            print(split_event_groupings_accum)




def performance_graphs():
    """Print the performance graphs from the excel sheet generated in the above function."""

    # Load and clean data
    df = pd.read_csv('output_graphs/models_diffs.csv', header=0)
    df.columns = ['ID', 'Cultivar', 'Season', 'Event', 'Model', 'Diff', 'Predicted Day', 'True Day']
    df['Diff'] = pd.to_numeric(df['Diff'], errors='coerce')
    df.dropna(subset=['Diff'], inplace=True)
    df['Abs_Diff'] = df['Diff'].abs()

    # Create output directory if it doesn't exist
    os.makedirs("output_graphs/roc", exist_ok=True)

    # Set of all events
    events = ['Budbreak', 'Full Bloom', 'Veraison 50%']
    #custom_colors = ['blue', 'green', 'red', 'purple']
    custom_colors = ['green', 'blue', 'red', 'purple']

    for event in events:
        # Filter event-specific data
        event_df = df[df['Event'] == event].copy()

        # Special case: remove Parker model from Budbreak
        if event == 'Budbreak':
            event_df = event_df[event_df['Model'] != 'Parker']

        models = event_df['Model'].unique()
        performance_data = []

        for model in models:
            model_df = event_df[event_df['Model'] == model].copy()
            model_df = model_df.sort_values(by='Abs_Diff')
            num_observations = len(model_df)
            if num_observations > 0:
                model_df['Fraction_Solved'] = np.arange(1, num_observations + 1) / num_observations
                model_df['Model_Name'] = model
                performance_data.append(model_df[['Abs_Diff', 'Fraction_Solved', 'Model_Name']])

        if not performance_data:
            print(f"No data to plot for {event}. Skipping.")
            continue

        performance_df = pd.concat(performance_data)

        # Determine left-to-right order of models based on median Abs_Diff
        model_order = (
            performance_df.groupby('Model_Name')['Abs_Diff']
            .median()
            .sort_values()
            .index.tolist()
        )
        
        chart = alt.Chart(performance_df).mark_line(point=True).encode(
            x=alt.X('Abs_Diff', type='quantitative', title='Absolute Difference (Days)'),
            y=alt.Y('Fraction_Solved', type='quantitative', title='Fraction of Predictions'),
            color=alt.Color('Model_Name', legend=alt.Legend(title="Model"), 
                            scale=alt.Scale(domain=model_order, range=custom_colors)),
            tooltip=['Model_Name', 'Abs_Diff', 'Fraction_Solved']
        ).properties(title=f"ROC for {event}")

        # Save the chart for each event
        filename = f"output_graphs/roc/roc_{event}.png"
        chart.save(filename)

    # okay now let's combine all three images
    from PIL import Image

    # Load the image
    image0 = Image.open(f"output_graphs/roc/roc_{events[0]}.png")
    image1 = Image.open(f"output_graphs/roc/roc_{events[1]}.png")
    image2 = Image.open(f"output_graphs/roc/roc_{events[2]}.png")

    # Get original dimensions
    width, height = image0.size

    # Create a new blank image with triple width
    combined = Image.new('RGB', (width * 3, height))
    combined.paste(image0, (0, 0))
    combined.paste(image1, (width, 0))
    combined.paste(image2, (width*2, 0))
    # Save or show
    combined.save(f"output_graphs/roc/roc_all_combined.png")



def rmse_table():
    """Generate the RMSE table from the outputted models_diffs_rmses.csv file"""
    # Load CSV
    df = pd.read_csv('output_graphs/models_diffs_rmses.csv', skiprows=2, names=["Cultivar", "Event", "Model", "RMSE"])

    # Pivot to get Cultivar as rows, Event+Model as MultiIndex columns
    pivot = df.pivot_table(
        index="Cultivar",
        columns=["Event", "Model"],
        values="RMSE"
    )

    # Sort to keep consistent output
    pivot = pivot.sort_index(axis=1, level=[0, 1])

    # Format values and build LaTeX rows
    latex_rows = []
    for cultivar, row in pivot.iterrows():
        formatted = [f"{row.get((event, model), ''):.2f}" if pd.notna(row.get((event, model))) else ''
                     for event, model in pivot.columns]
        latex_rows.append(f"{cultivar} & " + " & ".join(formatted) + r" \\")

    # Output the LaTeX code block
    print("\n".join(latex_rows))



def validation_ranks_table():

    # Load the CSV
    df = pd.read_csv("output_graphs/cultivar_ranks_validation.csv")

    # Clean up: remove any unnamed index column and use first row as header
    df.columns = df.iloc[0]
    df = df[1:]

    # Rename relevant columns for convenience
    df = df.rename(columns={
        'Cultivar': 'Cultivar',
        'Phenology': 'Phenology',
        'Range': 'Range',
        'Contains Truth': 'Contains_Truth'
    })

    # Select only the needed columns
    df = df[['Cultivar', 'Phenology', 'Range', 'Contains_Truth']]

    # Convert Range to numeric and Contains_Truth to string
    df['Range'] = pd.to_numeric(df['Range'], errors='coerce')
    df['Contains_Truth'] = df['Contains_Truth'].astype(str)

    # Group by cultivar + phenology and calculate stats
    grouped = df.groupby(['Cultivar', 'Phenology']).agg(
        True_Percentage=('Contains_Truth', lambda x: 100 * (x == 'True').sum() / len(x)),
        Average_Range=('Range', 'mean')
    ).reset_index()

    # Pivot to create one row per cultivar with separate columns for each event's stats
    pivoted = grouped.pivot(index='Cultivar', columns='Phenology')

    # Flatten MultiIndex columns
    pivoted.columns = [f"{phen}_{stat}" for stat, phen in pivoted.columns]

    pivoted = pivoted.round(2)  # round all numeric columns

    # Fill any missing values and round again
    pivoted = pivoted.fillna('').reset_index()

    # Generate LaTeX table rows
    latex_rows = pivoted.apply(
        lambda row: f"\\textbf{{{row['Cultivar']}}} & {row.get('Budbreak_True_Percentage', '')}\\% & {row.get('Budbreak_Average_Range', '')} "
                    f"& {row.get('Full Bloom_True_Percentage', '')}\\% & {row.get('Full Bloom_Average_Range', '')} "
                    f"& {row.get('Veraison 50%_True_Percentage', '')}\\% & {row.get('Veraison 50%_Average_Range', '')} \\\\",
        axis=1
    ).tolist()

    for row in latex_rows:
        print(row)






def print_seasons():
    """Print out the seasons for each of the cultivars."""
    # load input files
    with open("input/data_output.pkl", "rb") as file:
        """
        list[(cultivar: str, season: str, true: np.array(1, 366, 5), pred: np.array(1, 366, 5))]
        """
        phenotracker_data_raw: dict = pickle.load(file)

    phenotracker_data_train = list(map(lambda x: (x[0], x[1], x[2].reshape((366, 5)).transpose(), np.array([x[3], x[4], x[5], x[6], x[7]])), phenotracker_data_raw["training"]))
    phenotracker_data_valid = list(map(lambda x: (x[0], x[1], x[2].reshape((366, 5)).transpose(), np.array([x[3], x[4], x[5], x[6], x[7]])), phenotracker_data_raw["validation"]))

    cultivar_seasons: dict[str, tuple[set, set]] = dict()
    for (cultivar, season, true, pred) in phenotracker_data_train:
        if cultivar not in cultivar_seasons:
            cultivar_seasons[cultivar] = ({season}, set())
        else:
            cultivar_seasons[cultivar][0].add(season)
    for (cultivar, season, true, pred) in phenotracker_data_valid:
        if cultivar not in cultivar_seasons:
            cultivar_seasons[cultivar] = (set(), {season})
        else:
            cultivar_seasons[cultivar][1].add(season)

    accum_valid = []
    for cultivar, train_valid in cultivar_seasons.items():
        print()
        print(cultivar.upper())
        (train, valid) = train_valid
        print("\tTRAIN")
        for s in sorted(train):
            print(s)

        print("\tVALID")
        for s in sorted(valid):
            print(s)
            accum_valid.append((cultivar, s))

    print()
    print("TO DOWNLOAD (just the four main cultivars), REMEMBER THESE HAVE BEEN ADJUSTED TO BE OFFSET BY ONE TO MATCH AWN")
    for s in filter(lambda x: x[0] in good_cultivars, accum_valid):
        print((s[0], int(s[1][:4]) + 1))




if __name__ == "__main__":
    generate_graphs_csvs() #main
    performance_graphs()
    rmse_table()
    validation_ranks_table()
    #print_seasons()


