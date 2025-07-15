"""Used to process data files downloaded from AWN for the Zapata and Parker models."""

import os
import pickle



# include an entry for each station in the files, with the bool determining if we extract it or not
stations = [
    (True, "Roza.2"),
]



# used to store EVERYTHING, eventually gets pickled
huge_dict = dict()

def process_file(path_name: str):
    filename = path_name[path_name.rfind("/")+1:] #remove the folders
    cultivar = filename[:filename.index("_")]
    year = filename[filename.index("_")+1:filename.index(".")]

    with open(path_name) as file:
        lines = file.readlines()

    # separate the file into individual groupings based on each unique model entry (each model may appear several times)
    lis = []
    for line in lines:
        if line.startswith("Model"):
            # push line to a new list
            lis.append([line[:-1]])
        elif line == "\n":
            # skip any empty lines
            continue
        else:
            # push to existing group
            lis[-1].append(line[:-1])

    # break a group of lines into sections
    def process_group(group: list[str]):
        data_start = 0
        results_start = 0

        for i, line in enumerate(group):
            if line.startswith("DATE"):
                data_start = i
            elif line.startswith("Bud Break") or line.startswith("Full Bloom") or line.startswith("Veraison"):
                results_start = i
                break

        header = group[:data_start]
        data = group[data_start:results_start]
        footer = group[results_start:]

        # the footer can be unintuitive sometimes so let's fix it
        # this code deletes all repeats but preserves order (which a dict does but a set does not)
        footer = list(dict.fromkeys(filter(lambda x: not x.endswith(": "), footer)).keys())

        return (0 if results_start != 0 else 1, header, data, footer)

    # attach the station names and filter
    lis_f = []

    for i, group in enumerate(lis):
        station_num = i // 2
        station_tup = stations[station_num]
        if station_tup[0] == 1:
            proc = process_group(group)
            if proc[0] == 0:
                lis_f.append((station_tup[1], proc[1], proc[2], proc[3]))

    # extra processing to remove what is unneeded and reformat
    for station, header, _, predictions in lis_f:
        model = header[0][header[0].index(": ")+2:]

        predict_dict = dict()
        for line in predictions:
            pivot = line.index(":")
            start = line[:pivot]
            end = line[pivot+2:]
            predict_dict[start] = end

        huge_dict[(station, year, cultivar, model)] = predict_dict


folder_name = "input/other_models_roza.2"

for filename in os.listdir(folder_name):
    full_path = os.path.join(folder_name, filename)
    if os.path.isfile(full_path):
        process_file(full_path)

with open("output/other_models_dump.pkl", "wb") as file:
    pickle.dump(huge_dict, file)




