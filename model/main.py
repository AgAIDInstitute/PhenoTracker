import argparse
import datetime
from unified_api import train_progress_stage, run_progress_stage
from create_dataset import create_dataset_multiple_cultivars_phenologies
import torch

import os
import pickle
import glob
import pandas as pd
import gc
import numpy as np
from pathlib import Path

from models import progress_stage_net as nn_model

from global_vars import season_collect

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Running all experiments from here, preventing code duplication")
    parser.add_argument('--experiment', type=str, default="multiplicative_embedding", choices=['multiplicative_embedding', 'mtl', 'additive_embedding', 'concat_embedding', 'single', 'ferguson', 'pheno_embedding', 'basic_cultivar', 'encode_cultivar', 'encode_cultivar_2', 'concat_embedding_vectors', 'concat_embedding_vectors_phenologies'], help='type of the experiment')
    #arg for freeze/unfreeze, all/leaveoneout, afterL1, L2,etc, linear/non linear embedding, scratch/linear combination for finetune, task weighting
    parser.add_argument('--setting', type=str, default="all", choices=['all','leaveoneout','allfinetune','embed','oracle','rand_embeds','embed_lin','baseline_avg','baseline_wavg','baseline_all','baseline_each','test_pheno'], help='experiment setting')
    parser.add_argument('--variant', type=str, default='none',choices=['none','afterL1','afterL2','afterL3','afterL4'])
    parser.add_argument('--unfreeze', type=str, default='no', choices=['yes','no'], help="unfreeze weights during finetune")
    #todo
    parser.add_argument('--nonlinear', type=str, default='no', choices=['yes','no'],help='try non linear embedding/prediction head')
    #todo
    parser.add_argument('--scratch', type=str, default='no', choices=['yes','no'],help='try learning embedding from scratch')
    parser.add_argument('--weighting', type=str, default='none', choices=['none', 'inverse_freq', 'uncertainty'],
                        help="loss weighting strategy")
    parser.add_argument('--name', type=str, default=datetime.datetime.now(
    ).strftime("%d_%b_%Y_%H_%M_%S"), help='name of the experiment')
    parser.add_argument('--epochs', type=int, default=400,
                        help='No of epochs to run the model for')
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--lr_emb', type=float, default=1e-4, help="Learning Rate for embedding")
    parser.add_argument('--no_seasons', type=int, default=-1, help="no of seasons to select for the Riesling Cultivar")
    parser.add_argument('--batch_size', type=int,
                        default=12, help="Batch size")
    parser.add_argument('--evalpath', type=int,
                        default=None, help="Evaluation Path")
    parser.add_argument('--data_path', type=str,
                        default='./input/', help="csv Path")
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help="pretrained model to load for finetuning")
    parser.add_argument('--embedding_path', type=str, default=None,
                        help="pretrained embedding model to load for cultivar prediction ground truth")
    parser.add_argument('--specific_cultivar', type=str, default=None,
                        help="specific cultivar to train for")
    parser.add_argument('--include_lte', action='store_true',
                        help="include lte loss in training")
    parser.add_argument('--include_pheno', action='store_true',
                        help="include pheno loss in training")
    parser.add_argument('--phenos', type=str, default=None,
                        help="comma seperated list of phenological events to predict, replace spaces with _")
    parser.add_argument('--train_cultivars', type=str, default=None,
                        help="comma seperated list of training cultivars, replace spaces with _, order matters, must include whenever model does not use default cultivars")
    parser.add_argument('--test_cultivars', type=str, default=None,
                        help="comma seperated list of test cultivars to predict, replace spaces with _")
    parser.add_argument('--train_embeds_LTE', action='store_true',
                        help="train embeds with LTE instead of pheno")
    parser.add_argument('--a_weight', type=str, default=None,
                        help="constant used in baseline weighted average") 
    parser.add_argument('--extra_embeds', action='store_true',
                        help="consider extra cultivars")    
    parser.add_argument('--exclude_source', action='store_true',
                        help="dont use original cultivar embeddings")                         
    parser.add_argument('--allow_cpu', action='store_true',
                        help="allow the use of CPU instead of GPU") #if don't include this, program will stop if no GPU available
    parser.add_argument('--skip_training', action='store_true',
                        help="skip training and go straight to testing")
    parser.add_argument('--get_seasons', action='store_true',
                        help="return the seasons available for each cultivar")
    args = parser.parse_args()


    # don't change these, these list all the valid cultivars, not the ones being trained on
    valid_cultivars = [
        'Barbera',
        'Cabernet Franc',
        'Cabernet Sauvignon',
        'Chardonnay',
        'Chenin Blanc',
        'Concord',
        'Gewurztraminer',
        'Grenache',
        'Lemberger',
        'Malbec',
        'Merlot',
        'Mourvedre',
        'Nebbiolo',
        'Pinot Gris',
        'Riesling',
        'Sangiovese',
        'Sauvignon Blanc',
        'Semillon',
        'Viognier',
        'Zinfandel',
    ]
    if args.train_cultivars != None:
        args.valid_cultivars = args.train_cultivars.replace('_',' ').split(',')
        valid_cultivars = args.valid_cultivars #both are used throughout the code, make sure both are the same
    else:
        args.valid_cultivars = valid_cultivars
    args.bb_day_diff = {cultivar:list() for cultivar in args.valid_cultivars}
    if args.phenos != None:
        args.phenos_list = args.phenos.replace('_',' ').split(',')
    else:
        args.phenos_list = []
        
    if args.test_cultivars != None:
        args.formatted_test_cultivars = args.test_cultivars.replace('_',' ').split(',')    
    else:
        args.formatted_test_cultivars = valid_cultivars


    print(args.test_cultivars)
    args.proper_cultivars = list(filter(lambda x: x in valid_cultivars, args.formatted_test_cultivars))

    args.cultivar_file_dict = {
        cultivar: pd.read_csv(
            glob.glob(args.data_path+'*'+cultivar+'*')[0]
        ) for cultivar in args.proper_cultivars
    }

    print(args.allow_cpu)
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("NOTICE: USING GPU")
    else:
        args.device = torch.device("cpu")
        if not args.allow_cpu: #if we are disallowing the use of CPU
            raise Exception("GPU WAS NOT AVAILABLE")
        print("NOTICE: GPU WAS NOT AVAILABLE, USING CPU")


    overall_loss = dict()
    #single model training



    #################

    args.features = [ #used as inputs FOR TRAINING
        'MIN_AT',
        'MEAN_AT',
        'MAX_AT',

        'MIN_RH',
        'AVG_RH',
        'MAX_RH',

        'MIN_DEWPT',
        'AVG_DEWPT',
        'MAX_DEWPT',

        'P_INCHES',

        'WS_MPH',
        'MAX_WS_MPH',

        'COUNTER'
    ]



    # model predicted variables, what we aim to predict
    args.label = ['PHEN_PROGRESS', "STAGE_V_0", "STAGE_V_1", "STAGE_V_2", "STAGE_V_3"]
    args.no_outputs = len(args.label)

    args.output_folder = "./output"
    Path(args.output_folder).mkdir(parents = True, exist_ok = True) #ensure this folder exists


    # print the seasons if requested
    args.training = True #just leave it
    if args.get_seasons == True:
        print("Retrieving seasons for all cultivars:")
        
        args.current_cultivar = 'all'
        args.no_of_cultivars = len(valid_cultivars)
        # get dataset by selecting features
        args.cultivar_list = list(valid_cultivars)
        args.nn_model = nn_model
        # similar for all experiments
        args.trial = 'trial_1'
        args.dataset = create_dataset_multiple_cultivars_phenologies(args)
        
        # print all the season names now
        print("\n\n\n\n\n\n")
        for s in season_collect:
            print(s)
        exit()


    training_trials = 9 #IMPORTANT, TODO


    args.training = True
    if args.skip_training == False:
        print("WILL NOW BEGIN TRAINING")
        #train
        overall_loss = dict()
        loss_dicts = dict()
        finetune_loss_dicts = dict()
        
        if args.pretrained_path == None: #if we aren't simply running the trained model
            args.current_cultivar = 'all'
            args.no_of_cultivars = len(valid_cultivars)
            # get dataset by selecting features
            args.cultivar_list = list(valid_cultivars)
            args.nn_model = nn_model
            # similar for all experiments
            #for trial in range(3, training_trials):
            for trial in range(training_trials):
                args.trial = 'trial_'+str(trial)
                args.dataset = create_dataset_multiple_cultivars_phenologies(args)
                args.pretrained_path = os.path.join(args.output_folder, args.trial + ".pt")
                loss_dicts[args.trial] = train_progress_stage(args)
            overall_loss[args.experiment] = loss_dicts
    


    print("WILL NOW BEGIN TESTING THE STUFF INSTEAD OF TRAINING, DONE WITH TRAINING")
    args.training = False
    
    overall_rnn_vectors = dict()
    overall_penul_vectors = dict()
    overall_progress_dict = dict()
    args.nn_model = nn_model

    accuracy_dict = dict()
    output_store_list = [[], []] #training, validation
    output_store_dict = {"training": [], "validation": []}

    for left_out in args.proper_cultivars: #enumerate(valid_cultivars):
        c_id = args.valid_cultivars.index(left_out)
        gc.collect()
        loss_dicts = dict()
        rnn_vectors = dict()
        penul_vectors = dict()
        progress_dict = dict()
        other_cultivars = list([left_out])
        args.current_cultivar = left_out
        args.no_of_cultivars = len(valid_cultivars)
        # get dataset by selecting features
        args.cultivar_list = list([left_out])
        for trial in range(training_trials):
            print(left_out, " Trial ", trial)
            args.trial = 'trial_'+str(trial)
            args.dataset = create_dataset_multiple_cultivars_phenologies(args)
            print(args.dataset.keys())

            if args.pretrained_path == None:
                args.pretrained_path = os.path.join(args.output_folder, args.trial + ".pt")

            rnn_vectors[args.trial], penul_vectors[args.trial], progress_dict[args.trial], training_rounds, validation_rounds = run_progress_stage(args, c_id)
            '''
            all_rounds.append((true_label, y.flatten(), out_progress.cpu().flatten(), out_stage.cpu().flatten()))
            '''

            # training
            for true_label, y, pred_progress, pred_stage0, pred_stage1, pred_stage2, pred_stage3 in training_rounds:

                ## the collected values from running the model are pretty wack, so here we reorganize them to be nicer
                output_store_dict["training"].append((true_label[0][0], true_label[1][0], y.numpy(), pred_progress.numpy(), pred_stage0.numpy(), pred_stage1.numpy(), pred_stage2.numpy(), pred_stage3.numpy()))




            # validation
            for true_label, y, pred_progress, pred_stage0, pred_stage1, pred_stage2, pred_stage3 in validation_rounds:

                ## the collected values from running the model are pretty wack, so here we reorganize them to be nicer
                output_store_dict["validation"].append((true_label[0][0], true_label[1][0], y.numpy(), pred_progress.numpy(), pred_stage0.numpy(), pred_stage1.numpy(), pred_stage2.numpy(), pred_stage3.numpy()))


        overall_rnn_vectors[left_out] = rnn_vectors
        overall_penul_vectors[left_out] = penul_vectors
        overall_progress_dict[left_out] = progress_dict
        

    with open(args.output_folder + "/data_output.pkl", "wb") as file:
        pickle.dump(output_store_dict, file, protocol = pickle.HIGHEST_PROTOCOL)


    print("FINAL RESULTS:")
    print(list(map(lambda x: (x[0], list(map(lambda y: y[0], x[1]))), accuracy_dict.items()))) #ignore the y and pred_y


    #################

    print(overall_loss)
    
    with open(os.path.join(args.output_folder + "/losses.pkl"), 'wb') as f:
        pickle.dump(overall_loss, f)

    # hidden vectors
    Path(args.output_folder + "/hidden_vectors").mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.output_folder + "/hidden_vectors", "rnn_vectors.pkl"), 'wb') as f:
        pickle.dump(overall_rnn_vectors, f)
    with open(os.path.join(args.output_folder + "/hidden_vectors", "penul_vectors.pkl"), 'wb') as f:
        pickle.dump(overall_penul_vectors, f)




