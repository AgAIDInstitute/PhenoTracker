import torch
from torch.autograd import Variable
from create_dataset import MyDataset, MyDatasetPhen
#from util.data_processing import evaluate
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import os
from pathlib import Path
import pickle
from global_vars import debugging_dict
#import ipdb as pdb



def train_progress_stage(args, finetune = False):
    """
    Used for training, progress and stage.
    """
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    no_of_cultivars = args.no_of_cultivars
    no_outputs = args.no_outputs

    model = args.nn_model(feature_len, no_of_cultivars, nonlinear = args.nonlinear)  #load parameters into model

    if args.unfreeze=='yes':
        for param in model.parameters():
            param.requires_grad = True
    if finetune:
        model.load_state_dict(torch.load(args.pretrained_path), strict=False)
    model.to(args.device)
    trainable_params = sum([np.prod(p.size()) for p in filter(
        lambda p: p.requires_grad, model.parameters())])
    print("Trainable Parameters:", trainable_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #criterion2 = nn.BCELoss(reduction='none')
    criterion = nn.MSELoss(reduction='none') #progress
    criterion.to(args.device)
    criterion1 = nn.MSELoss(reduction='none') #stage0
    criterion1.to(args.device)
    criterion2 = nn.MSELoss(reduction='none') #stage1
    criterion2.to(args.device)
    criterion3 = nn.MSELoss(reduction='none') #stage2
    criterion3.to(args.device)
    criterion4 = nn.MSELoss(reduction='none') #stage3
    criterion4.to(args.device)

    log_dir = os.path.join('./tensorboard/',args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch, args.trial, args.current_cultivar)

    writer = SummaryWriter(log_dir)
    train_dataset = MyDataset(dataset['train'])
    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = MyDataset(dataset['test'])
    valLoader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    #embeds = dict()

    for epoch in range(args.epochs):
        # Training Loop

        model.train()
        total_loss = 0
        count = 0
        total_loss_pheno = 0
        for i, (x, y, cultivar_id, freq) in enumerate(trainLoader):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            freq = freq.to(args.device)
            cultivar_id_torch = cultivar_id.to(args.device)
            count += 1

            out_progress, out_stage0, out_stage1, out_stage2, out_stage3, _, int_vector, state_vector = model(x_torch, cultivar_label=cultivar_id_torch)

            optimizer.zero_grad()       # zero the parameter gradients
            #replace nan in gt with 0s, replace corresponding values in pred with 0s
            nan_locs_progress = y_torch[:, :, 0].isnan()
            nan_locs_stage0 = y_torch[:, :, 1].isnan()
            nan_locs_stage1 = y_torch[:, :, 2].isnan()
            nan_locs_stage2 = y_torch[:, :, 3].isnan()
            nan_locs_stage3 = y_torch[:, :, 4].isnan()
            out_progress[:,:,0][nan_locs_progress] = 0
            out_stage0[:,:,0][nan_locs_stage0] = 0
            out_stage1[:,:,0][nan_locs_stage1] = 0
            out_stage2[:,:,0][nan_locs_stage2] = 0
            out_stage3[:,:,0][nan_locs_stage3] = 0
            y_torch = torch.nan_to_num(y_torch)
            loss_progress = criterion(out_progress[:,:,0], y_torch[:, :, 0])[~nan_locs_progress]
            loss_stage0 = criterion1(out_stage0[:,:,0], y_torch[:, :, 1])[~nan_locs_stage0]
            loss_stage1 = criterion2(out_stage1[:,:,0], y_torch[:, :, 2])[~nan_locs_stage1]
            loss_stage2 = criterion3(out_stage2[:,:,0], y_torch[:, :, 3])[~nan_locs_stage2]
            loss_stage3 = criterion4(out_stage3[:,:,0], y_torch[:, :, 4])[~nan_locs_stage3]

            loss = 0
            loss += loss_progress.mean()
            loss += loss_stage0.mean()
            loss += loss_stage1.mean()
            loss += loss_stage2.mean()
            loss += loss_stage3.mean()

            loss.backward()             # backward +
            optimizer.step()            # optimize
            total_loss += loss.item()

        writer.add_scalar('Train_Loss', total_loss / count, epoch)
        writer.add_scalar('Train_Loss_Pheno', total_loss_pheno / count, epoch)
        # Validation Loop
        with torch.no_grad():
            model.eval()
            total_loss = 0
            count = 0
            for i, (x, y, cultivar_id, freq) in enumerate(valLoader):
                x_torch = x.to(args.device)
                y_torch = y.to(args.device)
                cultivar_id_torch = cultivar_id.to(args.device)
                count += 1

                out_progress, out_stage0, out_stage1, out_stage2, out_stage3, _, int_vector, state_vector = model(x_torch, cultivar_label=cultivar_id_torch)

                #replace nan in gt with 0s, replace corresponding values in pred with 0s
                nan_locs_progress = y_torch[:, :, 0].isnan()
                nan_locs_stage0 = y_torch[:, :, 1].isnan()
                nan_locs_stage1 = y_torch[:, :, 2].isnan()
                nan_locs_stage2 = y_torch[:, :, 3].isnan()
                nan_locs_stage3 = y_torch[:, :, 4].isnan()
                out_progress[:,:,0][nan_locs_progress] = 0
                out_stage0[:,:,0][nan_locs_stage0] = 0
                out_stage1[:,:,0][nan_locs_stage1] = 0
                out_stage2[:,:,0][nan_locs_stage2] = 0
                out_stage3[:,:,0][nan_locs_stage3] = 0
                y_torch = torch.nan_to_num(y_torch)
                # getting non nan values is slow right now due to copying to cpu, write pytorch gpu version
                loss_progress = criterion(out_progress[:,:,0], y_torch[:, :, 0])[~nan_locs_progress]  # LT10 GT
                loss_stage0 = criterion1(out_stage0[:,:,0], y_torch[:, :, 1])[~nan_locs_stage0]
                loss_stage1 = criterion2(out_stage1[:,:,0], y_torch[:, :, 2])[~nan_locs_stage1]
                loss_stage2 = criterion3(out_stage2[:,:,0], y_torch[:, :, 3])[~nan_locs_stage2]
                loss_stage3 = criterion4(out_stage3[:,:,0], y_torch[:, :, 4])[~nan_locs_stage3]
                
                loss = 0
                loss += loss_progress.mean()
                loss += loss_stage0.mean()
                loss += loss_stage1.mean()
                loss += loss_stage2.mean()
                loss += loss_stage3.mean()

                total_loss += loss.mean().item()
            writer.add_scalar('Val_Loss', total_loss / count, epoch)
    loss_dict = dict()
    torch.save(model.state_dict(), os.path.join(args.output_folder, args.trial + ".pt"))
    total_loss_progress, total_loss_stage0, total_loss_stage1, total_loss_stage2, total_loss_stage3 = 0, 0, 0, 0, 0
    with torch.no_grad():
        model.eval()
        for i, ((x, y, cultivar_id, freq), cultivar) in enumerate(zip(valLoader,args.cultivar_list)):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            cultivar_id_torch = cultivar_id.to(args.device)

            out_progress, out_stage0, out_stage1, out_stage2, out_stage3, _, int_vector, state_vector = model(x_torch, cultivar_label=cultivar_id_torch)

            #replace nan in gt with 0s, replace corresponding values in pred with 0s
            nan_locs_progress = y_torch[:, :, 0].isnan()
            nan_locs_stage0 = y_torch[:, :, 1].isnan()
            nan_locs_stage1 = y_torch[:, :, 2].isnan()
            nan_locs_stage2 = y_torch[:, :, 3].isnan()
            nan_locs_stage3 = y_torch[:, :, 4].isnan()
            out_progress[:,:,0][nan_locs_progress] = 0
            out_stage0[:,:,0][nan_locs_stage0] = 0
            out_stage1[:,:,0][nan_locs_stage1] = 0
            out_stage2[:,:,0][nan_locs_stage2] = 0
            out_stage3[:,:,0][nan_locs_stage3] = 0
            y_torch = torch.nan_to_num(y_torch)
            loss_progress = criterion(out_progress[:,:,0], y_torch[:, :, 0])[~nan_locs_progress].mean().item()
            loss_stage0 = criterion1(out_stage0[:,:,0], y_torch[:, :, 1])[~nan_locs_stage0].mean().item()
            loss_stage1 = criterion2(out_stage1[:,:,0], y_torch[:, :, 2])[~nan_locs_stage1].mean().item()
            loss_stage2 = criterion3(out_stage2[:,:,0], y_torch[:, :, 3])[~nan_locs_stage2].mean().item()
            loss_stage3 = criterion4(out_stage3[:,:,0], y_torch[:, :, 4])[~nan_locs_stage3].mean().item()

            total_loss_progress += loss_progress
            total_loss_stage0 += loss_stage0
            total_loss_stage1 += loss_stage1
            total_loss_stage2 += loss_stage2
            total_loss_stage3 += loss_stage3
            loss_dict[cultivar] = list([np.sqrt(loss_progress), np.sqrt(loss_stage0), np.sqrt(loss_stage1), np.sqrt(loss_stage2), np.sqrt(loss_stage3)])
    loss_dict['overall'] = list([np.sqrt(total_loss_progress), np.sqrt(total_loss_stage0), np.sqrt(total_loss_stage1), np.sqrt(total_loss_stage2), np.sqrt(total_loss_stage3)])

    return loss_dict


def run_progress_stage(args, cultivar_id):
    """
    Used for running, progress no stage.
    """
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    no_of_cultivars = args.no_of_cultivars
    model = args.nn_model(feature_len, no_of_cultivars, nonlinear = args.nonlinear)
    model.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
    model.to(args.device)
    
    train_dataset = MyDatasetPhen(dataset['train'])
    val_dataset = MyDatasetPhen(dataset['test'])
    trainLoader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    valLoader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    internal_dict = dict()
    state_dict = dict()
    progress_dict = dict()

    training_rounds = list()
    validation_rounds = list()

    debugging_dict["cultivar_arrs"] = []
    debugging_dict["xs"] = []
    debugging_dict["seasons"] = []

    with torch.no_grad():
        model.eval()

        # just do this once, used to be done in the validation loop
        cultivar_label = iter(trainLoader).__next__()[2] #grab the cultivar label (same for every instance)
        for c1 in range(cultivar_label.shape[0]):
            for c2 in range(cultivar_label.shape[1]):
                cultivar_label[c1,c2] = cultivar_id

        cultivar_id_torch = cultivar_label.to(args.device)


        count = 0

        # THE TRAINED LOOP (running on seasons that were used for training)
        for (x, y, cultivar_label, true_label) in trainLoader:
            x_torch = x.to(args.device)

            # run through the model
            out_progress, out_stage0, out_stage1, out_stage2, out_stage3, _, int_vector, state_vector = model(x_torch, cultivar_label=cultivar_id_torch)

            # collect output data
            training_rounds.append((true_label, y, out_progress.cpu().flatten(), out_stage0.cpu().flatten(), out_stage1.cpu().flatten(), out_stage2.cpu().flatten(), out_stage3.cpu().flatten()))

            count += 1



        # THE VALIDATION LOOP (running on seasons that weren't used for training)
        for (x, y, cultivar_label, true_label) in valLoader:
            x_torch = x.to(args.device)

            # run through the model
            out_progress, out_stage0, out_stage1, out_stage2, out_stage3, _, int_vector, state_vector = model(x_torch, cultivar_label=cultivar_id_torch)

            # collect output data
            validation_rounds.append((true_label, y, out_progress.cpu().flatten(), out_stage0.cpu().flatten(), out_stage1.cpu().flatten(), out_stage2.cpu().flatten(), out_stage3.cpu().flatten()))

            # collect data on the hidden states
            internal_dict[str(count)] = int_vector.detach().cpu().numpy()
            state_dict[str(count)] = state_vector.detach().cpu().numpy()
            progress_dict[str(count)] = {'progress': out_progress.detach().cpu().numpy()[0,:,0]}

            count += 1


    return internal_dict, state_dict, progress_dict, training_rounds, validation_rounds




