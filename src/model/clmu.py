import torch
import sys
import numpy as np
from torch.autograd import Function
from torch import nn
import matplotlib.pyplot as plt

sys.path.append("..")
from model import neural_process as nproc
from model.neural_process import MAE, random_split_context_target
from util import train as util_train
from util import split_data, load_mix_data_sim, SimDataset
# %load_ext autoreload
# %autoreload 2
from p_level_mapping import PLevels

from model.LMUseq2seq import CondLMUSeq2Seq 

from torch.utils import data
from datetime import datetime 
# SEQ_STEPS = 90

import wandb
import argparse

import pickle
import os

import logging

parser = argparse.ArgumentParser(description='Train a CLMU model on a specified cohort using PyTorch.')
parser.add_argument('-seq_steps', '--seq_steps', type=int, metavar='<steps>', required=True,
                    help='Length of input and output sequences.')
parser.add_argument('-cuda', '--cuda_number', type=str, metavar='<device>', required=True,
                    help='Specify the CUDA device number to use.')
parser.add_argument('-seq_dim', '--seq_dim', type=int, metavar='<dim>', default=32,
                    help='Specify the sequence dimension.')
parser.add_argument('-r_dim', '--r_dim', type=int, metavar='<dim>', default=4,
                    help='Specify the r dimension.')
parser.add_argument('-z_dim', '--z_dim', type=int, metavar='<dim>', default=16,
                    help='Specify the z dimension.')
parser.add_argument('-bc', '--batch_size', type=int, metavar='<size>', default=64,
                    help='Specify the batch size.')
parser.add_argument('-lr', '--learning_rate', type=float, metavar='<rate>', default=1e-3,
                    help='Specify the learning rate.')
parser.add_argument('-cohort', '--cohort', type=str, metavar='<cohort>', default='hrpci',
                    help='Specify the cohort to use.', choices=['hrpci', 'amicgs'])
parser.add_argument('-nepochs', '--nepochs', type=int, metavar='<epochs>', default=5000,
                    help='Specify the number of epochs to train for.')
parser.add_argument('-patience', '--patiences', type=int, metavar='<patiences>', default=200,
                    help='Specify the patience for early stopping.')
parser.add_argument('-subsample', '--subsample', type=float, metavar='<float>', default=0.02,
                    help='Specify the subsample ratio of the cohort (between 0 and 1).')

args = parser.parse_args()
device = torch.device(args.cuda_number)

wandb.init(
    project="abiomed-clmu",
    config=vars(args)
)
# wandb.init(mode="disabled")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
# if not os.path.exists("../logger_info_logs"):
#     os.mkdir("../logger_info_logs")
# if not os.path.exists("../logger_info_logs"):
#     os.mkdir("../logger_info_logs")
log_dir = "../logger_info_logs/CLMU_log"
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(log_dir + '/experiment_CLMU{}.log'.format(datetime.now().strftime("%d_%m_%Y_%H:%M")))
file_handler.setLevel(logging.CRITICAL)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def read_data(device, cohort='hrpci', bc=64):
    # data
    if cohort == 'hrpci': 
        rwd_train = torch.load("data/processed/rwd_at_new_pp_train_hrpci.pt")
        rwd_test = torch.load("data/processed/rwd_at_new_pp_test_hrpci.pt")
    if cohort == 'amicgs':
        import random
        import math
        random.seed(1551)
        rwd_train = torch.load("data/processed/rwd_at_new_pp_train_amicgs.pt")
        rwd_test = torch.load("data/processed/rwd_at_new_pp_test_amicgs.pt")
        M = int(len(rwd_train) * args.subsample)
        interval = len(rwd_train) // M
        train_sub_idx = [range(len(rwd_train))[i * interval] for i in range(M)]
        # train_sub_idx = random.sample(range(rwd_train.shape[0]),math.floor(rwd_train.shape[0]*subsample))
        test_sub_idx = list(range(0, int(len(rwd_test) * args.subsample)))
        # random.sample(range(rwd_test.shape[0]),math.floor(rwd_test.shape[0]*subsample))
        idx_dir = "../CLMU_results/{}/sub_idx".format(cohort)
        # final_idx_dir = idx_dir + "/" + datetime.now().strftime("%d_%m_%Y_%H:%M")
        os.makedirs(idx_dir, exist_ok=True)
        with open(idx_dir + "/" + datetime.now().strftime("%d_%m_%Y_%H:%M"), "wb") as f:
            pickle.dump(test_sub_idx, f)
            
        rwd_train = rwd_train[train_sub_idx]  # Subsample in loader
        rwd_test = rwd_test[test_sub_idx]
        logger.debug("rwd_train_shape is {}".format(rwd_train.shape))
        logger.debug("rwd_test_shape is {}".format(rwd_test.shape))

    # note this source
    sim_data = torch.from_numpy(np.load("data/processed/sim_data.npy"))
    logger.info("finished loading sim data and real data for cohort {}".format(cohort))

    class Dataset(data.Dataset):
        def __init__(self, x, y, seq):
            self.x = x
            self.y = y
            self.seq = seq

        def __getitem__(self, index):
            x = self.x[index]
            y = self.y[index]
            seq = self.seq[index]
            return x, y, seq

        def __len__(self):
            return self.x.shape[0]
        
    # Sim Data 
    np.random.seed(0)
    # idx = np.random.choice(sim_data.shape[0], int(sim_data.shape[0]*0.8), replace=False)
    idx = np.arange(0, int(sim_data.shape[0] * 0.8))

    sim_data_train = sim_data[idx, ...]

    means = sim_data_train.mean(axis=(0, 1))[None, None, :]
    sim_stds_old = sim_data_train.std(axis=(0, 1))[None, None, :]
    sim_means_old = means

    x_n = (sim_data - means) / sim_stds_old

    x_train = x_n[idx][..., [0, 2, 3, 4, 5, 6]]

    y_train = x_n[idx][..., 1:2]
    logger.debug("x_train_shape is {}".format(x_train.shape))
    logger.debug("y_train_shape is {}".format(y_train.shape))
    all_train = x_n[idx]
    logger.debug("all_train_shape is {}".format(all_train.shape))

    sim_loader_train = data.DataLoader(
        Dataset(x_train, y_train, all_train),
        batch_size=bc,
        shuffle=True,
        num_workers=4,
        persistent_workers=True, 
        pin_memory=True
    )

    x_test = np.delete(x_n[:, :, [0, 2, 3, 4, 5, 6]], idx, axis=0)
    y_test = np.delete(x_n[:, :, 1:2], idx, axis=0)
    all_test = np.delete(x_n, idx, axis=0)
    logger.debug("x_test_shape is {}".format(x_test.shape))
    logger.debug("y_test_shape is {}".format(y_test.shape))
    logger.debug("all_test_shape is {}".format(all_test.shape))

    sim_loader_test = data.DataLoader(
        Dataset(x_test, y_test, all_test),
        batch_size=bc,
        shuffle=False,
        num_workers=4,
        persistent_workers=True, 
        pin_memory=True
    )

    np.random.seed(0)
    # idx = np.random.choice(rwd.shape[0], int(rwd.shape[0]*0.8), replace=False)

    # idx = np.arange(0, int(rwd.shape[0]*0.8))
    # rwd_train = rwd[idx, ...]

    rwd_means = rwd_train.mean(axis=(0, 1))[None, None, :]
    rwd_stds = rwd_train.std(axis=(0, 1))[None, None, :]

    x_n = (rwd_train - rwd_means) / rwd_stds

    x_train_t = x_n[..., [1, 2, 3, 4, 5, 6]]
    y_train_t = x_n[..., 0:1]
    all_train_t = x_n

    logger.info("x_train_t_shape is {}".format(x_train_t.shape))
    logger.info("y_train_t_shape is {}".format(y_train_t.shape))
    logger.info("all_train_t_shape is {}".format(all_train_t.shape))

    rwd_loader_train = data.DataLoader(
        Dataset(x_train_t, y_train_t, all_train_t),
        batch_size=bc,
        shuffle=True,
        num_workers=4,
        persistent_workers=True, 
        pin_memory=True
    )
    x_n_test = (rwd_test - rwd_means) / rwd_stds
    x_test_t = x_n_test[..., [1, 2, 3, 4, 5, 6]]
    y_test_t = x_n_test[..., 0:1]
    all_test_t = x_n_test

    logger.info("x_test_t_shape is {}".format(x_test_t.shape))
    logger.info("y_test_t_shape is {}".format(y_test_t.shape))
    logger.info("all_test_t_shape is {}".format(all_test_t.shape))

    rwd_loader_test = data.DataLoader(
        Dataset(x_test_t, y_test_t, all_test_t),
        batch_size=bc,
        shuffle=False,
        num_workers=4,
        persistent_workers=True, 
        pin_memory=True
    )
    sim_stds = torch.index_select(sim_stds_old, 2, torch.LongTensor([1, 0, 2, 3, 4, 5, 6]))
    sim_means = torch.index_select(sim_means_old, 2, torch.LongTensor([1, 0, 2, 3, 4, 5, 6]))
    return (sim_loader_train, sim_loader_test, rwd_loader_train, rwd_loader_test), \
        (x_train_t, y_train_t, all_train_t), \
        (x_test_t, y_test_t, all_test_t), \
        (sim_means, sim_stds, rwd_means, rwd_stds) 


device = torch.device(args.cuda_number)
all_data = read_data(device, args.cohort, args.batch_size)
sim_means, sim_stds, rwd_means, rwd_stds = all_data[3]
cond = torch.vstack([rwd_stds[0, :], sim_stds[0, :]]).float().to(device)
clmu = CondLMUSeq2Seq(1, 30, cond, 7, 1, 32, 32, 784, device, attn=True, perturb="uniform", encode_only=True).to(device)

r_dim = args.r_dim
z_dim = args.z_dim
x_dim = 6
y_dim = 1 
seq_dim = args.seq_dim

model = nproc.DCRNNModel(x_dim, y_dim, r_dim, z_dim, seq_model=clmu, encode_only=False, device=device).to(device)
SEQ_STEPS = args.seq_steps
# n_epochs = 50000
# n_display=20
# patience = 5000
# mae_losses = []
# kld_losses = []

# opt = torch.optim.Adam(model.parameters(), 1e-3) #1e-4 2e-5

# min_loss = 0. # for early stopping
# wait = 0
# min_loss = float('inf')
logger.info("training starts")
n_epochs = args.nepochs
n_display = 1
patience = args.patiences
display_losses = []
best_model = None
opt = torch.optim.Adam(model.parameters(), args.learning_rate)  # 1e-4 2e-5
wait = 0
min_loss = float('inf')
model_dir = "../CLMU_results/trained_clmu"
loss_dir = "../CLMU_results/loss_plots"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)


def create_label(shape, sim=True):
    # create a tensor of size (shape, 1) with all zeros
    zeros = torch.zeros((shape, 1))

    # create a tensor of size (shape, 1) with all ones
    ones = torch.ones((shape, 1))

    # concatenate the tensors along the second dimension to get a tensor of size (n, 2)
    if sim:
        tensor = torch.cat((zeros, ones), dim=1)
    else:
        tensor = torch.cat((ones, zeros), dim=1)

    return tensor


for t in range(n_epochs): 
    print("---epoch # {} starts---".format(t))
    opt.zero_grad()
    
    # sim_iter = iter(sim_loader_train)
    rwd_loader_train = all_data[0][2]
    sim_iter_train = all_data[0][0]

    n_batches = len(rwd_loader_train)
    losses = []

    for i in range(n_batches):
        # x_train, y_train, all_train, labels = next(iter(sim_loader_train))
        x_train, y_train, all_train = next(iter(sim_iter_train))
        labels = create_label(y_train.shape[0], True)
  
        # x_train_t, y_train_t, all_train_t, labels_td = next(iter(rwd_loader_train))
        x_train_t, y_train_t, all_train_t = next(iter(rwd_loader_train))
        labels_td = create_label(y_train_t.shape[0], False)

        p = max(1, float(t) / (5000))
        lambda_grl = 2 / (1 + np.exp(-10 * p)) - 1 

        ## Generate data and process
        x_context, y_context, x_target, y_target, mask = random_split_context_target(
            x_train, y_train, int(len(y_train) * 0.05)
        )

        x_context_t, y_context_t, x_target_t, y_target_t, mask_t = random_split_context_target(
            x_train_t, y_train_t, len(mask)
        )

        x_c = x_context.float().to(device)
        x_t = x_target.float().to(device)
        y_c = y_context.float().to(device)
        y_t = y_target.float().to(device)
        
        x_ct = torch.cat([x_c, x_t], dim=0).float().to(device)
        y_ct = torch.cat([y_c, y_t], dim=0).float().to(device)

        x_c_t = x_context_t.float().to(device)
        x_t_t = x_target_t.float().to(device)
        y_c_t = y_context_t.float().to(device)
        y_t_t = y_target_t.float().to(device)
        
        x_ct_t = torch.cat([x_c_t, x_t_t], dim=0).float().to(device)
        y_ct_t = torch.cat([y_c_t, y_t_t], dim=0).float().to(device)
        
        # Source
        seqs = all_train[:, :SEQ_STEPS, :].float().to(device)
        start_seq_c = seqs[mask, :, :].float().to(device)
        labels_c = labels[mask, :].float().to(device)

        anti_mask = torch.ones(seqs.shape[0], dtype=bool)
        anti_mask[mask] = False
        start_seq_t = seqs[anti_mask, :, :].float().to(device)
        labels_t = labels[anti_mask, :].float().to(device)

        start_seq_ct = torch.cat([start_seq_c, start_seq_t], axis=0).float().to(device)
        labels_ct = torch.cat([labels_c, labels_t], axis=0).float().to(device)

        y_pred = model(
            x_t[:, :SEQ_STEPS, :], x_c, y_c, x_ct, y_ct, start_seq_c, start_seq_t, start_seq_ct, SEQ_STEPS,
            labels_c, labels_t, labels_ct)

        ## Target
        seqs = all_train_t[:, :SEQ_STEPS, :].float().to(device)
        start_seq_c_t = seqs[mask_t, :, :].float().to(device)
        labels_c_t = labels_td[mask_t, :].float().to(device)

        anti_mask = torch.ones(seqs.shape[0], dtype=bool)
        anti_mask[mask_t] = False
        start_seq_t_t = seqs[anti_mask, :, :]
        labels_t_t = labels_td[anti_mask, :].float().to(device)

        start_seq_ct_t = torch.cat([start_seq_c_t, start_seq_t_t], axis=0)
        labels_ct_t = torch.cat([labels_c_t, labels_t_t], axis=0).float().to(device)
        # print(y_pred.shape)
        # print(y_t[:,SEQ_STEPS:,:].shape)
        
        mae_loss = MAE(y_pred, y_t[:, SEQ_STEPS:, :].reshape(-1, 1))
        kld_sim = model.KLD_gaussian()

        y_pred_t = model(
            x_t_t[:, :SEQ_STEPS, :], x_c_t, y_c_t, x_ct_t, y_ct_t, start_seq_c_t, start_seq_t_t, start_seq_ct_t, 
            SEQ_STEPS, labels_c_t, labels_t_t, labels_ct_t
        )

        loss_t = MAE(y_pred_t, y_t_t[:, SEQ_STEPS:, :].reshape(-1, 1)) + model.KLD_gaussian()

        loss = mae_loss + kld_sim + loss_t
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # 10
        opt.step()

        losses.append(loss.item())
        
    cum_loss = np.mean(losses)
    wandb.log({"train_loss": cum_loss})
    display_losses.append(cum_loss)

    if cum_loss < min_loss:
        wait = 0
        best_model = model
        min_loss = cum_loss
            
    elif cum_loss >= min_loss:
        wait += 1
        if wait == patience:
            logger.info('Early stopping at epoch: %d' % t)
            # torch.save(best_model, "../trained_dann/best_model_loss{}.pt".format(round(min_loss, 2)))
            torch.save(best_model, model_dir + "/final_best_model_loss{}_{}.pt"
                       .format(round(min_loss, 2), datetime.now().strftime("%d_%m_%Y_%H:%M")))
    
            plt.plot(display_losses)
            plt.xlabel('epoch number')
            plt.ylabel('normalized MAE ')
            plt.title('training loss')
            plt.savefig(loss_dir + "/final_loss_plots_epoch{}_{}".format(t, datetime.now().strftime("%d_%m_%Y_%H:%M")))

    if t % 100 == 0:
        # torch.save(model, "../trained_dann/dann_pl_hrpci_epoch{}_loss{}.pt".format(t,round(cum_loss, 2)))
        torch.save(best_model, model_dir + "/best_clmu_pl_{}_epoch{}_loss{}_{}.pt"
                    .format(args.cohort, t, round(min_loss, 2), datetime.now().strftime("%d_%m_%Y_%H:%M")))
        torch.save(model, model_dir + "/clmu_pl_{}_epoch{}_loss{}_{}.pt"
                    .format(args.cohort, t, round(cum_loss, 2), datetime.now().strftime("%d_%m_%Y_%H:%M")))
    if t % n_display == 0:
        logger.info('epoch:{}, epoch loss:{}, cur_min_loss:{}'.format(t, cum_loss, min_loss))
        plt.plot(display_losses)
        plt.xlabel('epoch number')
        plt.ylabel('normalized MAE ')
        plt.title('training loss')
        # plt.savefig("../loss_plots/n_display_epoch{}".format(t))
        plt.savefig(loss_dir + "/n_display_epoch{}_{}"
                    .format(t, datetime.now().strftime(datetime.now().strftime("%d_%m_%Y_%H:%M"))))
    else:
        if t % 5 == 0:
            logger.debug('epoch:{}, epoch loss:{}, cur_min_loss:{}'.format(t, cum_loss, min_loss))
                
torch.save(best_model, model_dir + "/final_best_model_loss{}_{}.pt"
            .format(round(min_loss, 2), datetime.now().strftime("%d_%m_%Y_%H:%M")))
plt.plot(display_losses)
plt.xlabel('epoch number')
plt.ylabel('normalized MAE ')
plt.title('training loss')
plt.savefig(loss_dir + "/final_loss_plots_epoch{}_{}".format(t, datetime.now().strftime("%d_%m_%Y_%H:%M")))
print("finishing training")


logger.info("inference starting")
main_dir = "../CLMU_results/{}".format(args.cohort)


def inference(cohort, seq_steps, data_, best_model, device):
    logger.info("inference starting")
    main_dir = "../CLMU_results/{}".format(cohort)
    
    def test(x_train, y_train, x_test, start_seq_train, start_seq_test, seq_steps, best_model, device=device):
        # original_x_test = x_test
        x_test = x_test[:, :seq_steps, :]
        with torch.no_grad():
            z_mu, z_logvar = best_model.data_to_z_params(x_train.to(device), y_train.to(device), start_seq_train, 
                                                         create_label(x_train.shape[0], sim=False).to(device))
            output_list = []
            for i in range(x_test.shape[0]):
                zsamples = best_model.sample_z(z_mu, z_logvar)
                start_seq = start_seq_test[i:i + 1, :, :].to(device)
                # logger.debug("------logger.debuging start_seq------")
                # logger.debug(start_seq.shape)
                
                x_t = x_test[i:i + 1, :, :].to(device)

                zs_reshaped = zsamples.unsqueeze(-1)[:, :, None].expand(len(zsamples), x_t.shape[1], 
                                                                        x_t.shape[0]).permute(2, 1, 0)

                # Include features from start sequence for the target
                seq_encoding = best_model.seq_model(start_seq,  create_label(len(start_seq), sim=False).to(device))

                if type(seq_encoding) == tuple:
                    seq_encoding, _ = seq_encoding
                # logger.debug("----temp----")
                # logger.debug(seq_encoding.shape)

                seq_encoding = seq_encoding[:, -1:, :].repeat(1, x_t.shape[1], 1)
                # logger.debug("----- logger.debuging pl-----")
                xz = torch.cat([x_t, seq_encoding, zs_reshaped], dim=2)
                xz = xz.reshape(-1, xz.shape[2])
                
                # logger.debug("-----logger.debuging xz------")
                # logger.debug(xz.shape)

                output = best_model.decoder(xz)
                output_list.append(output.cpu().numpy())
        return np.concatenate(output_list, axis=0)

    train_, test_, stats_ = data_
    x_train_t, y_train_t, all_train_t = train_
    x_test_t, y_test_t, all_test_t = test_
    sim_means, sim_stds, rwd_means, rwd_stds = stats_
         
    y_pred_list = []
    test_mae_list = []
    for i in range(4):
        logger.debug("epoch {}".format(i))
        start_seq = all_test_t[:, :, :][:, :seq_steps, :].to(device).float()
        start_seq_train = all_train_t[:, :seq_steps, :].to(device).float()
        start_seq = all_test_t[:, :seq_steps, :].to(device).float()
        start_seq_train = all_train_t[:, :seq_steps, :].to(device).float()

        y_pred_t = test(x_train_t.float(), y_train_t.float(),
                        x_test_t.float(), start_seq_train, start_seq, seq_steps, best_model)
        
        y_pred_list.append(y_pred_t)
        mae = np.mean(np.abs(y_pred_t - y_test_t[:, seq_steps:, :].reshape(-1, 1).numpy()))
        # f.write("epoch :{}, mae:{} \n".format(i, mae)) 
        test_mae_list.append(mae)
    logger.info('MAE:{}'.format(np.mean(test_mae_list)))
    wandb.log({"test_mae": np.mean(test_mae_list)})
    # Create the mae_list directory if it doesn't exist
    # if not os.path.exists("DANN/mae_list"):
    mae_dir = main_dir + "/mae_list"
    os.makedirs(mae_dir, exist_ok=True)
      
    # logger.debug(os.path.exists("mae_list"))
    # Save the list to a pickle file
    path = mae_dir + "/mae_list_{}.pkl".format(datetime.now().strftime("%d_%m_%Y_%H:%M"))
    with open(path, "wb") as f:
        pickle.dump(test_mae_list, f)
        
    y_pred = np.concatenate(y_pred_list, 1)
    y_pred_un = y_pred * rwd_stds[0, 0, 0].item() + rwd_means[0, 0, 0].item()
    y_pred_mean = np.mean(y_pred_un, 1)
    y_pred_std = np.std(y_pred_un, 1)
    y_pred_lower = y_pred_mean - 2 * y_pred_std
    y_pred_upper = y_pred_mean + 2 * y_pred_std
    
    # if not os.path.exists("DANN/output_results"):
    output_results_dir = main_dir + "/output_results"
    os.makedirs(output_results_dir, exist_ok=True)
    
    with open(output_results_dir + '/y_pred_mean_{}.pkl'.format(datetime.now().strftime("%d_%m_%Y_%H:%M")), 'wb') as f:
        pickle.dump(y_pred_mean, f)
    with open(output_results_dir + '/y_pred_std_{}.pkl'.format(datetime.now().strftime("%d_%m_%Y_%H:%M")), 'wb') as f:
        pickle.dump(y_pred_std, f)
    with open(output_results_dir + '/y_pred_lower_{}.pkl'.format(datetime.now().strftime("%d_%m_%Y_%H:%M")), 'wb') as f:
        pickle.dump(y_pred_lower, f)
    with open(output_results_dir + '/y_pred_upper_{}.pkl'.format(datetime.now().strftime("%d_%m_%Y_%H:%M")), 'wb') as f:
        pickle.dump(y_pred_upper, f)
    un_MAE_results_dir = main_dir + "/un_MAE_results"
    # if not os.path.exists("DANN/un_MAE_results"):
    os.makedirs(un_MAE_results_dir, exist_ok=True)
    with open(un_MAE_results_dir + "/{}".format(datetime.now().strftime("%d_%m_%Y_%H:%M")), "w") as f:
        f.write("MAE for cohort {} is {}\n".format(cohort, np.mean(test_mae_list) * rwd_stds[0, 0, 0].item()))
    f.close()

    return y_pred_mean, y_pred_std, y_pred_lower, y_pred_upper


output = inference(args.cohort, args.seq_steps, all_data[1:], best_model, device)
logger.info("proceed to plotting.py for sample prediction")
logger.info("Finished this expereiment for cohort {} with sequence steps = {}, lr =® {}, n_poches = {}, patience = {}"
            .format(args.cohort, args.seq_steps, args.learning_rate, args.nepochs, args.patiences))
