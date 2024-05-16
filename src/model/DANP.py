import torch
import sys
import numpy as np
from torch.autograd import Function
from torch import nn
import matplotlib.pyplot as plt
import os
from datetime import datetime
sys.path.append("..")
from model import neural_process_add_pl as nproc
from model.neural_process import MAE, random_split_context_target
from torch.utils import data

import argparse
import pickle

import logging
import wandb



class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grads):
        output = -ctx.alpha * grads

        return output, None


class DANP(nn.Module):
    def __init__(self, seq_steps, device, seq_dim_=32, r_dim_=4, z_dim_=16):
        super().__init__()

        r_dim = r_dim_
        z_dim = z_dim_

        x_dim = 6
        y_dim = 1 
        seq_dim = seq_dim_

        self.enc = nproc.DCRNNModel(x_dim, y_dim, r_dim, z_dim, device, seq_dim_).to(device)
        # logger.debug("self.enc")
        # logger.debug(self.enc)

        self.dec = nproc.Decoder(x_dim + z_dim + seq_dim + 1, y_dim).to(device)
        # logger.debug("self.dec")
        # logger.debug(self.dec)

        self.domain_classifier = nn.Sequential(
            nn.Linear(z_dim + x_dim + seq_dim + 1, 100),
            nn.BatchNorm1d(100),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        ).to(device)

    def forward(self, x_t, x_c, y_c, x_ct, y_ct, start_seq_c, start_seq_t, start_seq_ct, seq_steps, lambda_grl=1):
        features = self.enc(x_t, x_c, y_c, x_ct, y_ct, start_seq_c, start_seq_t, start_seq_ct, seq_steps)
        # logger.debug(features.shape)
        features_grl = GradientReversal.apply(features, lambda_grl)
        y_pred = self.dec(features)
        domain_pred = self.domain_classifier(features_grl)

        return y_pred, domain_pred


def initialize_DANP(seq_steps, cuda_number, seq_dim_=32, r_dim_=4, z_dim_=16):
    model = DANP(seq_steps, cuda_number, seq_dim_=seq_dim_, r_dim_=r_dim_, z_dim_=z_dim_)
    return model


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


def read_data(device, cohort='hrpci', subsample=0.02, bc=64):
    # data
    if cohort == 'hrpci': 
        rwd_train = torch.load("data/processed/train_real.pt")
        rwd_test = torch.load("data/processed/test_real.pt")
    else:
        raise ValueError("Invalid cohort specified. Please specify either 'hrpci' or implement others.")
    
    sim_data = torch.load(np.load("data/processed/sim_data.pt"))
        
    logger.info("finished loading sim data and real data for cohort {}".format(cohort))
    # note this source
    
    # Sim Data 
    np.random.seed(0)
    idx = np.arange(0, int(sim_data.shape[0] * 0.8))

    sim_data_train = sim_data[idx, ...]

    means = sim_data_train.mean(axis=(0, 1))[None, None, :]
    sim_stds = sim_data_train.std(axis=(0, 1))[None, None, :]

    x_n = (sim_data - means) / sim_stds

    x_train = x_n[idx][..., [0, 2, 3, 4, 5, 6]]

    y_train = x_n[idx][..., 1:2]
    # logger.debug("x_train_shape is {}".format(x_train.shape))
    # logger.debug("y_train_shape is {}".format(y_train.shape))
    all_train = x_n[idx]
    # logger.debug("all_train_shape is {}".format(all_train.shape))

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

    sim_loader_test = data.DataLoader(
        Dataset(x_test, y_test, all_test),
        batch_size=bc,
        shuffle=False,
        num_workers=4,
        persistent_workers=True, 
        pin_memory=True
    )

    # HRPCI/AMICGS
    np.random.seed(0)
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
    main_dir = "../DANP_results/{}".format(cohort)
    dist_plot_dir = main_dir + "/dis_b4_train"
    os.makedirs(dist_plot_dir, exist_ok=True)
    plot_distr(rwd_train[:, :, 0].reshape(-1), sim_data[:, :, 1].reshape(-1), cohort, dist_plot_dir)
 
    return (sim_loader_train, sim_loader_test, rwd_loader_train, rwd_loader_test), \
        (x_train_t, y_train_t, all_train_t), \
            (x_test_t, y_test_t, all_test_t), \
                (rwd_means, rwd_stds) 


def plot_distr(data1, data2, cohort, path):
    """Parameters:
    data1: real data.
    data2: sim data from LP Model.
    """
    import seaborn as sns
    import numpy as np

    # Create density plots for the two datasets with different colors
    sns.kdeplot(data=data1, color='blue', label='{} Train'.format(cohort))
    sns.kdeplot(data=data2, color='orange', label='LP Model')

    # Set the plot title and legend
    plt.legend()
    plt.xlabel('MAP mmHg')
    plt.ylabel('Density')
    plt.title('Distribution of Real Data and LP Model')
    plt.savefig(path + "/" + datetime.now().strftime("%d_%m_%Y_%H:%M"))
    wandb.log({'Distribution of Real Data and LP Model': plt.gcf()})
    plt.clf()


def training_DANP(cohort, sim_data_loader, train_data_loader, model, device, 
                  SEQ_STEPS, lr_=1e-3, n_epochs_=5000, patience_=200):
    logger.info(" ------ training starts ------")
    n_epochs = n_epochs_
    n_display = 2
    patience = patience_
    display_losses = []
    best_danp_model = None
    opt = torch.optim.Adam(model.parameters(), lr_)  # 1e-4 2e-5
    wait = 0
    min_loss = float('inf')
    main_dir = "../DANP_results/{}".format(cohort)
    model_dir = main_dir + "/trained_danp"
    loss_dir = main_dir + "/loss_plots"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    for t in np.arange(n_epochs):
        logger.debug("---epoch # {} starts---".format(t))
        opt.zero_grad()

        rwd_loader_train = train_data_loader
        sim_loader_train = sim_data_loader
        rwd_iter = iter(rwd_loader_train)

        n_batches = len(rwd_loader_train)
        losses = []
        for i in range(n_batches):
            x_train, y_train, all_train = next(iter(sim_loader_train))
            x_train_t, y_train_t, all_train_t = next(rwd_iter)

            p = max(1, float(t) / (5000))
            lambda_grl = 2 / (1 + np.exp(-10 * p)) - 1 
            # if t == 0:
            #     logger.debug("-------- logger.debuging lambda_grl -------")
            #     logger.debug(lambda_grl)
            
            x_context, y_context, x_target, y_target, mask = random_split_context_target(
                x_train, y_train, int(len(y_train) * 0.05)
            )

            x_context_t, y_context_t, x_target_t, y_target_t, mask_t = random_split_context_target(
                x_train_t, y_train_t, len(mask)
            )
            
            # Simulation data
            x_c = x_context.float().to(device)
            x_t = x_target.float().to(device)
            y_c = y_context.float().to(device)
            y_t = y_target.float().to(device)
            
            x_ct = torch.cat([x_c, x_t], dim=0).float().to(device)
            y_ct = torch.cat([y_c, y_t], dim=0).float().to(device)

            # Real data
            x_c_t = x_context_t.float().to(device)
            x_t_t = x_target_t.float().to(device)
            y_c_t = y_context_t.float().to(device)
            y_t_t = y_target_t.float().to(device)
            
            x_ct_t = torch.cat([x_c_t, x_t_t], dim=0).float().to(device)
            y_ct_t = torch.cat([y_c_t, y_t_t], dim=0).float().to(device)

            # context points
            seqs = all_train[:, :SEQ_STEPS, :].float().to(device)
            start_seq_c = seqs[mask, :, :]
            anti_mask = torch.ones(seqs.shape[0], dtype=bool)
            anti_mask[mask] = False
            # target points
            start_seq_t = seqs[anti_mask, :, :]
            start_seq_ct = torch.cat([start_seq_c, start_seq_t], axis=0)

            y_pred, domain_pred = model(
                x_t, x_c, y_c, x_ct, y_ct, start_seq_c, start_seq_t, start_seq_ct, SEQ_STEPS, lambda_grl 
            )

            domain_loss = torch.nn.NLLLoss()(
                domain_pred, torch.zeros(domain_pred.shape[0]).to(device).long()
            )

            # Real
            seqs = all_train_t[:, :SEQ_STEPS, :].float().to(device)
            # context points
            start_seq_c_t = seqs[mask_t, :, :]
            anti_mask = torch.ones(seqs.shape[0], dtype=bool)
            anti_mask[mask_t] = False
            # target points
            start_seq_t_t = seqs[anti_mask, :, :]
            start_seq_ct_t = torch.cat([start_seq_c_t, start_seq_t_t], axis=0)

            mae_loss = MAE(y_pred, y_t[:, SEQ_STEPS:, :].reshape(-1, 1))
            kld_sim = model.enc.KLD_gaussian()
            
            y_pred_t, domain_pred = model(
                x_t_t, x_c_t, y_c_t, x_ct_t, y_ct_t, start_seq_c_t, start_seq_t_t, start_seq_ct_t, SEQ_STEPS, lambda_grl
            )
            
            domain_loss_t = torch.nn.NLLLoss()(
                domain_pred, torch.ones(domain_pred.shape[0]).to(device).long()
            )

            loss_t = MAE(y_pred_t, y_t_t[:, SEQ_STEPS:, :].reshape(-1, 1)) + model.enc.KLD_gaussian()

            loss = mae_loss + kld_sim + loss_t + domain_loss_t + domain_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            losses.append(loss.item())

        cum_loss = np.mean(losses)
        display_losses.append(cum_loss)
        
        wandb.log({"train_loss": cum_loss})

        if cum_loss < min_loss:
            wait = 0
            best_danp_model = model
            min_loss = cum_loss
                
        elif cum_loss >= min_loss:
            wait += 1
            if wait == patience:
                logger.info('Early stopping at epoch: %d' % t)
                torch.save(best_danp_model, 
                           model_dir + "/final_best_danp_model_{}_loss{}_{}.pt"
                           .format(cohort, round(min_loss, 2), datetime.now().strftime("%d_%m_%Y_%H:%M")))
     
                plt.plot(display_losses)
                plt.xlabel('epoch number')
                plt.ylabel('normalized MAE ')
                plt.title('training loss')
                plt.savefig(loss_dir + "/final_loss_plots_epoch{}_{}"
                            .format(t, datetime.now().strftime("%d_%m_%Y_%H:%M")))
                plt.clf()
                return best_danp_model

        if (t % 100 == 0) and (t != 0):
            # torch.save(model, "../trained_danp/danp_pl_hrpci_epoch{}_loss{}.pt".format(t,round(cum_loss, 2)))
            torch.save(best_danp_model, model_dir + "/best_danp_{}_epoch{}_loss{}_{}.pt"
                       .format(cohort, t, round(min_loss, 2), 
                               datetime.now().strftime("%d_%m_%Y_%H:%M")))
            torch.save(model, model_dir + "/danp_{}_epoch{}_loss{}_{}.pt"
                       .format(cohort, t, round(cum_loss, 2), datetime.now().strftime("%d_%m_%Y_%H:%M")))
        
        if (t % n_display == 0) and (t != 0):
            logger.info('epoch:{}, epoch loss:{}, cur_min_loss:{}'.format(t, cum_loss, min_loss))
            plt.plot(display_losses)
            plt.xlabel('epoch number')
            plt.ylabel('normalized MAE ')
            plt.title('training loss')
            # plt.savefig("../loss_plots/n_display_epoch{}".format(t))
            plt.savefig(loss_dir + "/n_display_epoch{}_{}"
                        .format(t, datetime.now().strftime(datetime.now().strftime("%d_%m_%Y_%H:%M"))))
            plt.clf()
        else:
            if (t % 5 == 0) and (t != 0):
                logger.debug('epoch:{}, epoch loss:{}, cur_min_loss:{}'.format(t, cum_loss, min_loss))
    torch.save(best_danp_model, model_dir + "/final_best_danp_model_{}_loss{}_{}.pt"
               .format(cohort, round(min_loss, 2), datetime.now().strftime("%d_%m_%Y_%H:%M")))
    plt.plot(display_losses)
    plt.xlabel('epoch number')
    plt.ylabel('normalized MAE ')
    plt.title('{} training loss'.format(cohort))
    plt.savefig(loss_dir + "/final_loss_plots_epoch{}_{}".format(t, datetime.now().strftime("%d_%m_%Y_%H:%M")))
    plt.clf()

    return best_danp_model


def eval(seq_steps, data_, best_danp_model, device):

    def test(x_train, y_train, x_test, start_seq_train, start_seq_test, seq_steps, best_danp_model, device=device):
        original_x_test = x_test
        x_test = x_test[:, :seq_steps, :]
        with torch.no_grad():
            z_mu, z_logvar = best_danp_model.enc.data_to_z_params(x_train.to(device), y_train.to(device), start_seq_train)
            output_list = []
            for i in range(x_test.shape[0]):
                zsamples = best_danp_model.enc.sample_z(z_mu, z_logvar)
                start_seq = start_seq_test[i:i + 1, :, :].to(device)
                # logger.debug("------logger.debuging start_seq------")
                # logger.debug(start_seq.shape)
                
                x_t = x_test[i:i + 1, :, :].to(device)

                zs_reshaped = zsamples.unsqueeze(-1)[:, :, None].expand(zsamples.shape[0], x_t.shape[1], x_t.shape[0])
                zs_reshaped = zs_reshaped.permute(2, 1, 0)

            # Include features from start sequence for the target
                seq_encoding = best_danp_model.enc.seq_model(start_seq)
    
                if type(seq_encoding) == tuple:
                    seq_encoding, _ = seq_encoding
                # logger.debug("----temp----")
                # logger.debug(seq_encoding.shape)

                seq_encoding = seq_encoding[:, -1:, :].repeat(1, x_t.shape[1], 1)

                pl = original_x_test[i:i + 1, seq_steps:, :1].to(device)
                # logger.debug("----- logger.debuging pl-----")
                xz = torch.cat([pl, x_t, seq_encoding, zs_reshaped], dim=2)
                xz = xz.reshape(-1, xz.shape[2])
                
                # logger.debug("-----logger.debuging xz------")
                # logger.debug(xz.shape)

                output = best_danp_model.dec(xz)
                output_list.append(output.cpu().numpy())
                # logger.debug("logger.debuging output")
                # logger.debug(output.shape)
                # logger.debug("logger.debuging final")
                # logger.debug(np.concatenate(output_list, axis=0).shape)
        return np.concatenate(output_list, axis=0)
    
    loaders, train_, test_, stats_ = data_
    x_train_t, y_train_t, all_train_t = train_
    x_test_t, y_test_t, all_test_t = test_
    rwd_means, rwd_stds = stats_

    y_pred_list = []
    test_mae_list = []
    for i in range(4):

        start_seq = all_test_t[:, :seq_steps, :].to(device).float()
        start_seq_train = all_train_t[:, :seq_steps, :].to(device).float()

        y_pred_t = test(x_train_t.float(), y_train_t.float(), x_test_t.float(), 
                        start_seq_train, start_seq, seq_steps, best_danp_model)
        y_pred_list.append(y_pred_t.reshape([x_test_t.shape[0], seq_steps]))
        mae = np.mean(np.abs(y_pred_t - y_test_t[:, seq_steps:, :].reshape(-1, 1).numpy()))
        test_mae_list.append(mae)
        return y_pred_list, mae

def inference(cohort, seq_steps, data_, best_danp_model, device):
    logger.info("inference starting")
    main_dir = "../DANP_results/{}".format(cohort)
    
    y_pred_list, test_mae_list = eval(seq_steps, data_, best_danp_model, device)

    logger.info('MAE:{}'.format(np.mean(test_mae_list)))
    wandb.log({"test_mae": np.mean(test_mae_list)})
    
    # Create the mae_list directory if it doesn't exist
    mae_dir = main_dir + "/mae_list"
    os.makedirs(mae_dir, exist_ok=True)
      
    # Save the list to a pickle file
    path = mae_dir + "/mae_list_{}.pkl".format(datetime.now().strftime("%d_%m_%Y_%H:%M"))
    with open(path, "wb") as f:
        pickle.dump(test_mae_list, f)

    y_pred = np.mean(y_pred_list, axis=0)
    
    loaders, train_, test_, stats_ = data_
    x_train_t, y_train_t, all_train_t = train_
    x_test_t, y_test_t, all_test_t = test_
    rwd_means, rwd_stds = stats_

    gt = y_test_t[:,seq_steps:, :].numpy().squeeze() * rwd_stds[0, 0, 0].item()
    pr = y_pred.squeeze() * rwd_stds[0, 0, 0].item()
    y_pred_un = y_pred * rwd_stds[0, 0, 0].item() + rwd_means[0, 0, 0].item()

    y_pred_mean = y_pred 
    y_pred_std =  np.std(y_pred_list, 0)
    y_pred_lower = y_pred_mean - 2 * y_pred_std
    y_pred_upper = y_pred_mean + 2 * y_pred_std
    
    # if not os.path.exists("DANP/output_results"):
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
    # if not os.path.exists("DANP/un_MAE_results"):
    os.makedirs(un_MAE_results_dir, exist_ok=True)
    with open(un_MAE_results_dir + "/{}".format(datetime.now().strftime("%d_%m_%Y_%H:%M")), "w") as f:
        f.write("MAE for cohort {} is {}\n".format(cohort, np.mean(test_mae_list) * rwd_stds[0, 0, 0].item()))
    f.close()

    return y_pred_mean, y_pred_std, y_pred_lower, y_pred_upper


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    log_dir = "../logger_info_logs/DANP_log"
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_dir + '/experiment_DANP_{}.log'
                                    .format(datetime.now().strftime("%d_%m_%Y_%H:%M")))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    parser = argparse.ArgumentParser(description='Train a DANP model on a specified cohort using PyTorch.')
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

    wandb.init(
        project="abiomed",
        group="danp",
        config=vars(args)
    )

    device = torch.device(args.cuda_number)
    model = initialize_DANP(args.seq_steps, device, seq_dim_=args.seq_dim, r_dim_=args.r_dim, z_dim_=args.z_dim)
    all_data = read_data(device, args.cohort, args.subsample, args.batch_size)
    logger.info("finished reading data for cohort {}".format(args.cohort))
    loader = all_data[0]
    
    trained_model = training_DANP(args.cohort, loader[0], loader[2], model, device, args.seq_steps, 
                                  lr_=args.learning_rate, 
                                  n_epochs_=args.nepochs, patience_=args.patiences)
   
    output = inference(args.cohort, args.seq_steps, all_data, trained_model, device)
    logger.info("training finised and all results saved")
    logger.info("proceed to plotting.py for sample prediction")
    logger.info("Finished experiment for cohort {} with sequence steps = {}, lr =® {}, n_poches = {}, patience = {}"
                .format(args.cohort, args.seq_steps, args.learning_rate, args.nepochs, args.patiences))
