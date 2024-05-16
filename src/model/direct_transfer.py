# importing
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
from util import train as util_train
from util import split_data

# from model.LMUseq2seq import CondLMUSeq2Seq
from p_level_mapping import PLevels

# %load_ext autoreload
# %autoreload 2
# device = torch.device("cuda:0")
from torch.utils import data

import argparse
import pickle
import wandb

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(message)s")

# formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
if not os.path.exists("../logger_info_logs"):
    os.mkdir("../logger_info_logs")

file_handler = logging.FileHandler("../logger_info_logs/experiment_direct_transfer.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


parser = argparse.ArgumentParser(description="DANN Implementation")
parser.add_argument(
    "-seq_steps",
    "--seq_steps",
    type=int,
    metavar="",
    required=True,
    help="lenght of input and output",
)
parser.add_argument(
    "-cuda",
    "--cuda_number",
    type=str,
    metavar="",
    required=True,
    help="specify cuda number",
)
parser.add_argument(
    "-seq_dim", "--seq_dim", type=int, metavar="", default=32, help="specify seq dim"
)
parser.add_argument(
    "-r_dim", "--r_dim", type=int, metavar="", default=4, help="specify r dim"
)
parser.add_argument(
    "-z_dim", "--z_dim", type=int, metavar="", default=16, help="specify z dim"
)
parser.add_argument(
    "-bc",
    "--batch_size",
    type=int,
    metavar="<size>",
    default=64,
    help="Specify the batch size.",
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    metavar="",
    default=1e-3,
    help="specify learning rate",
)
parser.add_argument(
    "-cohort", "--cohort", type=str, metavar="", default="hrpci", help="specify cohort"
)
parser.add_argument(
    "-nepochs",
    "--nepochs",
    type=int,
    metavar="",
    default=5000,
    help="specify epoch numbers",
)
parser.add_argument(
    "-patience",
    "--patiences",
    type=int,
    metavar="",
    default=200,
    help="specify patiences",
)
parser.add_argument(
    "-subsample",
    "--subsample",
    type=float,
    metavar="<float>",
    default=0.02,
    help="Specify the subsample ratio of the cohort (between 0 and 1).",
)

# group = parser.add_mutually_exclusive_group()
# group.add_argument('-q', '--quiet', action = 'store_true', help= 'logger.debug quiet')
# group.add_argument('-v', '--verbose', action = 'store_true', help= 'logger.debug verbose')
args = parser.parse_args()

wandb.init(project="abiomed-dt", config=vars(args))
# wandb.init(project="abiomed-dt", mode="disabled")


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grads):
        output = -ctx.alpha * grads

        return output, None


class DANN(nn.Module):
    def __init__(self, seq_steps, device, seq_dim_=32, r_dim_=4, z_dim_=16):
        super().__init__()

        # r_dim = 4 # can be anything
        # z_dim = 16 #4
        # x_dim = 4
        r_dim = r_dim_
        z_dim = z_dim_

        x_dim = 6
        y_dim = 1
        seq_dim = seq_dim_

        self.enc = nproc.DCRNNModel(x_dim, y_dim, r_dim, z_dim, device).to(device)
        # logger.debug(self.enc)

        self.dec = nproc.Decoder(x_dim + z_dim + seq_dim + 1, y_dim).to(device)
        # logger.debug(self.dec)

        self.domain_classifier = nn.Sequential(
            nn.Linear(z_dim + x_dim + seq_dim + 1, 100),
            nn.BatchNorm1d(100),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1),
        ).to(device)

    def forward(
        self,
        x_t,
        x_c,
        y_c,
        x_ct,
        y_ct,
        start_seq_c,
        start_seq_t,
        start_seq_ct,
        seq_steps,
        lambda_grl=1,
    ):
        features = self.enc(
            x_t, x_c, y_c, x_ct, y_ct, start_seq_c, start_seq_t, start_seq_ct, seq_steps
        )
        # logger.debug(features.shape)
        features_grl = GradientReversal.apply(features, lambda_grl)
        y_pred = self.dec(features)
        domain_pred = self.domain_classifier(features_grl)

        return y_pred, domain_pred


def initialize_DANN(seq_steps, cuda_number, seq_dim_=32, r_dim_=4, z_dim_=16):
    # r_dim = 4 # can be anything
    # z_dim = 16 #4
    # x_dim = 4
    r_dim = r_dim_
    z_dim = z_dim_

    x_dim = 6
    y_dim = 1
    seq_dim = seq_dim_

    # logger.debug(self.dec)
    model = DANN(
        seq_steps, cuda_number, seq_dim_=seq_dim_, r_dim_=r_dim_, z_dim_=z_dim_
    )
    model.enc = nproc.DCRNNModel(x_dim, y_dim, r_dim, z_dim, device).to(device)
    model.dec = nproc.Decoder(x_dim + z_dim + seq_dim + 1, y_dim).to(device)

    return model


def read_data(device, cohort="hrpci", subsample=0.02, bc=64):
    # data
    if cohort == "hrpci":
        rwd_train = torch.load("data/processed/rwd_at_new_pp_train_hrpci.pt")
        rwd_test = torch.load("data/processed/rwd_at_new_pp_test_hrpci.pt")
    if cohort == "amicgs":
        import random
        import math

        random.seed(1551)
        rwd_train = torch.load("data/processed/rwd_at_new_pp_train_amicgs.pt")
        rwd_test = torch.load("data/processed/rwd_at_new_pp_test_amicgs.pt")
        M = int(len(rwd_train) * subsample)
        interval = len(rwd_train) // M
        train_sub_idx = [range(len(rwd_train))[i * interval] for i in range(M)]
        test_sub_idx = list(range(0, int(len(rwd_test) * subsample)))
        idx_dir = "../direct_transfer_results/{}/sub_idx".format(cohort)
        os.makedirs(idx_dir, exist_ok=True)
        with open(idx_dir + "/" + datetime.now().strftime("%d_%m_%Y_%H:%M"), "wb") as f:
            pickle.dump(test_sub_idx, f)

        rwd_train = rwd_train[train_sub_idx]  # Subsample in loader
        rwd_test = rwd_test[test_sub_idx]
    logger.debug("rwd_train_shape is {}".format(rwd_train.shape))
    logger.debug("rwd_test_shape is {}".format(rwd_test.shape))

    # note this source
    sim_data = torch.from_numpy(np.load("data/processed/sim_data.npy"))

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
        pin_memory=True,
    )

    x_test = np.delete(x_n[:, :, [0, 2, 3, 4, 5, 6]], idx, axis=0)
    y_test = np.delete(x_n[:, :, 1:2], idx, axis=0)
    all_test = np.delete(x_n, idx, axis=0)
    # logger.debug("x_test_shape is {}".format(x_test.shape))
    # logger.debug("y_test_shape is {}".format(y_test.shape))
    # logger.debug("all_test_shape is {}".format(all_test.shape))

    sim_loader_test = data.DataLoader(
        Dataset(x_test, y_test, all_test),
        batch_size=bc,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    # HRPCI/AMICGS

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
        pin_memory=True,
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
        pin_memory=True,
    )
    return (
        (sim_loader_train, sim_loader_test, rwd_loader_train, rwd_loader_test),
        (x_train_t, y_train_t, all_train_t),
        (x_test_t, y_test_t, all_test_t),
        (rwd_means, rwd_stds),
    )


def training_DANN(
    sim_data_loader,
    train_data_loader,
    model,
    device,
    SEQ_STEPS,
    lr_=1e-3,
    n_epochs_=5000,
    patience_=200,
):
    logger.info("training starts")
    n_epochs = n_epochs_
    n_display = 50
    patience = patience_
    display_losses = []
    best_model = None
    opt = torch.optim.Adam(model.parameters(), lr_)  # 1e-4 2e-5
    wait = 0
    min_loss = float("inf")
    if not os.path.exists("../direct_transfer_results/"):
        os.mkdir("../direct_transfer_results/")
    if not os.path.exists("../direct_transfer_results/model"):
        os.mkdir("../direct_transfer_results/model")
    if not os.path.exists("../direct_transfer_results/loss_plots"):
        os.mkdir("../direct_transfer_results/loss_plots")

    for t in np.arange(n_epochs):
        logger.debug("---epoch # {} starts---".format(t))
        opt.zero_grad()

        rwd_loader_train = train_data_loader
        sim_loader_train = sim_data_loader
        # sim_iter = iter(sim_loader_train)
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
            # Generate data and process
            (
                x_context,
                y_context,
                x_target,
                y_target,
                mask,
            ) = random_split_context_target(x_train, y_train, int(len(y_train) * 0.05))

            (
                x_context_t,
                y_context_t,
                x_target_t,
                y_target_t,
                mask_t,
            ) = random_split_context_target(x_train_t, y_train_t, len(mask))

            # simulation data
            x_c = x_context.float().to(device)
            x_t = x_target.float().to(device)
            y_c = y_context.float().to(device)
            y_t = y_target.float().to(device)
            # if t == 0:
            #     logger.debug("-------- logger.debuging x_c -------")
            #     logger.debug(x_c.shape)
            #     logger.debug("-------- logger.debuging x_t -------")
            #     logger.debug(x_t.shape)
            #     logger.debug("-------- logger.debuging y_c -------")
            #     logger.debug(y_c.shape)
            #     logger.debug("-------- logger.debuging y_t -------")
            #     logger.debug(y_t.shape)

            x_ct = torch.cat([x_c, x_t], dim=0).float().to(device)
            y_ct = torch.cat([y_c, y_t], dim=0).float().to(device)

            # real data
            # x_c_t = x_context_t.float().to(device)
            # x_t_t = x_target_t.float().to(device)
            # y_c_t = y_context_t.float().to(device)
            # y_t_t = y_target_t.float().to(device)

            # x_ct_t = torch.cat([x_c_t, x_t_t], dim=0).float().to(device)
            # y_ct_t = torch.cat([y_c_t, y_t_t], dim=0).float().to(device)

            # if t == 0:
            #     logger.debug("-------- logger.debuging x_ct -------")
            #     logger.debug(x_ct.shape)
            #     logger.debug("-------- logger.debuging y_ct -------")
            #     logger.debug(y_ct.shape)
            #     logger.debug("-------- logger.debuging x_c_t -------")
            #     logger.debug(x_c_t.shape)
            #     logger.debug("-------- logger.debuging x_t_t -------")
            #     logger.debug(x_t_t.shape)
            #     logger.debug("-------- logger.debuging x_ct_t -------")
            #     logger.debug(x_ct_t.shape)
            #     logger.debug("-------- logger.debuging y_ct_t -------")
            #     logger.debug(y_ct_t.shape)
            # Simulation
            # context points
            seqs = all_train[:, :SEQ_STEPS, :].float().to(device)
            start_seq_c = seqs[mask, :, :]
            anti_mask = torch.ones(seqs.shape[0], dtype=bool)
            anti_mask[mask] = False
            # target points
            start_seq_t = seqs[anti_mask, :, :]
            start_seq_ct = torch.cat([start_seq_c, start_seq_t], axis=0)
            # if t == 0:
            #     logger.debug("-------- logger.debuging start seqs_c-------")
            #     logger.debug(start_seq_c.shape)
            #     logger.debug("-------- logger.debuging start seqs_t-------")
            #     logger.debug(start_seq_t.shape)
            #     logger.debug("-------- logger.debuging start seqs_ct-------")
            #     logger.debug(start_seq_ct.shape)

            # y_pred = y_t[:,:SEQ_STEPS,:].reshape(-1, 1)
            y_pred, _ = model(
                x_t,
                x_c,
                y_c,
                x_ct,
                y_ct,
                start_seq_c,
                start_seq_t,
                start_seq_ct,
                SEQ_STEPS,
                lambda_grl,
            )

            # logger.debug("--------y_pred-------")
            # logger.debug(y_pred.shape)
            # logger.debug("--------domain_pred-------")
            # # 61 * 180 = 10890
            # logger.debug(domain_pred.shape)

            # domain_loss = torch.nn.NLLLoss()(
            #     domain_pred, torch.zeros(domain_pred.shape[0]).to(device).long()
            # )

            # # Real
            # seqs = all_train_t[:, :SEQ_STEPS, :].float().to(device)
            # # context points
            # start_seq_c_t = seqs[mask_t,:,:]
            # anti_mask = torch.ones(seqs.shape[0], dtype=bool)
            # anti_mask[mask_t] = False
            # # target points
            # start_seq_t_t = seqs[anti_mask,:,:]
            # start_seq_ct_t = torch.cat([start_seq_c_t, start_seq_t_t], axis=0)
            # # if t == 0:
            # #     logger.debug("-------- logger.debuging start seqs_c_t-------")
            # #     logger.debug(start_seq_c_t.shape)
            # #     logger.debug("-------- logger.debuging start seqs_t_t-------")
            # #     logger.debug(start_seq_t_t.shape)
            # #     logger.debug("-------- logger.debuging start seqs_ct_t-------")
            # #     logger.debug(start_seq_ct_t.shape)

            mae_loss = MAE(y_pred, y_t[:, SEQ_STEPS:, :].reshape(-1, 1))
            kld_sim = model.enc.KLD_gaussian()

            # y_pred_t, domain_pred = model(
            #     x_t_t, x_c_t, y_c_t, x_ct_t, y_ct_t, start_seq_c, start_seq_t, start_seq_ct, SEQ_STEPS, lambda_grl
            # )

            # domain_loss_t = torch.nn.NLLLoss()(
            #     domain_pred, torch.ones(domain_pred.shape[0]).to(device).long()
            # )

            # loss_t = MAE(y_pred_t, y_t_t[:, SEQ_STEPS:, :].reshape(-1, 1)) + model.enc.KLD_gaussian()

            loss = (
                kld_sim + mae_loss
            )  # mae_loss + kld_sim + loss_t + domain_loss_t + domain_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # 10
            opt.step()
            losses.append(loss.item())

            # if t % n_display ==0:
            #     logger.debug(
            #         'epoch:', t,
            #         'total:', loss.item(),
            #         'mae:', mae_loss.item(),
            #         'kld:', kld_sim.item(),
            #         'mae_t', loss_t.item(),
            #         'domain_loss:', domain_loss.item(),
            #         'domain_loss_t:', domain_loss_t.item())

        cum_loss = np.mean(losses)
        wandb.log({"loss": cum_loss})
        display_losses.append(cum_loss)

        if cum_loss < min_loss:
            wait = 0
            best_model = model
            min_loss = cum_loss

        elif cum_loss >= min_loss:
            wait += 1
            if wait == patience:
                logger.info("Early stopping at epoch: %d" % t)
                # torch.save(best_model, "../direct_transfer_results/model/best_model_loss{}.pt"
                #           .format(round(min_loss, 2)))
                torch.save(
                    best_model,
                    "../direct_transfer_results/model/final_best_model_loss{}_{}.pt".format(
                        round(min_loss, 2), datetime.now().strftime("%d_%m_%Y_%H:%M")
                    ),
                )

                plt.plot(display_losses)
                plt.xlabel("epoch number")
                plt.ylabel("normalized MAE ")
                plt.title("training loss")
                plt.savefig(
                    "../direct_transfer_results/loss_plots/loss_plots_epoch{}_{}".format(
                        t, datetime.now().strftime("%d_%m_%Y_%H:%M")
                    )
                )
                return best_model

        if t % 100 == 0:
            # torch.save(model, "../direct_transfer_results/model/dann_pl_hrpci_epoch{}_loss{}.pt"
            #            .format(t,round(cum_loss, 2)))
            torch.save(
                best_model,
                "../direct_transfer_results/model/best_dann_pl_hrpci_epoch{}_loss{}_{}.pt".format(
                    t, round(min_loss, 2), datetime.now().strftime("%d_%m_%Y_%H:%M")
                ),
            )
            torch.save(
                model,
                "../direct_transfer_results/model/dann_pl_hrpci_epoch{}_loss{}_{}.pt".format(
                    t, round(cum_loss, 2), datetime.now().strftime("%d_%m_%Y_%H:%M")
                ),
            )
        if t % n_display == 0:
            logger.info(
                "epoch:{}, epoch loss:{}, cur_min_loss:{}".format(t, cum_loss, min_loss)
            )
            plt.plot(display_losses)
            plt.xlabel("epoch number")
            plt.ylabel("normalized MAE ")
            plt.title("training loss")
            # plt.savefig("../direct_transfer_results/loss_plots/n_display_epoch{}".format(t))
            plt.savefig(
                "../direct_transfer_results/loss_plots/n_display_epoch{}_{}".format(
                    t,
                    datetime.now().strftime(datetime.now().strftime("%d_%m_%Y_%H:%M")),
                )
            )
        else:
            if t % 5 == 0:
                logger.debug(
                    "epoch:{}, epoch loss:{}, cur_min_loss:{}".format(
                        t, cum_loss, min_loss
                    )
                )
    torch.save(
        best_model,
        "../direct_transfer_results/model/final_best_model_loss{}_{}.pt".format(
            round(min_loss, 2), datetime.now().strftime("%d_%m_%Y_%H:%M")
        ),
    )
    plt.plot(display_losses)
    plt.xlabel("epoch number")
    plt.ylabel("normalized MAE ")
    plt.title("training loss")
    plt.savefig(
        "../direct_transfer_results/loss_plots/loss_plots_epoch{}_{}".format(
            t, datetime.now().strftime("%d_%m_%Y_%H:%M")
        )
    )

    return best_model


def inference(seq_steps, data_, best_model, device):
    logger.info("inference starting")

    def test(
        x_train,
        y_train,
        x_test,
        start_seq_train,
        start_seq_test,
        seq_steps,
        best_model,
        device=device,
    ):
        original_x_test = x_test
        x_test = x_test[:, :seq_steps, :]
        with torch.no_grad():
            z_mu, z_logvar = best_model.enc.data_to_z_params(
                x_train.to(device), y_train.to(device), start_seq_train
            )
            output_list = []
            for i in range(x_test.shape[0]):
                zsamples = best_model.enc.sample_z(z_mu, z_logvar)
                start_seq = start_seq_test[i:i + 1, :, :].to(device)
                # logger.debug("------logger.debuging start_seq------")
                # logger.debug(start_seq.shape)

                x_t = x_test[i:i + 1, :, :].to(device)

                zs_reshaped = (
                    zsamples.unsqueeze(-1)[:, :, None]
                    .expand(zsamples.shape[0], x_t.shape[1], x_t.shape[0])
                    .permute(2, 1, 0)
                )

                # Include features from start sequence for the target
                seq_encoding = best_model.enc.seq_model(start_seq[..., :7])

                if type(seq_encoding) == tuple:
                    seq_encoding, _ = seq_encoding
                # logger.debug("----temp----")
                # logger.debug(seq_encoding.shape)

                seq_encoding = seq_encoding[:, -1:, :].repeat(1, x_t.shape[1], 1)
                #             logger.debug("-----logger.debuging xt -------")

                #             # 1*180*6
                #             logger.debug(x_t.shape)

                #             logger.debug("-----logger.debuging seq_encoding-------")
                #             # 1 * 180 * 32
                #             logger.debug(seq_encoding.shape)

                #             logger.debug("----- logger.debuging zs_reshaped-----")
                #             # 1 * 180 * 16
                #             logger.debug(zs_reshaped.shape)

                pl = original_x_test[i:i + 1, seq_steps:, :1].to(device)
                # logger.debug("----- logger.debuging pl-----")
                xz = torch.cat([pl, x_t, seq_encoding, zs_reshaped], dim=2)
                xz = xz.reshape(-1, xz.shape[2])

                # logger.debug("-----logger.debuging xz------")
                # logger.debug(xz.shape)

                output = best_model.dec(xz)
                output_list.append(output.cpu().numpy())
                # logger.debug("logger.debuging output")
                # logger.debug(output.shape)
                # logger.debug("logger.debuging final")
                # logger.debug(np.concatenate(output_list, axis=0).shape)
        return np.concatenate(output_list, axis=0)

    train_, test_, stats_ = data_
    x_train_t, y_train_t, all_train_t = train_
    x_test_t, y_test_t, all_test_t = test_
    rwd_means, rwd_stds = stats_
    #     logger.debug("rwd_means")
    #     logger.debug(torch.isnan(rwd_means).any().item())

    #     logger.debug("rwd_stds")
    #     logger.debug(torch.isnan(rwd_stds).any().item())

    #     logger.debug("x_test_t_nan")
    #     logger.debug(torch.isnan(x_test_t).any().item())

    #     logger.debug("y_test_t_nan")
    #     logger.debug(torch.isnan(y_test_t).any().item())

    y_pred_list = []
    test_mae_list = []
    # y_train_list = []
    # train_mae_list = []
    # with open("mae_list", "w") as f:
    for i in range(20):
        logger.debug("epoch {}".format(i))
        start_seq = all_test_t[:, :, :][:, :seq_steps, :].to(device).float()
        start_seq_train = all_train_t[:, :seq_steps, :].to(device).float()
        start_seq = all_test_t[:, :seq_steps, :].to(device).float()
        start_seq_train = all_train_t[:, :seq_steps, :].to(device).float()

        y_pred_t = test(
            x_train_t.float(),
            y_train_t.float(),
            x_test_t.float(),
            start_seq_train,
            start_seq,
            seq_steps,
            best_model,
        )

        y_pred_list.append(y_pred_t)
        # logger.debug(len(y_pred_t))
        # logger.debug(y_test_t.shape)
        # logger.debug(np.abs(y_pred_t - y_test_t[:, seq_steps:,:].reshape(-1,1).numpy()))
        # logger.debug("mean")
        # logger.debug(np.nanmean(np.abs(y_pred_t - y_test_t[:, seq_steps:,:].reshape(-1,1).numpy())))
        mae = np.mean(
            np.abs(y_pred_t - y_test_t[:, seq_steps:, :].reshape(-1, 1).numpy())
        )
        # f.write("epoch :{}, mae:{} \n".format(i, mae))
        test_mae_list.append(mae)
    logger.info("MAE:{}".format(np.mean(test_mae_list)))
    wandb.log({"test_mae": np.mean(test_mae_list)})
    # Create the mae_list directory if it doesn't exist
    if not os.path.exists("../direct_transfer_results/mae_list"):
        os.mkdir("../direct_transfer_results/mae_list")

    # logger.debug(os.path.exists("mae_list"))
    # Save the list to a pickle file
    path = "../direct_transfer_results/mae_list/mae_list_{}.pkl".format(
        datetime.now().strftime("%d_%m_%Y_%H:%M")
    )
    with open(path, "wb") as f:
        pickle.dump(test_mae_list, f)

    # Load the list from the pickle file
    # with open(path, "rb") as f:
    #     loaded_list = pickle.load(f)
    #     logger.debug("logger.debuging loading")
    #     logger.debug(loaded_list)

    y_pred = np.concatenate(y_pred_list, 1)
    # y_pred_train = np.concatenate(y_train_list,1)

    # y_pred_train_un = y_pred_train*std_MAP + mean_MAP
    y_pred_un = y_pred * rwd_stds[0, 0, 0].item() + rwd_means[0, 0, 0].item()
    # logger.debug(y_pred_un.shape)
    # y_train_un = y_train*std_MAP + mean_MAP
    # y_test_un = y_test_t * rwd_stds[0,0,0] + rwd_means[0,0,0]
    # y_test_un = y_test_t[:, :, 0] * rwd_stds[0, 0, 0].item() + rwd_means[0, 0, 0].item()
    y_pred_mean = np.mean(y_pred_un, 1)
    y_pred_std = np.std(y_pred_un, 1)
    y_pred_lower = y_pred_mean - 2 * y_pred_std
    y_pred_upper = y_pred_mean + 2 * y_pred_std

    if not os.path.exists("../direct_transfer_results/output_results"):
        os.mkdir("../direct_transfer_results/output_results")

    with open(
        "../direct_transfer_results/output_results/y_pred_mean_{}.pkl".format(
            datetime.now().strftime("%d_%m_%Y_%H:%M")
        ),
        "wb",
    ) as f:
        pickle.dump(y_pred_mean, f)
    with open(
        "../direct_transfer_results/output_results/y_pred_std_{}.pkl".format(
            datetime.now().strftime("%d_%m_%Y_%H:%M")
        ),
        "wb",
    ) as f:
        pickle.dump(y_pred_std, f)
    with open(
        "../direct_transfer_results/output_results/y_pred_lower_{}.pkl".format(
            datetime.now().strftime("%d_%m_%Y_%H:%M")
        ),
        "wb",
    ) as f:
        pickle.dump(y_pred_lower, f)
    with open(
        "../direct_transfer_results/output_results/y_pred_upper_{}.pkl".format(
            datetime.now().strftime("%d_%m_%Y_%H:%M")
        ),
        "wb",
    ) as f:
        pickle.dump(y_pred_upper, f)
    if not os.path.exists("../direct_transfer_results/un_MAE_results"):
        os.mkdir("../direct_transfer_results/un_MAE_results")
    with open(
        "../direct_transfer_results/un_MAE_results/{}".format(
            datetime.now().strftime("%d_%m_%Y_%H:%M")
        ),
        "w",
    ) as f:
        f.write(
            "MAE is {} \n".format(np.mean(test_mae_list) * rwd_stds[0, 0, 0].item())
        )
    f.close()

    return y_pred_mean, y_pred_std, y_pred_lower, y_pred_upper


# train_, test_, stats_ =
# x_train_t, y_train_t, all_train_t = train_
# x_test_t, y_test_t, all_test_t = test_
# rwd_means, rwd_stds = stats_

# def to_plot_specific_samp(data_, output, i):
#     p_levels = PLevels()
#     train_, test_, stats_ = data_
#     # x_train_t, y_train_t, all_train_t = train_
#     x_test_t, y_test_t, all_test_t = test_
#     rwd_means, rwd_stds = stats_
#     y_pred_mean, y_pred_std, y_pred_lower, y_pred_upper = output
#     logger.debug("----now is plotting sample {}-----".format(i))
#     y_t = y_test_t[i, :,0]*rwd_stds[0,0,0].item() + rwd_means[0,0,0].item()
#     # logger.debug(y_t.shape)
#     p_speed = x_test_t[i,:,0]*rwd_stds[0,0,1].item() + rwd_means[0,0,1].item()
#     p = p_levels.calculate_p_level(p_speed*10, pump_version=10)


#     x = range(y_t.shape[0])
#     plt.figure()
#     plt.plot(range(90, 180), y_pred_mean.reshape((-1,90))[i,:],color ='orange',label = "simulated")
#     plt.plot(x, y_t, color='blue',label = "real")

#     # plt.legend(title = 'Color', labels = ['Real', ' Simulated'])
#     ax = plt.gca()
#     ax.axvline(x=90, color='gray', linestyle='--')
#     leg = ax.legend()
#     plt.fill_between(range(90, 180), y_pred_upper.reshape((-1,90))[i,:], 
#                      y_pred_lower.reshape((-1,90))[i,:], color='b', alpha=0.5)
#     ax.set_ylabel('MAP')
#     ax.set_xlabel('Time')
#     ax2 = ax.twinx()
#     p2, = ax2.plot(x, p, color='green', label = 'p-level')
#     leg = ax2.legend(bbox_to_anchor=(1.05, 1), loc='center left')
#     ax2.set_ylabel('P_Level')
#     plt.show()
#     plt.savefig("direct_transfer/loss_plots/prediction_sample{}".format(i))


if __name__ == "__main__":
    device = torch.device(args.cuda_number)
    model = initialize_DANN(
        args.seq_steps,
        device,
        seq_dim_=args.seq_dim,
        r_dim_=args.r_dim,
        z_dim_=args.z_dim,
    )
    all_data = read_data(device, args.cohort, args.subsample, args.batch_size)
    logger.info("finished reading data")
    loader = all_data[0]
    data = all_data[1:]

    trained_model = training_DANN(
        loader[0],
        loader[2],
        model,
        device,
        args.seq_steps,
        lr_=args.learning_rate,
        n_epochs_=args.nepochs,
        patience_=args.patiences,
    )
    # logger.debug(trained_model)

    output = inference(args.seq_steps, data, trained_model, device)
    logger.info("training finised and all results saved")
    logger.info("proceed to plotting.py for sample prediction")
    logger.info(
        "Finished this expereiment for cohort {} with sequence steps = {}, lr = {}, n_poches = {}, patience = {}"
        .format(
            args.cohort,
            args.seq_steps,
            args.learning_rate,
            args.nepochs,
            args.patiences,
        )
    )
