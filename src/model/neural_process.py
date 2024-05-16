"""
Neural Process with seq model encoder + FF decoder
"""


import torch
import torch.nn as nn
import numpy as np


# reference: https://chrisorm.github.io/NGP.html
class REncoder(torch.nn.Module):
    """Encodes inputs of the form (x_i,y_i) into representations, r_i."""

    def __init__(
        self,
        in_dim,
        out_dim,
        seq_enc_hidden_size=10,
        seq_model=None,
        init_func=torch.nn.init.normal_,
    ):
        super(REncoder, self).__init__()
        self.l1_size = 4  # 16
        self.l2_size = 4  # 8

        self.l0 = seq_model

        # self.l0 = torch.nn.RNN(4, 10)
        # if seq_model:
        #     self.l0 = seq_model
        # else:
        #     self.l0 = torch.nn.RNN(seq_features, seq_enc_hidden_size)

        # self.l1 = torch.nn.Linear(in_dim, self.l1_size)

        self.l1 = torch.nn.Linear(in_dim + seq_enc_hidden_size, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, self.l2_size)
        self.l3 = torch.nn.Linear(self.l2_size, out_dim)
        self.a1 = torch.nn.Sigmoid()
        self.a2 = torch.nn.Sigmoid()

        if init_func is not None:
            # init_func(self.l0.weight)
            init_func(self.l1.weight)
            init_func(self.l2.weight)
            init_func(self.l3.weight)

    def forward(self, inputs, start_seq, labels=None):
        # TODO:  Adjust this for RNN
        if self.l0:
            if labels is not None:
                seq_encoding = self.l0(start_seq, labels)
            else:
                seq_encoding = self.l0(start_seq)

            if type(seq_encoding) == tuple:
                seq_encoding, _ = seq_encoding

            seq_encoding = seq_encoding[:, -1:, :].repeat(1, inputs.shape[1], 1)
            inputs_and_reps = torch.cat([inputs, seq_encoding], axis=2)
        else:
            inputs_and_reps = inputs

        inputs_and_reps = inputs_and_reps.reshape(-1, inputs_and_reps.shape[2])
        # inputs_and_reps = inputs.view(-1, 4)
        return self.l3(self.a2(self.l2(self.a1(self.l1(inputs_and_reps)))))


class ZEncoder(torch.nn.Module):
    """Takes an r representation and produces the mean & standard deviation of the
    normally distributed function encoding, z."""

    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):
        super(ZEncoder, self).__init__()
        self.m1_size = out_dim
        self.logvar1_size = out_dim

        self.m1 = torch.nn.Linear(in_dim, self.m1_size)
        self.logvar1 = torch.nn.Linear(in_dim, self.m1_size)

        if init_func is not None:
            init_func(self.m1.weight)
            init_func(self.logvar1.weight)

    def forward(self, inputs):
        # print(self.m1(inputs))
        # print("________")
        # print(self.logvar1(inputs))

        return self.m1(inputs), self.logvar1(inputs)


class Decoder(torch.nn.Module):
    """
    Takes the x star points, along with a 'function encoding', z, and makes predictions.
    """

    def __init__(
        self, in_dim, out_dim, seq_model=None, init_func=torch.nn.init.normal_
    ):
        super(Decoder, self).__init__()
        # self.l1_size = 2 #8
        # self.l2_size = 2 #16

        # self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        # self.l2 = torch.nn.Linear(self.l1_size, self.l2_size)
        # self.l3 = torch.nn.Linear(self.l2_size, out_dim)

        # if init_func is not None:
        #     init_func(self.l1.weight)
        #     init_func(self.l2.weight)
        #     init_func(self.l3.weight)

        # self.a1 = torch.nn.Sigmoid()
        # self.a2 = torch.nn.Sigmoid()

        self.l1_size = 8  # 16
        self.l2_size = 8  # 16
        self.l3_size = 8

        # self.l0 = torch.nn.RNN(4, 10):
        self.l0 = seq_model

        self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        # self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, self.l2_size)
        self.l3 = torch.nn.Linear(self.l2_size, self.l3_size)
        self.l4 = torch.nn.Linear(self.l3_size, out_dim)
        self.a1 = torch.nn.Sigmoid()
        self.a2 = torch.nn.Sigmoid()
        self.a3 = torch.nn.Sigmoid()

        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)
            init_func(self.l3.weight)
            init_func(self.l4.weight)

    def forward(self, xz):
        """x_pred: No. of data points, by x_dim
        z: No. of samples, by z_dim
        """
        return self.l4(self.a3(self.l3(self.a2(self.l2(self.a1(self.l1(xz)))))))


class DCRNNModel(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        r_dim,
        z_dim,
        device,
        seq_enc_hidden_size=32,
        seq_model=torch.nn.RNN(7, 32, batch_first=True),
        # seq_model= None
        init_func=torch.nn.init.normal_,
        # = device,
        encode_only=True,
    ):
        super().__init__()
        self.seq_model = seq_model

        if not seq_model:
            seq_enc_hidden_size = 0

        if seq_model is None:
            self.seq_model = torch.nn.RNN(
                x_dim + y_dim, seq_enc_hidden_size, batch_first=True
            )

        self.repr_encoder = REncoder(
            x_dim + y_dim,
            r_dim,
            seq_enc_hidden_size=seq_enc_hidden_size,
            seq_model=self.seq_model,
        )  # (x,y)->r
        self.z_encoder = ZEncoder(r_dim, z_dim)  # r-> mu, logvar
        self.decoder = Decoder(
            x_dim + z_dim + seq_enc_hidden_size, y_dim, seq_model=self.seq_model
        )  # (x*, z) -> y*
        self.z_mu_all = 0
        self.z_logvar_all = 0
        self.z_mu_context = 0
        self.z_logvar_context = 0
        self.zs = 0
        self.zdim = z_dim
        self.encode_only = encode_only
        self.device = device

    def data_to_z_params(self, x, y, start_seq, labels=None):
        """Helper to batch together some steps of the process."""
        xy = torch.cat([x, y], dim=2)
        rs = self.repr_encoder(xy, start_seq, labels)
        r_agg = rs.mean(dim=0)  # Average over samples
        return self.z_encoder(r_agg)  # Get mean and variance for q(z|...)

    def sample_z(self, mu, logvar, n=1):
        """Reparameterisation trick."""
        if n == 1:
            eps = torch.autograd.Variable(logvar.data.new(self.zdim).normal_()).to(
                self.device
            )
        else:
            eps = torch.autograd.Variable(logvar.data.new(n, self.zdim).normal_()).to(
                self.device
            )

        # std = torch.exp(0.5 * logvar)
        std = 0.1 + 0.9 * torch.sigmoid(logvar)
        return mu + std * eps.to(self.device)

    def KLD_gaussian(self):
        """Analytical KLD between 2 Gaussians."""
        mu_q, logvar_q, mu_p, logvar_p = (
            self.z_mu_all,
            self.z_logvar_all,
            self.z_mu_context,
            self.z_logvar_context,
        )

        std_q = 0.1 + 0.9 * torch.sigmoid(logvar_q)
        std_p = 0.1 + 0.9 * torch.sigmoid(logvar_p)
        p = torch.distributions.Normal(mu_p, std_p)
        q = torch.distributions.Normal(mu_q, std_q)
        return torch.distributions.kl_divergence(p, q).sum()

        # qs2 = (std_q)**2
        # ps2 = (std_p)**2
        # kld = (qs2/ps2 + ((mu_q-mu_p)**2)/ps2 + torch.log(ps2/qs2) - 1.0).sum()*0.5
        # return kld

        # kl_div = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / (torch.exp(logvar_p)) - 1.0 + logvar_p - logvar_q
        # kld = 0.5 * kl_div.sum()
        # return kld

    def make_posterior(self, x_t, start_seq, zs, seq_steps, labels=None):
        x_t = x_t[:, :seq_steps, :]
        zs_reshaped = (
            zs.unsqueeze(-1)[:, :, None]
            .expand(self.zs.shape[0], x_t.shape[1], x_t.shape[0])
            .permute(2, 1, 0)
        )

        # Include features from start sequence for the target

        if self.seq_model:
            if labels is not None:
                seq_encoding = self.seq_model(start_seq, labels)
            else:
                seq_encoding = self.seq_model(start_seq)

            if type(seq_encoding) == tuple:
                seq_encoding, _ = seq_encoding

            seq_encoding = seq_encoding[:, -1:, :].repeat(1, x_t.shape[1], 1)
            # pl = original_x_t[:,seq_steps:,:1]
            # print(pl.shape)

            xz = torch.cat([x_t, seq_encoding, zs_reshaped], dim=2)
            # print(xz.shape)
            # 6+32+16 = 54
        else:
            xz = torch.cat([x_t, zs_reshaped], dim=2)

        xz = xz.reshape(-1, xz.shape[2])

        return xz

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
        seq_steps=90,
        labels_c=None,
        labels_t=None,
        labels_ct=None,
    ):
        """ """
        self.z_mu_all, self.z_logvar_all = self.data_to_z_params(
            x_ct, y_ct, start_seq_ct, labels_ct
        )
        self.z_mu_context, self.z_logvar_context = self.data_to_z_params(
            x_c, y_c, start_seq_c, labels_c
        )
        self.zs = self.sample_z(self.z_mu_all, self.z_logvar_all)

        xz = self.make_posterior(x_t, start_seq_t, self.zs, seq_steps, labels_t)

        if self.encode_only:
            return xz
        else:
            return self.decoder(xz)


def random_split_context_target(x, y, n_context):
    """Helper function to split randomly into context and target"""
    x = x.cpu()
    y = y.cpu()
    ind = np.arange(x.shape[0])
    mask = np.random.choice(ind, size=n_context, replace=False)
    return (
        x[mask, :, :],
        y[mask, :, :],
        np.delete(x, mask, axis=0),
        np.delete(y, mask, axis=0),
        mask,
    )


def MAE(pred, target):
    loss = torch.abs(pred - target)
    return loss.mean()
