import numpy as np
import torch
from torch import nn


# reference: https://chrisorm.github.io/NGP.html
class REncoder(torch.nn.Module):
    """Encodes inputs of the form (x_i,y_i) into representations, r_i."""

    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):
        super(REncoder, self).__init__()
        self.l1_size = 4  # 16
        self.l2_size = 4  # 8

        self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, self.l2_size)
        self.l3 = torch.nn.Linear(self.l2_size, out_dim)
        self.a1 = torch.nn.Sigmoid()
        self.a2 = torch.nn.Sigmoid()

        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)
            init_func(self.l3.weight)

    def forward(self, inputs):
        return self.l3(self.a2(self.l2(self.a1(self.l1(inputs)))))


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
        return self.m1(inputs), self.logvar1(inputs)


class Decoder(torch.nn.Module):
    """
    Takes the x star points, along with a 'function encoding', z, and makes predictions.
    """

    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):
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

        self.l1 = torch.nn.Linear(in_dim, self.l1_size)
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

    def forward(self, x_pred, z):
        """x_pred: No. of data points, by x_dim
        z: No. of samples, by z_dim
        """
        zs_reshaped = (
            z.unsqueeze(-1).expand(z.shape[0], x_pred.shape[0]).transpose(0, 1)
        )
        xpred_reshaped = x_pred

        xz = torch.cat([xpred_reshaped, zs_reshaped], dim=1)

        # return self.l3(self.a2(self.l2(self.a1(self.l1(xz))))).squeeze(-1)
        return self.l4(self.a3(self.l3(self.a2(self.l2(self.a1(self.l1(xz)))))))


def MAE(pred, target):
    loss = torch.abs(pred - target)
    return loss.mean()


class DCRNNModel(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, z_dim, init_func=torch.nn.init.normal_):
        super().__init__()
        self.repr_encoder = REncoder(x_dim + y_dim, r_dim)  # (x,y)->r
        self.z_encoder = ZEncoder(r_dim, z_dim)  # r-> mu, logvar
        self.decoder = Decoder(x_dim + z_dim, y_dim)  # (x*, z) -> y*
        self.z_mu_all = 0
        self.z_logvar_all = 0
        self.z_mu_context = 0
        self.z_logvar_context = 0
        self.zs = 0
        self.zdim = z_dim

    def data_to_z_params(self, x, y):
        """Helper to batch together some steps of the process."""
        xy = torch.cat([x, y], dim=1)
        rs = self.repr_encoder(xy)
        r_agg = rs.mean(dim=0)  # Average over samples

        return self.z_encoder(r_agg)  # Get mean and variance for q(z|...)

    def sample_z(self, mu, logvar, n=1):
        """Reparameterisation trick."""
        if n == 1:
            eps = torch.autograd.Variable(logvar.data.new(z_dim).normal_()).to(device)
        else:
            eps = torch.autograd.Variable(logvar.data.new(n, z_dim).normal_()).to(
                device
            )

        # std = torch.exp(0.5 * logvar)
        std = 0.1 + 0.9 * torch.sigmoid(logvar)
        return mu + std * eps

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

    def forward(self, x_t, x_c, y_c, x_ct, y_ct):
        """ """

        self.z_mu_all, self.z_logvar_all = self.data_to_z_params(x_ct, y_ct)
        self.z_mu_context, self.z_logvar_context = self.data_to_z_params(x_c, y_c)
        self.zs = self.sample_z(self.z_mu_all, self.z_logvar_all)
        return self.decoder(x_t, self.zs)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dcrnn = DCRNNModel(2, 1, 4, 4).to(device)
opt = torch.optim.Adam(dcrnn.parameters(), lr=0.001)
z_dim = 16


def random_split_context_target(x, y, n_context):
    """Helper function to split randomly into context and target"""
    ind = np.arange(x.shape[0])
    mask = np.random.choice(ind, size=n_context, replace=False)
    return x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)


def sample_z(mu, logvar, n=1):
    """Reparameterisation trick."""
    if n == 1:
        eps = torch.autograd.Variable(logvar.data.new(z_dim).normal_())
    else:
        eps = torch.autograd.Variable(logvar.data.new(n, z_dim).normal_())

    std = 0.1 + 0.9 * torch.sigmoid(logvar)
    return mu + std * eps


def data_to_z_params(x, y):
    """Helper to batch together some steps of the process."""
    xy = torch.cat([x, y], dim=1)
    rs = dcrnn.repr_encoder(xy)
    r_agg = rs.mean(dim=0)  # Average over samples
    return dcrnn.z_encoder(r_agg)  # Get mean and variance for q(z|...)


def test(x_train, y_train, x_test):
    with torch.no_grad():
        z_mu, z_logvar = data_to_z_params(x_train.to(device), y_train.to(device))

        output_list = []
        for i in range(len(x_test)):
            zsamples = sample_z(z_mu, z_logvar)
            output = dcrnn.decoder(x_test[i:i + 1].to(device), zsamples).cpu()
            output_list.append(output.detach().numpy())

    return np.concatenate(output_list)


def train(n_epochs, x_train, y_train, n_display=1000, patience=5000):  # 2000, 20000
    losses = []
    mae_losses = []
    kld_losses = []

    # means_test = []
    # stds_test = []
    min_loss = 0.0  # for early stopping
    wait = 0
    min_loss = float("inf")

    for t in range(n_epochs):
        opt.zero_grad()
        # Generate data and process
        x_context, y_context, x_target, y_target = random_split_context_target(
            x_train, y_train, int(len(y_train) * 0.05)
        )  # 0.25

        x_c = torch.from_numpy(x_context).float().to(device)
        x_t = torch.from_numpy(x_target).float().to(device)
        y_c = torch.from_numpy(y_context).float().to(device)
        y_t = torch.from_numpy(y_target).float().to(device)

        x_ct = torch.cat([x_c, x_t], dim=0).float().to(device)
        y_ct = torch.cat([y_c, y_t], dim=0).float().to(device)

        y_pred = dcrnn(x_t, x_c, y_c, x_ct, y_ct)

        loss = MAE(y_pred, y_t) + dcrnn.KLD_gaussian()
        mae_loss = MAE(y_pred, y_t)
        kld_loss = dcrnn.KLD_gaussian()

        if t % (n_display / 10) == 0:
            losses.append(loss.item())
            mae_losses.append(mae_loss.item())
            kld_losses.append(kld_loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dcrnn.parameters(), 5)  # 10
        opt.step()
        if t % n_display == 0:
            print(
                "total:", loss.item(), "mae:", mae_loss.item(), "kld:", kld_loss.item()
            )

        # early stopping
        if loss < min_loss:
            wait = 0
            min_loss = loss

        elif loss >= min_loss:
            wait += 1
            if wait == patience:
                print("Early stopping at epoch: %d" % t)
                return (
                    losses,
                    mae_losses,
                    kld_losses,
                    dcrnn.z_mu_all,
                    dcrnn.z_logvar_all,
                )

    return losses, mae_losses, kld_losses, dcrnn.z_mu_all, dcrnn.z_logvar_all
