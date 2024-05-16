import torch
import numpy as np
from torch.utils import data
from torch import nn

from tqdm.auto import trange
from tqdm.contrib import tenumerate
from copy import copy

import matplotlib.pyplot as plt
from scipy import stats



def get_medium_dy(y):
    x = range(len(y))
    
    # simple linear regression
    slope, intercept, _, _, _ = stats.linregress(x, y)
    medium = intercept + len(y) // 2 * slope
    dy = slope * len(y)
    return (medium, dy)

def get_trend(at_slice):
    medium, dy = get_medium_dy(at_slice)
    if medium > 80:
        if dy >= 10:
            return 0
        if dy <= -10:
            return 1
        else:
            return 2
    else:
        if dy >= 5:
            return 0
        if dy <= -5:
            return 1
        else:
            return 2

class Dataset(data.Dataset):
    def __init__(self, X, Y, labels):
        assert X.shape[0] == Y.shape[0]
        assert Y.shape[0] == labels.shape[0]

        # One hot encoded labels: [C, M, N, D, I, S, A, H]
        self.X = X
        self.Y = Y
        self.labels = torch.zeros((len(X), 8)).to(X.device)

        # ['C', 'M', 'N', 'D', 'I', 'S', 'A', 'H']
        self.labels[:, 0][labels[:, 0] == ["C"]] = 1
        self.labels[:, 1][labels[:, 0] == ["M"]] = 1
        self.labels[:, 2][labels[:, 0] == ["N"]] = 1
        self.labels[:, 3][labels[:, 1] == ["D"]] = 1
        self.labels[:, 4][labels[:, 1] == ["I"]] = 1
        self.labels[:, 5][labels[:, 1] == ["S"]] = 1
        self.labels[:, 6][labels[:, 2] == ["A"]] = 1
        self.labels[:, 7][labels[:, 2] == ["H"]] = 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        label = self.labels[index]
        return x, y, label


class SimDataset(data.Dataset):
    def __init__(self, X, labels):
        # One hot encoded labels: [C, M, N, D, I, S, A, H]
        self.X = X[:, :, [0, 2, 3, 4, 5, 6]].to(X.device)
        self.Y = X[:, :, [1]].to(X.device)
        self.seq = X.to(X.device)

        self.labels = torch.zeros((len(X), 9)).to(X.device)

        # ['C', 'M', 'N', 'D', 'I', 'S', 'A', 'H']
        self.labels[:, 0][labels[:, 0] == ["C"]] = 1
        self.labels[:, 1][labels[:, 0] == ["M"]] = 1
        self.labels[:, 2][labels[:, 0] == ["N"]] = 1
        self.labels[:, 3][labels[:, 1] == ["D"]] = 1
        self.labels[:, 4][labels[:, 1] == ["I"]] = 1
        self.labels[:, 5][labels[:, 1] == ["S"]] = 1
        self.labels[:, 6][labels[:, 2] == ["A"]] = 1
        self.labels[:, 7][labels[:, 2] == ["H"]] = 1
        self.labels[:, 8][labels[:, 2] == ["sim"]] = 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        seq = self.seq[index]
        label = self.labels[index]
        return x, y, seq, label


def load_data(cohort, time_len, test=False, num_features=4):
    time_str = f"{time_len}_MIN"
    trends = ["i", "d", "s"]

    if not test:
        t_set = "train"
    else:
        t_set = "test"

    steps = time_len * 6 * 2
    train_set = torch.empty((0, steps, num_features), dtype=torch.float32)
    train_labels = np.empty((0, 2), dtype=np.float32)

    if cohort == "BOTH":
        co_list = ["HRPCI", "AMI_CGS"]
    else:
        co_list = [cohort]

    for co in co_list:
        for trend in trends:
            data = torch.load(
                f"../trends/{time_str}/{co}/{t_set}/trends_slices_{trend}"
            )
            labels = torch.load(
                f"../trends/{time_str}/{co}/{t_set}/trends_labels_{trend}"
            )
            train_set = torch.vstack([train_set, data])
            train_labels = np.vstack([train_labels, labels])

    return train_set, train_labels


def generate_trend_weights(labels):
    sample_weights = np.zeros(len(labels))
    inc_weight = 1 / (sum(labels[:, -1] == "I") / len(labels))
    dec_weight = 1 / (sum(labels[:, -1] == "D") / len(labels))
    stat_weight = 1 / (sum(labels[:, -1] == "S") / len(labels))

    for i, label in enumerate(labels):
        if label[1] == "I":
            sample_weights[i] = inc_weight
        elif label[1] == "D":
            sample_weights[i] = dec_weight
        elif label[1] == "S":
            sample_weights[i] = stat_weight

    return sample_weights


def split_data(
    scaled_train, labels, time_len, split=[4, 1, 1], batch_size=64, shuffle=False
):
    X, Y = partition_data(scaled_train, time_len)
    if type(split) == list:
        split = np.array(split)
    split *= len(X) // sum(split)
    a, b, _ = split
    b += a

    if shuffle:
        np.random.seed(0)
        idx = np.random.permutation(len(X))
        X = X[idx]
        Y = Y[idx]
        labels = labels[idx]

    train_X = X[:a]
    val_X = X[a:b]
    test_X = X[b:]

    train_Y = Y[:a]
    val_Y = Y[a:b]
    test_Y = Y[b:]

    train_label = labels[:a]
    val_label = labels[a:b]
    test_label = labels[b:]

    train_set = Dataset(train_X, train_Y, train_label)
    if len(train_set) == 0:
        train_loader = None
    else:
        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = Dataset(val_X, val_Y, val_label)
    if len(val_set) == 0:
        val_loader = None
    else:
        val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    test_set = Dataset(test_X, test_Y, test_label)
    if len(test_set) == 0:
        test_loader = None
    else:
        test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_mix_data(
    num_hrpci,
    num_amicgs,
    time_len,
    device,
    test=False,
    hrpci_mean=None,
    hrpci_std=None,
    amicgs_mean=None,
    amicgs_std=None,
):
    hrpci_train_set, hrpci_train_labels = load_data("HRPCI", time_len, test)
    amicgs_train_set, amicgs_train_labels = load_data("AMI_CGS", time_len, test)

    num_hrpci = min(num_hrpci, len(hrpci_train_set))
    num_amicgs = min(num_amicgs, len(amicgs_train_set))
    print("Number of HRPCI  samples:", num_hrpci)
    print("Number of AMICGS samples:", num_amicgs)

    #### Loader for HRPCI ####
    print("----- HRPCI ------")
    if hrpci_mean is None or hrpci_std is None:
        hrpci_scaled_train, hrpci_mean, hrpci_std = scale_data(
            hrpci_train_set[:num_hrpci]
        )
    else:
        hrpci_scaled_train, _, _ = scale_data(
            hrpci_train_set[:num_hrpci], hrpci_mean, hrpci_std
        )
    hrpci_scaled_train = hrpci_scaled_train.to(device)

    hrpci_labels = hrpci_train_labels[:num_hrpci]
    hrpci_labels = np.append(hrpci_labels, np.array([["H"] * num_hrpci]).T, axis=1)

    #### Loader for AMICGS ####
    print("---- AMI_CGS -----")
    if amicgs_mean is None or amicgs_std is None:
        amicgs_scaled_train, amicgs_mean, amicgs_std = scale_data(
            amicgs_train_set[:num_amicgs]
        )
    else:
        amicgs_scaled_train, _, _ = scale_data(
            amicgs_train_set[:num_amicgs], amicgs_mean, amicgs_std
        )
    amicgs_scaled_train = amicgs_scaled_train.to(device)

    amicgs_labels = amicgs_train_labels[:num_amicgs]
    amicgs_labels = np.append(amicgs_labels, np.array([["A"] * num_amicgs]).T, axis=1)

    #### Loader for both dataset ####
    scaled_train = torch.vstack([hrpci_scaled_train, amicgs_scaled_train])
    labels = np.vstack([hrpci_labels, amicgs_labels])

    return (
        (scaled_train, labels),
        (hrpci_scaled_train, hrpci_labels),
        (amicgs_scaled_train, amicgs_labels),
        (hrpci_mean, hrpci_std),
        (amicgs_mean, amicgs_std),
    )


def load_mix_data_sim(
    num_hrpci,
    num_amicgs,
    time_len,
    device,
    test=False,
    hrpci_mean=None,
    hrpci_std=None,
    amicgs_mean=None,
    amicgs_std=None,
):
    hrpci_train_set, hrpci_train_labels = load_data(
        "HRPCI", time_len, test, num_features=7
    )
    amicgs_train_set, amicgs_train_labels = load_data(
        "AMI_CGS", time_len, test, num_features=7
    )

    num_hrpci = min(num_hrpci, len(hrpci_train_set))
    num_amicgs = min(num_amicgs, len(amicgs_train_set))
    print("Number of HRPCI  samples:", num_hrpci)
    print("Number of AMICGS samples:", num_amicgs)

    #### Loader for HRPCI ####
    print("----- HRPCI ------")
    if hrpci_mean is None or hrpci_std is None:
        hrpci_scaled_train, hrpci_mean, hrpci_std = scale_data(
            hrpci_train_set[:num_hrpci]
        )
    else:
        hrpci_scaled_train, _, _ = scale_data(
            hrpci_train_set[:num_hrpci], hrpci_mean, hrpci_std
        )
    hrpci_scaled_train = hrpci_scaled_train.to(device)

    hrpci_labels = hrpci_train_labels[:num_hrpci]
    hrpci_labels = np.append(hrpci_labels, np.array([["H"] * num_hrpci]).T, axis=1)

    #### Loader for AMICGS ####
    print("---- AMI_CGS -----")
    if amicgs_mean is None or amicgs_std is None:
        amicgs_scaled_train, amicgs_mean, amicgs_std = scale_data(
            amicgs_train_set[:num_amicgs]
        )
    else:
        amicgs_scaled_train, _, _ = scale_data(
            amicgs_train_set[:num_amicgs], amicgs_mean, amicgs_std
        )
    amicgs_scaled_train = amicgs_scaled_train.to(device)

    amicgs_labels = amicgs_train_labels[:num_amicgs]
    amicgs_labels = np.append(amicgs_labels, np.array([["A"] * num_amicgs]).T, axis=1)

    #### Loader for both dataset ####
    scaled_train = torch.vstack([hrpci_scaled_train, amicgs_scaled_train])
    labels = np.vstack([hrpci_labels, amicgs_labels])

    return (
        (scaled_train, labels),
        (hrpci_scaled_train, hrpci_labels),
        (amicgs_scaled_train, amicgs_labels),
        (hrpci_mean, hrpci_std),
        (amicgs_mean, amicgs_std),
    )


# Returns scaled data and target mean + STD
def scale_data(data, tar_mean=None, tar_std=None):
    if tar_mean is None:
        tar_mean = data.mean(axis=(0, 1))
    print("Mean:", "{:.4f}".format(tar_mean[1].item()))
    if tar_std is None:
        tar_std = data.std(axis=(0, 1))
    print("STD :", "{:.4f}".format(tar_std[1].item()))
    scaled_data = (data - tar_mean) / tar_std
    return scaled_data, tar_mean, tar_std


# Undo the scaled pressure using STD + mean
# ah_mean: [2,], [amicgs_mean, hrpci_mean]
# ah_std : [2,], [amicgs_std , hrpci_std]
def unscale_data(scaled_pressure, label, ah_mean, ah_std):
    return torch.sum(label[:, -2:] * ah_std, -1).unsqueeze(
        -1
    ) * scaled_pressure + torch.sum(label[:, -2:] * ah_mean, -1).unsqueeze(-1)


def partition_data(data, output_time):
    output_steps = output_time * 6
    total_steps = data.shape[1]
    part = total_steps - output_steps

    X = data[:, 0:part]
    Y = data[:, part:, 1]
    return X, Y


def sequence_weights(sample_weights, seq_len):
    return np.tile(sample_weights, (seq_len, 1)).T


def train(
    model,
    train_loader,
    val_loader,
    ah_mean,
    ah_std,
    scale=1.0,
    nepoch=20,
    lr=1e-2,
    use_label=False,
    max_train_iter=300,
    max_val_iter=100,
    amicgs_weight=1.0,
):
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in trange(nepoch):
        train_loss = []
        val_loss = []

        model.train()
        for i, (train_x, train_y, train_label) in tenumerate(train_loader):
            if i == max_train_iter:
                break

            optimizer.zero_grad()

            # Compute loss
            if use_label:
                pred_y = model(train_x, train_label).squeeze(-1)
            else:
                pred_y = model(train_x).squeeze(-1)

            if amicgs_weight != 1.0:
                mask = train_label[:, -2] == 1
                loss = loss_func(pred_y[mask], train_y[mask]) * amicgs_weight
                loss += loss_func(pred_y[~mask], train_y[~mask])
            else:
                loss = loss_func(pred_y, train_y)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            pred_y = unscale_data(pred_y, train_label, ah_mean, ah_std)
            train_y = unscale_data(train_y, train_label, ah_mean, ah_std)

            # Print statistics
            train_loss.append(np.sqrt(loss_func(pred_y, train_y).item()) * scale)

        model.eval()
        for i, (val_x, val_y, val_label) in enumerate(val_loader):
            if i == max_val_iter:
                break

            optimizer.zero_grad()
            if use_label:
                pred_y = model(val_x, val_label).squeeze(-1)
            else:
                pred_y = model(val_x).squeeze(-1)

            pred_y = unscale_data(pred_y, val_label, ah_mean, ah_std)
            val_y = unscale_data(val_y, val_label, ah_mean, ah_std)

            val_loss.append(np.sqrt(loss_func(pred_y, val_y).item()) * scale)

        train_loss = sum(train_loss) / len(train_loss)
        val_loss = sum(val_loss) / len(val_loss)

        if len(val_losses) == 0 or val_loss < min(val_losses):
            best_model = copy(model)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch} \t Train RMSE: {train_loss:6.4f}" + f"\t Val RMSE:   {val_loss:6.4f}"
        )

    return train_losses, val_losses, best_model


def test(model, test_loader, ah_mean, ah_std, nexample=5, use_label=False, trend=None):
    model.eval()
    test_loss = []
    loss_func = nn.MSELoss()
    flag = nexample  # Semaphore
    for _, (test_x, test_y, test_label) in tenumerate(test_loader):
        if use_label:
            pred_y = model(test_x, test_label).squeeze(-1)
            pred_y = unscale_data(pred_y, test_label, ah_mean, ah_std)
        else:
            pred_y = model(test_x).squeeze(-1)
            pred_y = unscale_data(pred_y, test_label, ah_mean, ah_std)

        if trend is None:
            mask = torch.ones(len(pred_y), dtype=torch.bool)  # all trends
        else:
            idx_dict = {
                "D": 3,
                "I": 4,
                "S": 5,
                "d": 3,
                "i": 4,
                "s": 5,
                "C": 0,
                "M": 1,
                "N": 2,
                "c": 0,
                "m": 1,
                "n": 2,
            }
            mask = test_label[:, idx_dict[trend]] == 1
            if sum(mask) == 0:
                continue

        gt = unscale_data(test_y, test_label, ah_mean, ah_std)

        if flag > 0:
            example_x = unscale_data(test_x[..., 1], test_label, ah_mean, ah_std)[0]
            example_predict = pred_y[0].detach()
            example_y = gt[0]
            plt.figure()
            A = len(example_x)
            B = len(example_x) + len(example_y)
            plt.plot(range(A), example_x.cpu().numpy(), "r")
            plt.plot(range(A, B), example_y.cpu().numpy(), "r", label="gt")
            plt.plot(range(A, B), example_predict.cpu().numpy(), "b", label="predict")
            plt.legend()
            print(np.sqrt(loss_func(example_predict, example_y).item()))
            flag -= 1

        loss = loss_func(pred_y[mask], gt[mask])
        test_loss.append(np.sqrt(loss.item()))

    return sum(test_loss) / len(test_loss)
