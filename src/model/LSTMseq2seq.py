import torch
from torch import nn


class EncoderLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, device):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        ).to(device)

        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.device = device
        self.to(device)

    # x : [batch_size, seq_len, input_size]
    def forward(self, x, state=None):
        input = self.fc(x)
        output = []

        # Assuming batch dimension is always first, followed by seq. length as the second dimension
        batch_size = x.size(0)
        # seq_len = x.size(1)

        # Initial state (h_0, c_0)
        if state is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
                self.device
            )
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
                self.device
            )
            state = (h0, c0)

        # Iterate over the timesteps
        output, state = self.rnn(input, state)

        return output


class DecoderLSTM(nn.Module):
    def __init__(self, num_layers, seq_len, output_size, hidden_size, device):
        super(DecoderLSTM, self).__init__()
        self.seq_len = seq_len

        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ).to(device)

        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.device = device
        self.to(device)

    # x is generated by EncoderLSTM, [batch_size, hidden_size]
    def forward(self, x, state=None):
        output = []

        # Assuming batch dimension is always first
        batch_size = x.size(0)

        # Initial state (h_0, m_0)
        if state is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
                self.device
            )
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
                self.device
            )
            state = (h0, c0)

        # Iterate over the timesteps
        output = []
        if len(x.shape) == 2:
            input = x.unsqueeze(1)
        else:
            input = x

        for _ in range(self.seq_len):
            _, (h_t, c_t) = self.rnn(input, state)
            state = (h_t, c_t)
            h_t = torch.sum(h_t, 0)
            output.append(h_t)
            input = h_t.unsqueeze(1)

        output = torch.stack(output, 1)  # [seq_len, batch_size, hidden_size]
        output = self.fc(output)

        return output


class DecoderLSTMAttn(nn.Module):
    def __init__(self, num_layers, seq_len, output_size, hidden_size, device):
        super(DecoderLSTMAttn, self).__init__()
        self.seq_len = seq_len

        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ).to(device)

        # bilinear attn function
        self.bil = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.device = device
        self.to(device)

    # x is generated by EncoderLSTM, [batch_size, inp_len, hidden_size]
    def forward(self, x, state=None):
        output = []

        # Assuming batch dimension is always first
        batch_size = x.size(0)

        # Initial state (h_0, m_0)
        if state is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
                self.device
            )
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
                self.device
            )
            state = (h0, c0)
        else:
            (h0, c0) = state

        # Iterate over the timesteps
        output = []

        score = torch.softmax(
            x.matmul(self.bil).bmm(torch.sum(h0, 0).unsqueeze(-1)), dim=1
        )
        input = torch.sum(x * score, 1)

        for _ in range(self.seq_len):
            _, (h_t, c_t) = self.rnn(input.unsqueeze(1), state)
            state = (h_t, c_t)
            score = torch.softmax(
                x.matmul(self.bil).bmm(torch.sum(h_t, 0).unsqueeze(-1)), dim=1
            )

            input = torch.sum(x * score, 1)
            output.append(torch.sum(h_t, 0))

        output = torch.stack(output, 1)  # [seq_len, batch_size, hidden_size]
        output = self.fc(output)

        return output


class LSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        num_layers,
        seq_len,
        input_size,
        output_size,
        hidden_size,
        device,
        attn=False,
    ):
        super(LSTMSeq2Seq, self).__init__()
        self.seq_len = seq_len

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.device = device

        self.enc = EncoderLSTM(num_layers, input_size, hidden_size, device)
        self.attn = attn
        if attn:
            self.dec = DecoderLSTMAttn(
                num_layers, seq_len, output_size, hidden_size, device
            )
        else:
            self.dec = DecoderLSTM(
                num_layers, seq_len, output_size, hidden_size, device
            )

    # x : [batch_size, seq_len, input_size]
    def forward(self, x):
        s = self.enc(x.to(self.device))
        if self.attn:
            output = self.dec(s)  # [batch_size, output_seq_len, output_size]
        else:
            output = self.dec(s[:, -1, :])  # Only take the last step output
        if output.shape[-1] == 1:
            output = output.squeeze(-1)
        return output


class CondLSTMSeq2Seq(nn.Module):
    """
    cond: shape (2, 4)
    feature std, first row ami-cgs, second row hrpci
    """

    def __init__(
        self,
        num_layers,
        seq_len,
        cond,
        input_size,
        output_size,
        hidden_size,
        device,
        attn=False,
        perturb="uniform",
    ):
        super(CondLSTMSeq2Seq, self).__init__()
        self.seq_len = seq_len

        self.cond = cond.to(device)
        self.cond_size = cond.shape[-1]

        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.device = device
        self.perturb = perturb

        self.enc = EncoderLSTM(num_layers, input_size, hidden_size, device)
        self.attn = attn
        if attn:
            self.dec = DecoderLSTMAttn(
                num_layers, seq_len, output_size, hidden_size, device
            )
        else:
            self.dec = DecoderLSTM(
                num_layers, seq_len, output_size, hidden_size, device
            )

        self.cen = nn.Sequential(  # Condition encoder
            nn.Linear(self.cond_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(self.hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(hidden_size, 2 * hidden_size),
        ).to(device)

    # Use each sequence's feature average as a time-invariant condition
    # x : [batch_size, seq_len, input_size]
    def forward(self, x, labels):
        labels = labels[:, -2:]  # last 2 column: A or H

        scale = torch.std(self.cond, 0)
        cond = labels.mm(self.cond)
        if self.training:
            if self.perturb == "uniform":
                cond += (
                    (torch.rand(len(x), self.cond.shape[-1]).to(self.device) - 1) * 0.3 * scale
                )
            elif self.perturb == "normal":
                cond += torch.normal(torch.zeros_like(scale), 0.1 * scale).to(
                    self.device
                )

        state = torch.split(
            self.cen(cond), [self.hidden_size, self.hidden_size], dim=-1
        )
        state = [s.unsqueeze(0).repeat(self.num_layers, 1, 1) for s in state]
        s = self.enc(x.to(self.device), state)
        if self.attn:
            output = self.dec(s, state)  # [batch_size, output_seq_len, output_size]
        else:
            output = self.dec(s[:, -1, :], state)  # Only take the last step output
        if output.shape[-1] == 1:
            output = output.squeeze(-1)
        return output