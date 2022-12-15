import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hps = hparams

        # Initial hidden state for IC encoder
        self.ic_enc_h0 = nn.Parameter(
            torch.zeros((2, 1, hps.ic_enc_dim), requires_grad=True)
        )
        # Initial condition encoder
        self.ic_enc = nn.GRU(
            input_size=hps.encod_data_dim,
            hidden_size=hps.ic_enc_dim,
            bidirectional=True,
            batch_first=True,
        )
        # Mapping from final IC encoder state to IC parameters
        self.ic_linear = nn.Linear(hps.ic_enc_dim * 2, hps.ic_dim * 2)
        # Decide whether to use the controller
        self.use_con = all(
            [
                hps.ci_enc_dim > 0,
                hps.con_dim > 0,
                hps.co_dim > 0,
            ]
        )
        if self.use_con:
            # Initial hidden state for CI encoder
            self.ci_enc_h0 = nn.Parameter(
                torch.zeros((2, 1, hps.ci_enc_dim), requires_grad=True)
            )
            # CI encoder
            self.ci_enc = nn.GRU(
                input_size=hps.encod_data_dim,
                hidden_size=hps.ci_enc_dim,
                bidirectional=True,
                batch_first=True,
            )
            # Create learnable forward encoding
            fwd_steps = hps.recon_seq_len - hps.encod_seq_len
            if fwd_steps > 0:
                self.fwd_enc = nn.Parameter(
                    torch.zeros((1, fwd_steps, 2 * hps.ci_enc_dim), requires_grad=True)
                )
        # Activation dropout layer
        self.dropout = nn.Dropout(hps.dropout_rate)

    def forward(self, data: torch.Tensor):
        hps = self.hparams
        batch_size = data.shape[0]
        assert data.shape[1] == hps.encod_seq_len, (
            f"Sequence length specified in HPs ({hps.encod_seq_len}) "
            f"must match data dim 1 ({data.shape[1]})."
        )
        # option to use separate segment for IC encoding
        if hps.ic_enc_seq_len > 0:
            ic_enc_data = data[:, : hps.ic_enc_seq_len, :]
            ci_enc_data = data[:, hps.ic_enc_seq_len :, :]
        else:
            ic_enc_data = data
            ci_enc_data = data
        # Pass data through IC encoder
        ic_enc_h0 = torch.tile(self.ic_enc_h0, (1, batch_size, 1))
        _, h_n = self.ic_enc(ic_enc_data, ic_enc_h0)
        h_n = torch.cat([*h_n], dim=1)
        # Compute initial condition posterior
        h_n_drop = self.dropout(h_n)
        ic_params = self.ic_linear(h_n_drop)
        ic_mean, ic_logvar = torch.split(ic_params, hps.ic_dim, dim=1)
        ic_std = torch.sqrt(torch.exp(ic_logvar) + hps.ic_post_var_min)
        if self.use_con:
            # Pass data through CI encoder
            ci_enc_h0 = torch.tile(self.ci_enc_h0, (1, batch_size, 1))
            ci, _ = self.ci_enc(ci_enc_data, ci_enc_h0)
            # Add a lag to the controller input
            ci_fwd, ci_bwd = torch.split(ci, hps.ci_enc_dim, dim=2)
            ci_fwd = F.pad(ci_fwd, (0, 0, hps.ci_lag, 0, 0, 0))
            ci_bwd = F.pad(ci_bwd, (0, 0, 0, hps.ci_lag, 0, 0))
            ci_len = hps.encod_seq_len - hps.ic_enc_seq_len
            ci = torch.cat([ci_fwd[:, :ci_len, :], ci_bwd[:, -ci_len:, :]], dim=2)
            # Concatenate learnable forward encoding
            fwd_steps = hps.recon_seq_len - hps.encod_seq_len
            if fwd_steps > 0:
                fwd_enc = torch.tile(self.fwd_enc, (batch_size, 1, 1))
                ci = torch.cat([ci, fwd_enc], dim=1)
        else:
            # Create a placeholder if there's no controller
            ci = torch.zeros(data.shape[0], hps.recon_seq_len, 0).to(data.device)

        return ic_mean, ic_std, ci
