# Must install `torchcde` and `signatory`
# pip install torch==1.9
# pip install signatory==1.2.6.1.9.0 --no-cache-dir --force-reinstall
# pip install setuptools==59.5.0

from collections import namedtuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchcde
from torch import nn

from ..metrics import ExpSmoothedMetric, r2_score
from ..tuples import SessionBatch
from ..utils import transpose_lists
from .modules import augmentations
from .modules.initializers import init_linear_
from .modules.recurrent import BidirectionalClippedGRU

CtsSessionOutput = namedtuple(
    "CtsSessionOutput",
    [
        "output_params",
        "factors",
        "init_samp",
        "init_mean",
        "init_std",
        "input_samps",
        "input_means",
        "input_stds",
    ],
)


class Encoder(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hps = hparams
        # Set initial hidden state for all encoders
        self.init_enc_h0 = nn.Parameter(
            torch.zeros((2, 1, hps.init_enc_dim), requires_grad=True)
        )
        self.input_enc_h0 = nn.Parameter(
            torch.zeros((2, 1, hps.input_enc_dim), requires_grad=True)
        )
        # Create initial condition and input encoders
        self.init_enc = BidirectionalClippedGRU(
            input_size=hps.encod_data_dim,
            hidden_size=hps.init_enc_dim,
            clip_value=hps.cell_clip,
        )
        self.input_enc = BidirectionalClippedGRU(
            input_size=hps.encod_data_dim,
            hidden_size=hps.input_enc_dim,
            clip_value=hps.cell_clip,
        )
        # Create and re-initialize linear mappings to posterior distributions
        self.init_linear = nn.Linear(hps.init_enc_dim * 2, hps.latent_dim * 2)
        self.input_linear = nn.Linear(hps.input_enc_dim * 2, hps.input_dim * 2)
        init_linear_(self.init_linear)
        init_linear_(self.input_linear)
        # nn.init.zeros_(self.input_linear.weight)
        # Create dropout layer
        self.dropout = nn.Dropout(hps.dropout_rate)

    def forward(self, data: torch.Tensor):
        hps = self.hparams
        assert data.shape[1] == hps.encod_seq_len, (
            f"Sequence length specified in HPs ({hps.encod_seq_len}) "
            f"must match data dim 1 ({data.shape[1]})."
        )
        data_drop = self.dropout(data)
        # Pass the data through the encoder RNNs
        _, init_enc = self.init_enc(data_drop, self.init_enc_h0)
        input_enc, _ = self.input_enc(data_drop, self.input_enc_h0)
        # Add extra zeros if necessary for forward prediction
        fwd_steps = hps.recon_seq_len - hps.encod_seq_len
        if fwd_steps > 0:
            input_enc = F.pad(input_enc, (0, 0, 0, fwd_steps, 0, 0))
        # Compute the initial condition posterior
        init_enc_drop = self.dropout(init_enc)
        init_params = self.init_linear(init_enc_drop)
        init_mean, init_logvar = torch.split(init_params, hps.latent_dim, dim=1)
        init_std = torch.sqrt(torch.exp(init_logvar) + hps.init_post_var_min)
        # Compute the input posterior
        input_enc_drop = self.dropout(input_enc)
        input_params = self.input_linear(input_enc_drop)
        input_mean, input_logvar = torch.split(input_params, hps.input_dim, dim=2)
        input_std = torch.sqrt(torch.exp(input_logvar))

        return init_mean, init_std, input_mean, input_std


class NeuralCDE(nn.Module):
    def __init__(self, input_size, state_size, hidden_size, depth):
        super().__init__()
        self.input_size, self.state_size, = (
            input_size,
            state_size,
        )
        input_layer = nn.Linear(state_size, hidden_size)
        inner_layers = [nn.Linear(hidden_size, hidden_size) for _ in range(depth - 1)]
        self.early_layers = nn.ModuleList([input_layer] + inner_layers)
        self.output_layer = nn.Linear(hidden_size, (input_size + 1) * state_size)

    def forward(self, t, z):
        for layer in self.early_layers:
            z = layer(z).relu()
        output = self.output_layer(z).tanh()
        output = output.view(-1, self.state_size, self.input_size + 1)
        return output


class Decoder(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        self.neural_cde = hparams.neural_cde

    def forward(self, init, inputs):
        batch_size, n_steps, _ = inputs.shape
        t = torch.arange(n_steps, dtype=inputs.dtype, device=inputs.device)
        # t = torch.linspace(0, 1, n_steps).to(inputs.device)
        t_batch = t[None, :, None].repeat(batch_size, 1, 1)
        inputs = torch.cat([t_batch, inputs], axis=2)
        logsigs = torchcde.logsig_windows(inputs, depth=2, window_length=4, t=t)
        t_logsig = logsigs[0, :, 0].contiguous()
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            logsigs, t_logsig
        )
        path = torchcde.CubicSpline(coeffs, t_logsig)
        adjoint_params = tuple(self.neural_cde.parameters()) + (init, coeffs)
        factors = torchcde.cdeint(
            X=path,
            z0=init,
            func=self.neural_cde,
            t=t,
            adjoint_params=adjoint_params,
        )
        return factors.tanh()


class CTLFADS(pl.LightningModule):
    def __init__(
        self,
        encod_data_dim: int,
        encod_seq_len: int,
        recon_seq_len: int,
        init_enc_dim: int,
        input_enc_dim: int,
        input_dim: int,
        latent_dim: int,
        neural_cde: nn.Module,
        dropout_rate: float,
        reconstruction: nn.ModuleList,
        input_prior: nn.Module,
        init_prior: nn.Module,
        init_post_var_min: float,
        cell_clip: float,
        train_aug_stack: augmentations.AugmentationStack,
        infer_aug_stack: augmentations.AugmentationStack,
        readin: nn.ModuleList,
        readout: nn.ModuleList,
        loss_scale: float,
        recon_reduce_mean: bool,
        lr_scheduler: bool,
        lr_init: float,
        lr_stop: float,
        lr_decay: float,
        lr_patience: int,
        lr_adam_epsilon: float,
        weight_decay: float,
        # l2_start_epoch: int,
        # l2_increase_epoch: int,
        # l2_ic_enc_scale: float,
        # l2_input_enc_scale: float,
        kl_start_epoch: int,
        kl_increase_epoch: int,
        kl_init_scale: float,
        kl_input_scale: float,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "neural_cde",
                "init_prior",
                "input_prior",
                "reconstruction",
                "readin",
                "readout",
            ],
        )
        # Store `neural_cde` on `hparams` so it can be accessed in decoder
        self.hparams.neural_cde = neural_cde
        # Make sure the nn.ModuleList arguments are all the same length
        assert len(readin) == len(readout) == len(reconstruction)

        # Store the readin network
        self.readin = readin
        # Create the encoder and decoder
        self.encoder = Encoder(hparams=self.hparams)
        self.decoder = Decoder(hparams=self.hparams)
        # Store the readout network
        self.readout = readout
        # Create object to manage reconstruction
        self.recon = reconstruction
        # Store the trainable priors
        self.init_prior = init_prior
        self.input_prior = input_prior
        # Create metric for exponentially-smoothed `valid/recon`
        self.valid_recon_smth = ExpSmoothedMetric()
        # Store the data augmentation stacks
        self.train_aug_stack = train_aug_stack
        self.infer_aug_stack = infer_aug_stack

    def forward(
        self,
        batch: dict[SessionBatch],
        sample_posteriors: bool = False,
        output_means: bool = True,
    ) -> dict[CtsSessionOutput]:
        # Allow SessionBatch input
        if type(batch) == SessionBatch and len(self.readin) == 1:
            batch = {0: batch}
        # Determine which sessions are in the batch
        sessions = sorted(batch.keys())
        # Keep track of batch sizes so we can split back up
        batch_sizes = [len(batch[s].encod_data) for s in sessions]
        # Pass the data through the readin networks
        encod_data = torch.cat([self.readin[s](batch[s].encod_data) for s in sessions])
        # Pass the data through the encoders
        init_mean, init_std, input_means, input_stds = self.encoder(encod_data)
        # Create the posterior distributions
        init_post = self.init_prior.make_posterior(init_mean, init_std)
        input_post = self.input_prior.make_posterior(input_means, input_stds)
        # Choose to take a sample or to pass the mean
        sample_posteriors = False
        init_samp = init_post.rsample() if sample_posteriors else init_mean
        input_samps = input_post.rsample() if sample_posteriors else input_means
        # Unroll the decoder to estimate latent states
        factors = self.decoder(init_samp, input_samps)
        # Convert the factors representation into output distribution parameters
        factors = torch.split(factors, batch_sizes)
        output_params = [self.readout[s](f) for s, f in zip(sessions, factors)]
        # Separate parameters of the output distribution
        output_params = [
            self.recon[s].reshape_output_params(op)
            for s, op in zip(sessions, output_params)
        ]
        # Convert the output parameters to means if requested
        if output_means:
            output_params = [
                self.recon[s].compute_means(op)
                for s, op in zip(sessions, output_params)
            ]
        # Separate model outputs by session
        output = transpose_lists(
            [
                output_params,
                factors,
                torch.split(init_samp, batch_sizes),
                torch.split(init_mean, batch_sizes),
                torch.split(init_std, batch_sizes),
                torch.split(input_samps, batch_sizes),
                torch.split(input_means, batch_sizes),
                torch.split(input_stds, batch_sizes),
            ]
        )
        # Return the parameter estimates and all intermediate activations
        return {s: CtsSessionOutput(*o) for s, o in zip(sessions, output)}

    def configure_optimizers(self):
        hps = self.hparams
        # Create an optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=hps.lr_init,
            eps=hps.lr_adam_epsilon,
            weight_decay=hps.weight_decay,
        )
        if hps.lr_scheduler:
            # Create a scheduler to reduce the learning rate over time
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=hps.lr_decay,
                patience=hps.lr_patience,
                threshold=0.0,
                min_lr=hps.lr_stop,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "valid/recon_smth",
            }
        else:
            return optimizer

    def _shared_step(self, batch, batch_idx, split):
        hps = self.hparams
        # Check that the split argument is valid
        assert split in ["train", "valid"]
        # Determine which sessions are in the batch
        sessions = sorted(batch.keys())
        # Discard the extra data - only the SessionBatches are relevant here
        batch = {s: b[0] for s, b in batch.items()}
        # Process the batch for each session (in order so aug stack can keep track)
        aug_stack = self.train_aug_stack if split == "train" else self.infer_aug_stack
        batch = {s: aug_stack.process_batch(batch[s]) for s in sessions}
        # Perform the forward pass
        output = self.forward(batch, sample_posteriors=True, output_means=False)
        # Compute the reconstruction loss
        recon_all = [
            self.recon[s].compute_loss(batch[s].recon_data, output[s].output_params)
            for s in sessions
        ]
        # Apply losses processing
        recon_all = [
            aug_stack.process_losses(ra, batch[s], self.log, split)
            for ra, s in zip(recon_all, sessions)
        ]
        # Aggregate the heldout cost for logging
        if not hps.recon_reduce_mean:
            recon_all = [torch.sum(ra, dim=(1, 2)) for ra in recon_all]
        # Compute reconstruction loss for each session
        sess_recon = [ra.mean() for ra in recon_all]
        recon = torch.mean(torch.stack(sess_recon))
        # Compute the L2 penalty on recurrent weights
        # l2 = compute_l2_penalty(self, self.hparams)
        # l2_ramp = (self.current_epoch - hps.l2_start_epoch) / (
        #     hps.l2_increase_epoch + 1
        # )
        # Collect posterior parameters for fast KL calculation
        init_mean = torch.cat([output[s].init_mean for s in sessions])
        init_std = torch.cat([output[s].init_std for s in sessions])
        input_means = torch.cat([output[s].input_means for s in sessions])
        input_stds = torch.cat([output[s].input_stds for s in sessions])
        # Compute the KL penalty on posteriors
        init_kl = self.init_prior(init_mean, init_std) * self.hparams.kl_init_scale
        input_kl = (
            self.input_prior(input_means, input_stds) * self.hparams.kl_input_scale
        )
        kl_ramp = (self.current_epoch - hps.kl_start_epoch) / (
            hps.kl_increase_epoch + 1
        )
        # Clamp the ramps
        # l2_ramp = torch.clamp(torch.tensor(l2_ramp), 0, 1)
        kl_ramp = torch.clamp(torch.tensor(kl_ramp), 0, 1)
        # Compute the final loss
        loss = hps.loss_scale * (recon + kl_ramp * (init_kl + input_kl))
        # Compute the reconstruction accuracy, if applicable
        output_means = [
            self.recon[s].compute_means(output[s].output_params) for s in sessions
        ]
        r2 = torch.mean(
            torch.stack(
                [r2_score(om, batch[s].truth) for om, s in zip(output_means, sessions)]
            )
        )
        # Compute batch sizes for logging
        batch_sizes = [len(batch[s].encod_data) for s in sessions]
        # Log per-session metrics
        for s, recon_value, batch_size in zip(sessions, sess_recon, batch_sizes):
            self.log(
                name=f"{split}/recon/sess{s}",
                value=recon_value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        # Collect metrics for logging
        metrics = {
            f"{split}/loss": loss,
            f"{split}/recon": recon,
            f"{split}/r2": r2,
            # f"{split}/wt_l2": l2,
            # f"{split}/wt_l2/ramp": l2_ramp,
            f"{split}/wt_kl": init_kl + input_kl,
            f"{split}/wt_kl/init": init_kl,
            f"{split}/wt_kl/input": input_kl,
            f"{split}/wt_kl/ramp": kl_ramp,
        }
        if split == "valid":
            # Update the smoothed reconstruction loss
            self.valid_recon_smth.update(recon)
            # Add validation-only metrics
            metrics.update(
                {
                    "valid/recon_smth": self.valid_recon_smth,
                    "hp_metric": recon,
                    "cur_epoch": float(self.current_epoch),
                }
            )
        # Log overall metrics
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=sum(batch_sizes),
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "valid")

    def predict_step(self, batch, batch_ix, sample_posteriors=True):
        # Discard the extra data - only the SessionBatches are relevant here
        batch = {s: b[0] for s, b in batch.items()}
        # Process the batch for each session
        batch = {s: self.infer_aug_stack.process_batch(b) for s, b in batch.items()}
        # Reset to clear any saved masks
        self.infer_aug_stack.reset()
        # Perform the forward pass
        return self.forward(
            batch=batch,
            sample_posteriors=sample_posteriors,
            output_means=True,
        )

    def on_validation_epoch_end(self):
        # Log hyperparameters that may change during PBT
        self.log_dict(
            {
                "hp/lr_init": self.hparams.lr_init,
                "hp/dropout_rate": self.hparams.dropout_rate,
                # "hp/l2_gen_scale": self.hparams.l2_gen_scale,
                # "hp/l2_con_scale": self.hparams.l2_con_scale,
                "hp/kl_input_scale": self.hparams.kl_input_scale,
                "hp/kl_init_scale": self.hparams.kl_init_scale,
            }
        )
        # Log CD rate if CD is being used
        for aug in self.train_aug_stack.batch_transforms:
            if hasattr(aug, "cd_rate"):
                self.log("hp/cd_rate", aug.cd_rate)
