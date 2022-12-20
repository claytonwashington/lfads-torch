import torch


def compute_recurrent_size(hps):
    recurrent_units_and_weights = [
        (hps.ic_enc_dim, hps.l2_ic_enc_scale),
        (hps.ic_enc_dim, hps.l2_ic_enc_scale),
        (hps.ci_enc_dim, hps.l2_ci_enc_scale),
        (hps.ci_enc_dim, hps.l2_ci_enc_scale),
        (hps.gen_dim, hps.l2_gen_scale),
        (hps.con_dim, hps.l2_con_scale),
    ]
    model_recurrent_size = 0
    for units, weight in recurrent_units_and_weights:
        if weight > 0:
            model_recurrent_size += 3 * units ** 2
    return model_recurrent_size


def compute_l2_penalty(lfads, hps):
    recurrent_kernels_and_weights = [
        (lfads.encoder.ic_enc.weight_hh_l0, hps.l2_ic_enc_scale),
        (lfads.encoder.ic_enc.weight_hh_l0_reverse, hps.l2_ic_enc_scale),
        (lfads.decoder.rnn.cell.gen_cell.weight_hh, hps.l2_gen_scale),
    ]
    if lfads.use_con:
        recurrent_kernels_and_weights.extend(
            [
                (lfads.encoder.ci_enc.weight_hh_l0, hps.l2_ci_enc_scale),
                (lfads.encoder.ci_enc.weight_hh_l0_reverse, hps.l2_ci_enc_scale),
                (lfads.decoder.rnn.cell.con_cell.weight_hh, hps.l2_con_scale),
            ]
        )
    # Add recurrent penalty
    recurrent_penalty = 0.0
    for kernel, weight in recurrent_kernels_and_weights:
        recurrent_penalty += weight * 0.5 * torch.norm(kernel, 2) ** 2
    recurrent_size = compute_recurrent_size(hps)
    recurrent_penalty /= recurrent_size + 1e-8
    # Add recon penalty if applicable
    recon_penalty = 0.0
    for recon in lfads.recon:
        if hasattr(recon, "compute_l2"):
            recon_penalty += recon.compute_l2()
    return recurrent_penalty + recon_penalty
