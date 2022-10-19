import math

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn.functional as F
from torch import nn

# import torch
# import h5py


class FanInLinear(nn.Linear):
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias, 0.0)


# class PCRInitModuleList(nn.ModuleList):
#     def __init__(self, inits_path: str, modules: list[nn.Module]):
#         super().__init__(modules)
#         # Pull pre-computed initialization from the file, assuming correct order
#         with h5py.File(inits_path, "r") as h5file:
#             weights = [v["/" + k + "/matrix"][()] for k, v in h5file.items()]
#             biases = [v["/" + k + "/bias"][()] for k, v in h5file.items()]
#         # Load the state dict for each layer
#         for layer, weight, bias in zip(self, weights, biases):
#             state_dict = {"weight": torch.tensor(weight), "bias": torch.tensor(bias)}
#             layer.load_state_dict(state_dict)


class InvertibleNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, subnet_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = max(input_dim, output_dim)
        self.subnet_dim = subnet_dim
        inn = Ff.SequenceINN(self.hidden_dim)
        for _ in range(num_layers):
            inn.append(
                Fm.AllInOneBlock,
                subnet_constructor=self._subnet_constructor,
                permute_soft=True,
            )
        self.network = inn

    def _subnet_constructor(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.subnet_dim),
            nn.ReLU(),
            nn.Linear(self.subnet_dim, output_dim),
        )

    def forward(self, inputs, reverse=False):
        # Check that input dimensionality is appropriate
        batch_size, n_steps, n_inputs = inputs.shape
        if not reverse:
            assert n_inputs == self.input_dim
            n_outputs = self.output_dim
        else:
            assert n_inputs == self.output_dim
            n_outputs = self.input_dim
        # Pad the inputs if necessary
        inputs = F.pad(inputs, (0, self.hidden_dim - n_inputs))
        # Pass the inputs through the network
        outputs, _ = self.network(inputs.reshape(-1, self.hidden_dim), rev=reverse)
        outputs = outputs.reshape(batch_size, n_steps, self.hidden_dim)
        # Remove padded elements if necessary
        outputs = outputs[..., :n_outputs]
        return outputs
