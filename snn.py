import h5py
import torch
from lava.lib.dl import slayer
from lava.lib.dl.slayer.utils import quantize_hook_fx


class SNN(torch.nn.Module):
    def __init__(self, dataset_cls: type["BaseDataset"]):
        super().__init__()

        neuron_params = {
            'threshold': 1.25,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 3,
            'requires_grad': True,
        }

        self.scale = 64
        self.num_bits = 6

        self.inputs = dataset_cls.num_channels * dataset_cls.width * dataset_cls.height

        self.neurons_1 = 64
        self.neurons_2 = 128
        self.neurons_3 = 64
        self.neurons_4 = dataset_cls.num_classes

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.Dense(
                    neuron_params, self.inputs, self.neurons_1,
                    pre_hook_fx=lambda x: quantize_hook_fx(x, self.scale, self.num_bits),
                    delay=False
                ),
                slayer.block.cuba.Dense(
                    neuron_params, self.neurons_1, self.neurons_2,
                    pre_hook_fx=lambda x: quantize_hook_fx(x, self.scale, self.num_bits),
                    delay=False
                ),
                slayer.block.cuba.Dense(
                    neuron_params, self.neurons_2, self.neurons_3,
                    pre_hook_fx=lambda x: quantize_hook_fx(x, self.scale, self.num_bits),
                    delay=False
                ),
                slayer.block.cuba.Dense(
                    neuron_params, self.neurons_3, self.neurons_4,
                    pre_hook_fx=lambda x: quantize_hook_fx(x, self.scale, self.num_bits),
                    delay=False
                )
            ]
        )

        for block in self.blocks:
            with torch.no_grad():
                block.synapse.weight.data = block.synapse.weight.data.abs()

    def forward(self, spike):
        for block in self.blocks:
            print("WEIGHTS:", torch.sum(block.synapse.weight[:, :, 0, 0, 0]))
            spike = block(spike)
        return spike

    def export_hdf5(self, filename):
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))