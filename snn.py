import h5py
import torch
from lava.lib.dl import slayer


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

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.Dense(
                    neuron_params, dataset_cls.num_channels * dataset_cls.width * dataset_cls.height, 128,
                    delay=False
                ),
                slayer.block.cuba.Dense(
                    neuron_params, 128, 128,
                    delay=False
                ),
                slayer.block.cuba.Dense(
                    neuron_params, 128, 64,
                    delay=False
                ),
                slayer.block.cuba.Dense(
                    neuron_params, 64, dataset_cls.num_classes,
                    delay=False
                )
            ]
        )

    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike

    def export_hdf5(self, filename):
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))