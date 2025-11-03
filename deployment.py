import numpy as np
import torch
from lava.lib.dl.slayer.utils import quantize_hook_fx

from utils.deployment import is_power_of_two, next_power_of_two, hex_conv, concat_layer_expr


class Deployment():
    def __init__(self, net: "SNN"):
        self.net = net
        self.quantizer = lambda x: quantize_hook_fx(x, net.scale, net.num_bits, descale=True)

        self.weights = [self.net.blocks[i].synapse.weight[:, :, 0, 0, 0] for i in range(len(self.net.blocks))]

        self.current_decays = torch.tensor([block.neuron.current_decay.detach().cpu().numpy()[0] for block in self.net.blocks])
        self.voltage_decays = torch.tensor([block.neuron.voltage_decay.detach().cpu().numpy()[0] for block in self.net.blocks])
        self.thresholds = torch.tensor([block.neuron.threshold for block in self.net.blocks])

    def deploy(self):
        self._create_weights_files()
        self._create_config_file()

    def _create_weights_files(self):
        quantized_weights = [self.quantizer(weight).to(torch.int32) for weight in self.weights]
        neurons = [self.net.neurons_1, self.net.neurons_2, self.net.neurons_3, self.net.neurons_4]
        synapses = [self.net.inputs, self.net.neurons_1, self.net.neurons_2, self.net.neurons_3]
        weights = [["0000" for _ in range(max(neurons)//2 * max(synapses)//4 * len(neurons))] for _ in range(len(neurons))]

        MAX_LAYER = max(
            synapses[0] * neurons[0],
            synapses[1] * neurons[1],
            synapses[2] * neurons[2],
            synapses[3] * neurons[3],
        ) // 8

        for layer in range(len(neurons)):
            weights_int = quantized_weights[layer]
            for neuron in range(weights_int.shape[0]):
                for synapse in range(0, weights_int.shape[1], 4):
                    w1 = hex_conv(weights_int[neuron, synapse], 8)
                    w2 = hex_conv(weights_int[neuron, synapse + 1], 8)
                    w3 = hex_conv(weights_int[neuron, synapse + 2], 8)
                    w4 = hex_conv(weights_int[neuron, synapse + 3], 8)

                    w_a_hex = w1 + w2
                    w_b_hex = w3 + w4

                    mem_wr_addr = concat_layer_expr(layer, neuron // 2, synapse // 4, synapses[layer], neurons[layer], MAX_LAYER)

                    if neuron % 2:
                        weights[2][mem_wr_addr] = w_a_hex
                        weights[3][mem_wr_addr] = w_b_hex
                    else:
                        weights[0][mem_wr_addr] = w_a_hex
                        weights[1][mem_wr_addr] = w_b_hex

        files = ['weights_1.coe', 'weights_2.coe', 'weights_3.coe', 'weights_4.coe']

        for i, fname in enumerate(files):
            coe_data = (
                    "memory_initialization_radix=16;\n"
                    "memory_initialization_vector=\n" +
                    ",".join(weights[i]) + ";\n"
            )
            with open(fname, "w") as f:
                f.write(coe_data)
            print(f"[OK] zapisano {fname} ({len(weights[i])} pozycji)")


    def _create_config_file(self):
        with open('config.txt', 'wt') as f:
            f.write(f'`define INPUT_SPIKE_1 {self.net.inputs}\n')
            f.write(f'`define NEURON_1 {self.net.neurons_1}\n')
            f.write(f'`define NEURON_2 {self.net.neurons_2}\n')
            f.write(f'`define NEURON_3 {self.net.neurons_3}\n')
            if (is_power_of_two(self.net.neurons_4)):
                f.write(f'`define NEURON_4 {self.net.neurons_4}\n')
            else:
                f.write(f'`define NEURON_4 {next_power_of_two(self.net.neurons_4)}\n')
            i = 1
            for current_decay in self.current_decays:
                f.write(f'`define CURRENT_DECAY_{i} 4096-' + str(int(current_decay.round())) + '\n')
                i = i + 1
            f.write('\n')
            i = 1
            for voltage_decay in self.voltage_decays:
                f.write(f'`define VOLTAGE_DECAY_{i} 4096-' + str(int(voltage_decay.round())) + '\n')
                i = i + 1
            f.write('\n')
            i = 1
            for threshold in self.thresholds:
                f.write(f'`define THRESHOLD_{i} ' + str(int((threshold * self.net.scale).round())) + '\n')
                i = i + 1
