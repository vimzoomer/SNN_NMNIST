import numpy as np
import torch
from lava.lib.dl.slayer.utils import quantize_hook_fx

from utils.deploy import is_power_of_two, next_power_of_two, hex_conv


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
        files = ['weights_1.coe', 'weights_2.coe', 'weights_3.coe', 'weights_4.coe']

        with open(files[0], 'w') as fa, open(files[1], 'w') as fb, open(files[2], 'w') as fc, open(files[3], 'w') as fd:
            for f in [fa, fb, fc, fd]:
                f.write('memory_initialization_radix = 16;\n')
                f.write('memory_initialization_vector =\n')

            for k in range(len(self.weights)):
                weights_int = quantized_weights[k]

                for j in range(weights_int.shape[0]):
                    weights_a = []
                    weights_b = []

                    for i in range(0, weights_int.shape[1], 4):
                        w1 = hex_conv(weights_int[j, i], 8)
                        w2 = hex_conv(weights_int[j, i + 1], 8)
                        w3 = hex_conv(weights_int[j, i + 2], 8)
                        w4 = hex_conv(weights_int[j, i + 3], 8)

                        w_a_hex = w2 + w1
                        w_b_hex = w4 + w3

                        weights_a.append(w_a_hex)
                        weights_b.append(w_b_hex)

                    line_a = ','.join(weights_a)
                    line_b = ','.join(weights_b)

                    if is_power_of_two(weights_int.shape[1]):
                        padding = 0
                    else:
                        padding = int((next_power_of_two(weights_int.shape[1]) - weights_int.shape[1]) / 4)

                    pad_word = '0000'

                    if j % 2:
                        fc.write(line_a + (',' + pad_word) * padding + ',\n')
                        fd.write(line_b + (',' + pad_word) * padding + ',\n')
                    else:
                        fa.write(line_a + (',' + pad_word) * padding + ',\n')
                        fb.write(line_b + (',' + pad_word) * padding + ',\n')

                # Padding dodatkowych wierszy
                if is_power_of_two(weights_int.shape[0]):
                    padding2 = 0
                else:
                    padding2 = (next_power_of_two(weights_int.shape[0]) - weights_int.shape[0])

                int_padd = np.floor_divide(padding2, 2)
                remainder = np.remainder(padding2, 2)

                if is_power_of_two(weights_int.shape[1]):
                    synapses = int(weights_int.shape[1] / 4)
                else:
                    synapses = int(next_power_of_two(weights_int.shape[1]) / 4)

                pad_line = (pad_word + ',') * synapses + '\n'

                for _ in range(int_padd):
                    fa.write(pad_line)
                    fb.write(pad_line)
                for _ in range(int_padd + remainder):
                    fc.write(pad_line)
                    fd.write(pad_line)

                # Dodatkowy padding na końcu bloków
                nsm = max(
                    self.weights[0].shape[0] * self.weights[0].shape[1],
                    self.weights[1].shape[0] * self.weights[1].shape[1],
                    self.weights[2].shape[0] * self.weights[2].shape[1],
                    self.weights[3].shape[0] * self.weights[3].shape[1],
                ) / 4

                n, s = weights_int.shape
                if not is_power_of_two(n):
                    n = next_power_of_two(n)
                n /= 2
                if not is_power_of_two(s):
                    s = next_power_of_two(s)
                s /= 2

                padding = int(nsm - n * s)
                if padding > 0:
                    for f in [fa, fb, fc, fd]:
                        f.write((pad_word + ',') * (padding - 1) + pad_word + '\n')

            for f in [fa, fb, fc, fd]:
                f.write(';\n')

        # --- LICZENIE WSZYSTKICH SŁÓW PO ZAPISIE ---
        total_words = 0
        for filename in [files[0]]:
            with open(filename, 'r') as f:
                content = f.read()
                # usuń linię z nagłówkiem i średnik na końcu
                content = content.replace('memory_initialization_radix = 16;\n', '')
                content = content.replace('memory_initialization_vector =\n', '')
                content = content.replace(';\n', '')
                # podziel po przecinkach i policz wszystkie niepuste elementy
                words = [x for x in content.split(',') if x.strip()]
                total_words += len(words)

        print(f"Łącznie słów zapisanych do pamięci (w tym padding): {total_words}")

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
