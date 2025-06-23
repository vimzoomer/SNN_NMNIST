import glob
import os
import zipfile

import h5py
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import lava.lib.dl.slayer as slayer

def augment(event):
    x_shift = 4
    y_shift = 4
    theta = 10
    xjitter = np.random.randint(2*x_shift) - x_shift
    yjitter = np.random.randint(2*y_shift) - y_shift
    ajitter = (np.random.rand() - 0.5) * theta / 180 * 3.141592654
    sin_theta = np.sin(ajitter)
    cos_theta = np.cos(ajitter)
    event.x = event.x * cos_theta - event.y * sin_theta + xjitter
    event.y = event.x * sin_theta + event.y * cos_theta + yjitter
    return event

class NMNIST(Dataset):
    def __init__(self, dir, sampling_time, sample_length, transform=None, download=False, train=True):
        self.dir = dir

        if train:
            self.data_path = self.dir + '/Train'
            source = 'https://www.dropbox.com/sh/tg2ljlbmtzygrag/' \
                     'AABlMOuR15ugeOxMCX0Pvoxga/Train.zip'
        else:
            self.data_path = self.dir + '/Test'
            source = 'https://www.dropbox.com/sh/tg2ljlbmtzygrag/' \
                     'AADSKgJ2CjaBWh75HnTNZyhca/Test.zip'

        if download:
            if len(glob.glob(f'{self.data_path}')) == 0:
                print('Attempting download...')
                os.system(f'wget {source} -P {self.dir}/ -q --show-progress')
                print('Extracting files ...')
                with zipfile.ZipFile(self.data_path + '.zip') as zip_file:
                    for member in zip_file.namelist():
                        zip_file.extract(member, self.dir)
                print('Download complete.')

        self.transform = transform
        self.sampling_time = sampling_time
        self.sample_length = sample_length
        self.num_time_bins = int(sample_length / sampling_time)
        self.data = [glob.glob(f'{self.data_path}/{digit}/*.bin') for digit in range(10)]
        self.num_labels = len(self.data)

    def __read_spike(self, filename):
        event = slayer.io.read_2d_spikes(filename)

        if self.transform is not None:
            event = self.transform(event)

        spike = event.fill_tensor(
            torch.zeros(2, 34, 34, self.num_time_bins),
            sampling_time=self.sampling_time,
        )

        return spike

    def __len__(self):
        return sum(len(sublist) for sublist in self.data)

    def __getitem__(self, idx):
        flat_index = 0
        filename = None
        for sublist in self.data:
            if idx < flat_index + len(sublist):
                filename = sublist[idx - flat_index]
                break
            flat_index += len(sublist)

        return self.__read_spike(filename).reshape(-1, self.num_time_bins)

    def example_of_each_class(self, net):
        frames = []
        class_bars = []
        frame_labels = []

        spikes = torch.cat([self.__read_spike(self.data[i][0]) for i in range(self.num_labels)], dim=-1)
        spike_flattened = spikes.reshape(-1, self.num_labels * self.num_time_bins).unsqueeze(0)
        classified, _ = net(spike_flattened)

        for label in range(self.num_labels):
            for t in range(label * self.num_time_bins, (label+1) * self.num_time_bins):
                red_channel = spikes[1, :, :, t].numpy() * self.sampling_time
                blue_channel = spikes[0, :, :, t].numpy() * self.sampling_time

                rgb_image = np.zeros((red_channel.shape[0], red_channel.shape[1], 3), dtype=float)
                rgb_image[:, :, 0] = red_channel
                rgb_image[:, :, 2] = blue_channel
                frames.append(rgb_image)

                t0 = max(0, t - self.num_time_bins // 2)
                window = classified[:, :, t0:t]

                spike_sum = window.sum()
                if spike_sum.item() > 0:
                    bar = (window.sum(dim=2) / spike_sum).detach().cpu().numpy()
                else:
                    bar = np.zeros((1, self.num_labels))

                class_bars.append(bar)

                frame_labels.append(label)

        merge_factor = 4

        frames = [
            np.maximum.reduce(frames[i:i + merge_factor], axis=0)
            for i in range(0, len(frames), merge_factor)
        ]

        frame_labels = [
            np.min(frame_labels[i:i + merge_factor], axis=0)
            for i in range(0, len(frame_labels), merge_factor)
        ]

        class_bars = [
            np.mean(class_bars[i:i + merge_factor], axis=0)
            for i in range(0, len(class_bars), merge_factor)
        ]

        fig, (ax_img, ax_class) = plt.subplots(1, 2, figsize=(8, 4))

        im = ax_img.imshow(frames[0], interpolation='none', animated=True)
        ax_img.set_title(f"Label: {frame_labels[0]}")
        ax_img.axis('off')

        im_class = ax_class.imshow(class_bars[0], cmap='gray', aspect='auto', vmin=0, vmax=1, animated=True)
        ax_class.set_title("Class Activity")
        ax_class.set_yticks([])
        ax_class.set_xticks(range(class_bars[0].shape[1]))

        def update(i):
            im.set_data(frames[i])
            im_class.set_data(class_bars[i])
            ax_img.set_title(f"Label: {int(frame_labels[i])}")
            return [im, im_class]

        ani = FuncAnimation(fig, update, frames=len(frames), interval=20, blit=True)
        ani.save(f"vis_all_samples.gif", writer=PillowWriter(fps=30))
        plt.close(fig)


class SNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        neuron_params = {
            'threshold': 1.25,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 3,
            'requires_grad': False,
        }

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.Dense(
                    neuron_params, 34*34*2, 512,
                    weight_norm=True, delay=False
                ),
                slayer.block.cuba.Dense(
                    neuron_params, 512, 512,
                    weight_norm=True, delay=False
                ),
                slayer.block.cuba.Dense(
                    neuron_params, 512, 10,
                    weight_norm=True, delay=False
                )
            ]
        )

    def forward(self, spike):
        count = []
        for block in self.blocks:
            spike = block(spike)
            count.append(torch.mean(spike).item())
        return spike, torch.FloatTensor(count).reshape(
            (1, -1)).to(spike.device)

    def export_hdf5(self, filename):
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))


if __name__ == '__main__':
    trained_dir = './Train'
    nmnist = NMNIST(trained_dir, 1, 300)

    net = SNN()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    nmnist_loader = DataLoader(
        dataset=nmnist, batch_size=32, shuffle=True
    )

    error = slayer.loss.SpikeRate(
        true_rate=0.2, false_rate=0.03, reduction='sum'
    )

    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
        net, error, optimizer, stats,
        classifier=slayer.classifier.Rate.predict, count_log=True
    )

    epochs = 200

    for epoch in range(epochs):
        for i, (input, label) in enumerate(nmnist_loader):
            output, count = assistant.train(input, label)
            header = [
                'Event rate : ' +
                ', '.join([f'{c.item():.4f}' for c in count.flatten()]),
                'Output : ' + str(output[0].sum(dim=1) / output[0].sum()),
                'Label : ' + str(label[0])
            ]
            stats.print(epoch, iter=i, header=header, dataloader=nmnist_loader)

        stats.update()





