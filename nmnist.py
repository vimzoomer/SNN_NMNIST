import argparse
import glob
import os
import zipfile

import h5py
from lava.lib.dl.slayer.io import Event
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader, Dataset

import lava.lib.dl.slayer as slayer


class SNN(torch.nn.Module):
    def __init__(self):
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
                    neuron_params, 34*34*2, 64,
                    weight_norm=True, delay=False
                ),
                slayer.block.cuba.Dense(
                    neuron_params, 64, 128,
                    weight_norm=True, delay=False
                ),
                slayer.block.cuba.Dense(
                    neuron_params, 128, 64,
                    weight_norm=True, delay=False
                ),
                slayer.block.cuba.Dense(
                    neuron_params, 64, 10,
                    weight_norm=True, delay=False
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


def augment(event: Event) -> Event:
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
    num_classes = 10

    def __init__(self,
                 dir: str,
                 sampling_time: int,
                 sample_length: int,
                 transform: callable = None,
                 download: bool = False,
                 train: bool = True
                 ):
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
        self.data = [(filename, int(filename.split('/')[-2]))  for filename in glob.glob(f'{self.data_path}/*/*.bin')]
        self.data_by_class = [glob.glob(f'{self.data_path}/{digit}/*.bin') for digit in range(NMNIST.num_classes)]

    def __read_spike(self, filename: str) -> tensor:
        event = slayer.io.read_2d_spikes(filename)

        if self.transform is not None:
            event = self.transform(event)

        spike = event.fill_tensor(
            torch.zeros(2, 34, 34, self.num_time_bins),
            sampling_time=self.sampling_time,
        )

        return spike

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[tensor, tensor]:
        filename, label = self.data[idx]
        return self.__read_spike(filename).reshape(-1, self.num_time_bins), label

    def example_of_each_class(self, net: SNN, merge_factor: int = 4) -> None:
        frames = []
        class_bars = []
        frame_labels = []

        spikes = torch.cat([self.__read_spike(self.data_by_class[i][0]) for i in range(NMNIST.num_classes)], dim=-1)
        spike_flattened = spikes.reshape(-1, NMNIST.num_classes * self.num_time_bins).unsqueeze(0)
        classified = net(spike_flattened)

        for label in range(NMNIST.num_classes):
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
                    bar = np.zeros((1, NMNIST.num_classes))

                class_bars.append(bar)

                frame_labels.append(label)

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
        ani.save(f"vis_all_classes.gif", writer=PillowWriter(fps=30))
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train/test SNN on NMNIST")
    parser.add_argument('--dir', type=str, default='.', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--sampling_time', type=int, default=1, help='Sampling time for NMNIST')
    parser.add_argument('--sample_length', type=int, default=300, help='Sample length for NMNIST')
    parser.add_argument('--download', action='store_true', help='Download NMNIST if not present')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--load_model', type=str, default=None, help='Path to pre-trained model to load')

    args = parser.parse_args()

    transform = augment if args.augment else None

    nmnist = NMNIST(
        dir=args.dir,
        sampling_time=args.sampling_time,
        sample_length=args.sample_length,
        download=args.download,
        transform=transform
    )

    nmnist_test = NMNIST(
        dir=args.dir,
        sampling_time=args.sampling_time,
        sample_length=args.sample_length,
        train=False
    )

    model = SNN()
    if args.load_model is not None:
        print(f"Loading pre-trained model from {args.load_model}")
        state_dict = torch.load(args.load_model, map_location='cpu')
        model.load_state_dict(state_dict)

    net = model
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    nmnist_loader = DataLoader(
        dataset=nmnist, batch_size=args.batch_size, shuffle=True
    )

    nmnist_test_loader = DataLoader(
        dataset=nmnist_test, batch_size=args.batch_size, shuffle=False
    )

    error = slayer.loss.SpikeRate(
        true_rate=0.2, false_rate=0.03, reduction='sum'
    )

    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
        net, error, optimizer, stats,
        classifier=slayer.classifier.Rate.predict
    )

    for epoch in range(args.epochs):
        for i, (input, label) in enumerate(nmnist_loader):
            output = assistant.train(input, label)
            header = [
                'TRAIN\n' +
                'Output : ' + str(output[0].sum(dim=1) / output[0].sum()),
                'Label : ' + str(label[0])
            ]
            stats.print(epoch, iter=i, header=header, dataloader=nmnist_loader)

        for i, (input, label) in enumerate(nmnist_test_loader):
            output = assistant.test(input, label)
            header = [
                'TEST\n' +
                'Output : ' + str(output[0].sum(dim=1) / output[0].sum()),
                'Label : ' + str(label[0])
            ]
            stats.print(epoch, iter=i, header=header, dataloader=nmnist_test_loader)

        torch.save(net.state_dict(), args.dir + '/network.pt')
        stats.update()
