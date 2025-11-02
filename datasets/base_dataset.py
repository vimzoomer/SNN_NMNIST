from abc import ABC, abstractmethod

import numpy as np
import torch
from lava.lib.dl import slayer
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from torch import tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    num_classes = None
    width = None
    height = None
    num_channels = None

    def __init__(self,
                 dir: str,
                 sampling_time: int,
                 sample_length: int,
                 transform: callable = None,
                 download: bool = False,
                 train: bool = True
                 ):
        self.dir = dir

        self.data = None
        self.data_by_class = None
        self._get_data(download=download, train=train)

        self.transform = transform
        self.sampling_time = sampling_time
        self.sample_length = sample_length
        self.num_time_bins = int(sample_length / sampling_time)

    @abstractmethod
    def _get_data(self, download: bool, train: bool):
        pass

    def _read_spike(self, filename: str) -> tensor:
        event = slayer.io.read_2d_spikes(filename)

        if self.transform is not None:
            event = self.transform(event)

        spike = event.fill_tensor(
            torch.zeros(self.num_channels, self.height, self.width, self.num_time_bins),
            sampling_time=self.sampling_time,
        )

        return spike

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[tensor, tensor]:
        filename, label = self.data[idx]
        return self._read_spike(filename).reshape(-1, self.num_time_bins), label

    def get_spike_train(self, net: "SNN", cls: int, filepath: str = "spikes.txt"):
        torch.set_printoptions(threshold=torch.inf)

        spikes = self._read_spike(self.data_by_class[cls][0])
        spikes_flattened = spikes.reshape(-1, self.num_time_bins).unsqueeze(0)
        output = net(spikes_flattened)
        print("Output spike rate: ", slayer.classifier.Rate.rate(output))

        flat = spikes_flattened.T.flatten()

        pad = (4 - (flat.numel() % 4)) % 4
        if pad > 0:
            flat = torch.nn.functional.pad(flat, (0, pad), value=0)

        grouped = flat.reshape(-1, 4)

        weights = torch.tensor([8, 4, 2, 1], device=flat.device)
        ints = (grouped * weights).sum(dim=-1).to(torch.int32)

        hex_seq = [format(v.item(), 'X') for v in ints]

        with open(filepath, "w") as f:
            f.write("\n".join(hex_seq))

        print(f"Saved {len(hex_seq)} hex values to {filepath}")

    def example_of_each_class(self, net: "SNN", merge_factor: int = 4) -> None:
        frames = []
        class_bars = []
        frame_labels = []

        spikes = torch.cat([self._read_spike(self.data_by_class[i][0]) for i in range(self.num_classes)], dim=-1)
        spike_flattened = spikes.reshape(-1, self.num_classes * self.num_time_bins).unsqueeze(0)
        classified = net(spike_flattened)

        for label in range(self.num_classes):
            for t in range(label * self.num_time_bins, (label+1) * self.num_time_bins):
                red_channel = spikes[1, :, :, t].numpy() * self.sampling_time
                blue_channel = spikes[0, :, :, t].numpy() * self.sampling_time
                print(np.sum(red_channel))

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
                    bar = np.zeros((1, self.num_classes))

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