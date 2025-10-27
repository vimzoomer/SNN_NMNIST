import glob
import os
import zipfile

import numpy as np
import torch
from lava.lib.dl.slayer.io import Event

from datasets.base_dataset import BaseDataset
from datasets.load import load_raw_events, parse_raw_address, load_events


class CIFAR(BaseDataset):
    num_classes = 10
    width = 128
    height = 128
    num_channels = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _read_spike(self, filename: str) -> torch.Tensor:
        with open(filename, 'rb') as fp:
            t, x, y, p = load_events(fp,
                                     x_mask=0xfE,
                                     x_shift=1,
                                     y_mask=0x7f00,
                                     y_shift=8,
                                     polarity_mask=1,
                                     polarity_shift=None)

        t = t / 1000
        x = 127 - x
        y = 127 - y
        p = 1 - p.astype(int)
        event_obj = Event(x, y, p, t)

        if self.transform is not None:
            event_obj = self.transform(event_obj)

        spike = event_obj.fill_tensor(
            torch.zeros(self.num_channels, self.height, self.width, self.num_time_bins),
            sampling_time=self.sampling_time
        )

        return spike

    def _get_data(self, download: bool = True, train: bool = True) -> None:
        if not train:
            print("Warning! This dataset does not have test data, falling back to training data.")
        if download:
            print("This dataset can't be downloaded automatically, make sure the zip files exist in datasets/cifar!")

        self.data_path = os.path.join(self.dir, 'datasets/cifar/')

        zip_folder = os.path.join(self.dir, 'datasets/cifar')
        for zip_file in glob.glob(os.path.join(zip_folder, '*.zip')):
            extract_name = os.path.splitext(os.path.basename(zip_file))[0]
            extract_path = os.path.join(zip_folder, extract_name)

            if os.path.exists(extract_path):
                print(f'{extract_path} already exists, skipping extraction.')
                continue

            print(f'Extracting {zip_file} ...')
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(zip_folder)

        print('Extraction complete.')

        class_names = [d for d in os.listdir(self.data_path)
                       if os.path.isdir(os.path.join(self.data_path, d)) and not d.startswith('__')]
        class_names.sort()
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.data = []
        for class_name in class_names:
            class_folder = os.path.join(self.data_path, class_name)
            for file_path in glob.glob(os.path.join(class_folder, '*.aedat')):
                self.data.append((file_path, self.class_to_idx[class_name]))

        self.data_by_class = [
            glob.glob(os.path.join(self.data_path, name, '*.aedat')) for name in class_names
        ]
