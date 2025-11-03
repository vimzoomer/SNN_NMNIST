import glob
import os
import zipfile

from datasets.base_dataset import BaseDataset


class NMNIST(BaseDataset):
    num_classes = 10
    width = 34
    height = 34
    num_channels = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_data(self, download: bool = True, train: bool = True) -> None:
        if train:
            self.data_path = self.dir + '/datasets/nmnist/Train'
            source = 'https://www.dropbox.com/sh/tg2ljlbmtzygrag/' \
                     'AABlMOuR15ugeOxMCX0Pvoxga/Train.zip'
        else:
            self.data_path = self.dir + '/datasets/nmnist/Test'
            source = 'https://www.dropbox.com/sh/tg2ljlbmtzygrag/' \
                     'AADSKgJ2CjaBWh75HnTNZyhca/Test.zip'

        if download:
            if len(glob.glob(f'{self.data_path}')) == 0:
                print('Attempting download...')
                os.system(f'wget {source} -P {self.dir}/datasets/nmnist/ -q --show-progress')
                print('Extracting files ...')
                with zipfile.ZipFile(self.data_path + '.zip') as zip_file:
                    for member in zip_file.namelist():
                        zip_file.extract(member, f"{self.dir}/datasets/nmnist/")
                print('Download complete.')

        print(glob.glob(f'{self.data_path}/*/*.bin'))
        self.data = [(filename, int(filename.split('/')[-2]))  for filename in glob.glob(f'{self.data_path}/*/*.bin')]
        self.data_by_class = [glob.glob(f'{self.data_path}/{digit}/*.bin') for digit in range(self.num_classes)]
