import glob
import os
import zipfile

from datasets.base_dataset import BaseDataset


class NCALTECH(BaseDataset):
    num_classes = 101
    width = 128
    height = 128
    num_channels = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_data(self, download: bool = True, train: bool = True) -> None:
        if not train: print("Warning! This dataset does not have test data, falling back to training data.")
        self.data_path = os.path.join(self.dir, 'datasets/ncaltech/Caltech101')

        source = 'https://www.dropbox.com/scl/fo/4cbdh0uizneotdht26f2z/AO3Gt9s5jboBLBR4NBvpIA0?rlkey=z5weoe8zmw99fgj0taq9yyks1&e=1&dl=1'
        zip_path_1 = os.path.join(self.dir, 'datasets/ncaltech', 'N-Caltech101.zip')
        zip_path_2 = os.path.join(self.dir, 'datasets/ncaltech', 'Caltech101.zip')

        if download and not os.path.exists(self.data_path):
            print('Attempting download...')
            os.system(f'wget "{source}" -O "{zip_path_1}" -q --show-progress')
            print('Extracting files ...')
            with zipfile.ZipFile(zip_path_1, 'r') as zip_file:
                zip_file.extractall(os.path.join(self.dir, 'datasets/ncaltech'))
            with zipfile.ZipFile(zip_path_2, 'r') as zip_file:
                zip_file.extractall(os.path.join(self.dir, 'datasets/ncaltech'))
            print('Download complete.')

        class_names = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        class_names.sort()
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        self.data = [(filename, self.class_to_idx[filename.split('/')[-2]])
                     for filename in glob.glob(f'{self.data_path}/*/*.bin')]
        self.data_by_class = [glob.glob(os.path.join(self.data_path, name, '*.bin')) for name in class_names]



