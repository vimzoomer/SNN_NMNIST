# SNN NMNIST

This project implements Spiking Neural Networks (SNN) in PyTorch/Lava-DL and allows training on the NMNIST dataset, currently other datasets are added too. Based on the official lava-dl tutorials: https://github.com/lava-nc/lava-dl.

## Requirements

* Python 3.10
* Poetry 2.x
* wget
* Jupyter Notebook / Jupyter Lab
* When you use Windows running in WSL is recommended

## Installation

1. Clone the repository

```
git clone https://github.com/vimzoomer/SNN_NMNIST
cd SNN_NMNIST
```

2. Install dependencies

```
poetry install
```

3. Activate the project environment

Since Poetry 2.x, `poetry shell` is not installed by default. Use the following commands to activate the environment:

```
poetry env activate
source /path/to/venv/bin/activate  # replace with the path from previous command
```

## Preparing NMNIST Data

By default, the data is downloaded automatically when the `NMNIST` class is used for the first time with `download=True`.
To download manually:

```
wget https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABlMOuR15ugeOxMCX0Pvoxga/Train.zip -P ./datasets/nmnist/ -q --show-progress
wget https://www.dropbox.com/sh/tg2ljlbmtzygrag/AADSKgJ2CjaBWh75HnTNZyhca/Test.zip -P ./datasets/nmnist/ -q --show-progress
```

Some other datasets don't have automatic download.

## Running training via Script

The script `main.py` accepts the following command-line arguments:

* `--dir`: Dataset directory (default: `.`)
* `--epochs`: Number of training epochs (default: 2)
* `--batch_size`: Batch size for training and testing (default: 32)
* `--sampling_time`: Sampling time for spike events (default: 1)
* `--sample_length`: Length of each sample in time bins (default: 300)
* `--download`: Download dataset if not present (flag, default: False)
* `--augment`: Apply data augmentation to training data (flag, default: False)
* `--load_model`: Path to pre-trained model to load before training (default: None)

Example usage:

```
poetry run python ./main.py --epochs 5 --batch_size 64 --download --augment
```

## Running training in Jupyter Notebook

1. Activate Poetry environment as described above.

2. Launch Jupyter Notebook:

```
jupyter notebook snn_training.ipynb
```

3. The notebook already contains dataset setup, model initialization, training loops, and visualization. You can run the cells directly.

## Visualizing examples

The code provides the `example_of_each_class` function in the `NMNIST` class, which generates a GIF animation showing examples of all classes and network activity:

```python
nmnist.example_of_each_class(net, merge_factor)
```

* `merge_factor`: An integer that decides how many time bins should be merged into a single frame. Increasing merge_factor reduces the number of frames and can make the animation smoother.

The output will be saved as `vis_all_classes.gif`.

## Dependencies

The project uses the following packages:

* lava-dl (from GitHub)
* jupyter
* notebook
* ipykernel
* matplotlib
* numpy
* torch
* h5py
