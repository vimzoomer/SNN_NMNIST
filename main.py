import argparse

from lava.lib.dl.slayer.io import Event
import numpy as np
import torch
from torch.utils.data import DataLoader

import lava.lib.dl.slayer as slayer

from datasets.cifar.cifar import CIFAR
from snn import SNN


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train/test SNN on CIFAR")
    parser.add_argument('--dir', type=str, default='.', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--sampling_time', type=int, default=1, help='Sampling time for CIFAR')
    parser.add_argument('--sample_length', type=int, default=300, help='Sample length for CIFAR')
    parser.add_argument('--download', action='store_true', help='Download CIFAR if not present')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--load_model', type=str, default=None, help='Path to pre-trained model to load')

    args = parser.parse_args()

    transform = augment if args.augment else None

    cifar = CIFAR(
        dir=args.dir,
        sampling_time=args.sampling_time,
        sample_length=args.sample_length,
        download=args.download,
        transform=transform
    )

    cifar_test = CIFAR(
        dir=args.dir,
        sampling_time=args.sampling_time,
        sample_length=args.sample_length,
        download=args.download,
        train=False
    )

    model = SNN(dataset_cls=CIFAR)
    if args.load_model is not None:
        print(f"Loading pre-trained model from {args.load_model}")
        state_dict = torch.load(args.load_model, map_location='cpu')
        model.load_state_dict(state_dict)

    net = model
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    cifar_loader = DataLoader(
        dataset=cifar, batch_size=args.batch_size, shuffle=True
    )

    cifar_test_loader = DataLoader(
        dataset=cifar_test, batch_size=args.batch_size, shuffle=False
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
        for i, (input, label) in enumerate(cifar_loader):
            output = assistant.train(input, label)
            header = [
                'TRAIN\n' +
                'Output : ' + str(slayer.classifier.Rate.predict(output)),
                'Label : ' + str(label[0])
            ]
            stats.print(epoch, iter=i, header=header, dataloader=cifar_loader)

        torch.save(net.state_dict(), args.dir + '/network.pt')
        stats.update()

    cifar.example_of_each_class(model, 4)
