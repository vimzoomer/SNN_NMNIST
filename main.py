import argparse

from lava.lib.dl.slayer.io import Event
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import lava.lib.dl.slayer as slayer

from datasets.nmnist.nmnist import NMNIST
from deployment import Deployment
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
    parser = argparse.ArgumentParser(description="Train/test SNN on NMNIST")
    parser.add_argument('--dir', type=str, default='.', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
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
        download=args.download,
        train=False
    )

    model = SNN(dataset_cls=NMNIST)
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
                'Output : ' + str(slayer.classifier.Rate.predict(output)),
                'Label : ' + str(label[0])
            ]
            stats.print(epoch, iter=i, header=header, dataloader=nmnist_loader)

        torch.save(net.state_dict(), args.dir + '/network.pt')
        stats.update()

    Deployment(net).deploy()

    nmnist.get_spike_train(net, 0)

    """
    nmnist.example_of_each_class(net, 1)

    def weight_to_color(value, max, min):
        if value < 0:
            intensity = int(255 * abs(value) / abs(min))
            return (0, 0, intensity)
        elif value > 0:
            intensity = int(255 * value / abs(max))
            return (intensity, 0, 0)
        else:
            return (0, 0, 0)

    weights = net.blocks[0].synapse.weight[:,:,0,0,0]
    weights_reshaped = weights.reshape(2, 34, 34, 64)
    weights_1 = weights_reshaped[0, :, :, 32]
    weights_2 = weights_reshaped[1, :, :, 32]

    def visualize_weights(weights_1, weights_2):
        max_w = max(torch.max(weights_1), torch.max(weights_2))
        min_w = min(torch.min(weights_1), torch.min(weights_2))

        def make_image(weights):
            h, w = weights.shape
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    img[i, j] = weight_to_color(weights[i, j], max_w, min_w)
            return img

        img1 = make_image(weights_1)
        img2 = make_image(weights_2)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img1)
        axes[0].set_title("Weights 1")
        axes[0].axis("off")

        axes[1].imshow(img2)
        axes[1].set_title("Weights 2")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
    
    visualize_weights(weights_1, weights_2)
    """