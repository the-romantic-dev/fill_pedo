import json
from pathlib import Path

from matplotlib import pyplot as plt

from models import hidden_layer_sizes
from train_models import train_cycles

from statistics import mean

models_path = Path('models_data')


def plot_metric(act: str, metric, is_validation: bool = False):
    average_per_hls = {}
    for hls in hidden_layer_sizes:
        metric_per_hls = []
        for i in range(train_cycles):
            with open(Path(models_path, act, str(hls), str(i + 1), 'history.json'), 'r') as file:
                history = json.load(file)
            key = metric
            if is_validation:
                key = f'val_{metric}'
            metric_per_hls.append(history[key])
        average_per_hls[hls] = [mean(attempt) for attempt in zip(*metric_per_hls)]
        if is_validation:
            print(f"{act} {hls} {metric} : {average_per_hls[hls][-1]}")

    for hls in hidden_layer_sizes:
        plt.plot(average_per_hls[hls], label=f'{hls} neurons')
    if is_validation:
        plt.title(f'Validation {metric} for {act}')
    else:
        plt.title(f'Train {metric} for {act}')
    plt.ylabel(f'{metric}')
    plt.xlabel('Epoch')
    plt.legend()


def plot_accuracy():
    plt.figure(figsize=(16, 9))

    plt.subplot(3, 2, 1)
    plot_metric(act='relu', metric='accuracy')
    plt.subplot(3, 2, 2)
    plot_metric(act='relu', metric='accuracy', is_validation=True)

    plt.subplot(3, 2, 3)
    plot_metric(act='elu', metric='accuracy')
    plt.subplot(3, 2, 4)
    plot_metric(act='elu', metric='accuracy', is_validation=True)

    plt.subplot(3, 2, 5)
    plot_metric(act='leaky_relu', metric='accuracy')
    plt.subplot(3, 2, 6)
    plot_metric(act='leaky_relu', metric='accuracy', is_validation=True)

    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    plt.show()


def plot_loss():
    plt.figure(figsize=(16, 9))

    plt.subplot(3, 2, 1)
    plot_metric(act='relu', metric='loss')
    plt.subplot(3, 2, 2)
    plot_metric(act='relu', metric='loss', is_validation=True)

    plt.subplot(3, 2, 3)
    plot_metric(act='elu', metric='loss')
    plt.subplot(3, 2, 4)
    plot_metric(act='elu', metric='loss', is_validation=True)

    plt.subplot(3, 2, 5)
    plot_metric(act='leaky_relu', metric='loss')
    plt.subplot(3, 2, 6)
    plot_metric(act='leaky_relu', metric='loss', is_validation=True)

    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_accuracy()
    plot_loss()