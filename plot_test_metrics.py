from pathlib import Path
from statistics import mean

from matplotlib import pyplot as plt

from models import hidden_layer_sizes, models, activations
from train_models import train_cycles
from util import get_dataset

models_path = Path('models_data')
test = get_dataset('data/test')


def test_metrics(activation: str):
    accuracies = []
    losses = []
    for hls in hidden_layer_sizes:
        model = models[activation][hls]()
        local_accuracies = []
        local_losses = []
        for i in range(train_cycles):
            weights_path = Path(models_path, activation, str(hls), str(i + 1), 'model.weights.h5')
            model.load_weights(weights_path)
            test_loss, test_accuracy = model.evaluate(test)
            local_accuracies.append(test_accuracy)
            local_losses.append(test_loss)
        accuracies.append(mean(local_accuracies))
        losses.append(mean(local_losses))
    return accuracies, losses


def plot_test_metrics():

    act_to_accuracies = {}
    act_to_losses = {}
    for act in activations:
        accuracies, losses = test_metrics(act)
        act_to_accuracies[act] = accuracies
        act_to_losses[act] = losses

    plt.figure(figsize=(16, 9))
    for act in activations:
        plt.plot(hidden_layer_sizes, act_to_accuracies[act], marker='o', linestyle='-', label=act)

    plt.title(f'Test Accuracy vs. Hidden Layer Size')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.show()

    for act in activations:
        plt.plot(hidden_layer_sizes, act_to_losses[act], marker='o', linestyle='-', label=act)

    plt.title(f'Test Loss vs. Hidden Layer Size')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.grid(True)

    plt.show()

    for act in activations:
        acc_to_loss = [acc / loss for acc, loss in zip(act_to_accuracies[act], act_to_losses[act])]
        plt.plot(hidden_layer_sizes, acc_to_loss, marker='o', linestyle='-', label=act)

    plt.title(f'Acc to Loss vs. Hidden Layer Size')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Acc to Loss')
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    plot_test_metrics()
