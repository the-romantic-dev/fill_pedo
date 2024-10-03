from typing import Callable

from keras import Sequential
from keras.src.layers import Flatten, Dense

input_shape = (50, 50, 3)
activations = ['relu', 'elu', 'leaky_relu']
hidden_layer_sizes = [10, 50, 75, 100, 150, 200]
num_classes = 4


def create_model(hidden_layer_size: int, activation: str):
    model = Sequential(
        [
            Flatten(input_shape=input_shape, name="Input"),
            Dense(units=hidden_layer_size, activation=activation, name="Hidden"),
            Dense(units=num_classes, activation='softmax')
        ]
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


models: dict[str, dict[int, Callable[[], Sequential]]] = {
    act: {
        hls: lambda: create_model(hls, act) for hls in hidden_layer_sizes
    } for act in activations
}


