import tensorflow as tf
from keras import preprocessing
from keras.api.layers import Rescaling


def get_dataset(path: str):
    raw: tf.data.Dataset = preprocessing.image_dataset_from_directory(
        directory=path,
        image_size=(50, 50),
        label_mode='categorical',
        batch_size=32
    )
    normalization_layer = Rescaling(1. / 255)
    return raw.map(lambda x, y: (normalization_layer(x), y))

