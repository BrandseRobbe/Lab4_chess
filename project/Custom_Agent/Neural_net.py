import tensorflow
from tensorflow.keras import Sequential, Model, activations
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Input, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.losses import MeanSquaredError


def create_utilitymodel():
    model = Sequential([
        Input(shape=(8, 8, 13)),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPool2D((2, 2)),

        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPool2D((2, 2)),
        Flatten(),

        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(), loss=MeanSquaredError())
    return model
