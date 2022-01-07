import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model, activations
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Input, Flatten, Dropout, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.losses import MeanSquaredError


def create_utilitymodel():
    # model to process board information
    board_model_input = Input(name="board_data", shape=(8, 8, 13))
    board_model = Sequential(name="board_model", layers=[
        board_model_input,
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
        # Dense(1, activation='linear')
    ])

    # model to process extracted features
    n_features = 2
    features_model_input = Input(name="feature_data", shape=n_features)
    features_model = Sequential(name="features_model", layers=[
        features_model_input,
        Dense(8, activation='relu'),
        Dense(8, activation='relu'),
    ])

    # model to combine the two
    board_model_output = board_model(board_model_input)
    features_model_output = features_model(features_model_input)

    combo = concatenate([board_model_output, features_model_output])
    x = Dense(8, activation='relu')(combo)
    x = Dense(8, activation='relu')(x)
    x = Dense(4, activation='relu')(x)
    model_output = Dense(1, activation='linear')(x)

    model = Model(name="Utility_model", inputs=[board_model_input, features_model_input], outputs=model_output)
    model.compile(optimizer=Adam(), loss=MeanSquaredError())
    return model

# model = create_utilitymodel()
# print(model.summary())
# keras.utils.plot_model(model, expand_nested=True, show_shapes=True)
