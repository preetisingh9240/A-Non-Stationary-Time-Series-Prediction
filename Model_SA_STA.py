import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from sklearn.model_selection import train_test_split


class SpatialAttention(layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = layers.Conv1D(1, kernel_size=1, activation='sigmoid')

    def call(self, inputs):
        # Shape: (batch, time, features)
        avg_pool = tf.reduce_mean(inputs, axis=2, keepdims=True)  # (batch, time, 1)
        max_pool = tf.reduce_max(inputs, axis=2, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)  # (batch, time, 2)
        attention = self.conv(concat)  # (batch, time, 1)
        return inputs * attention


class TemporalAttention(layers.Layer):
    def __init__(self):
        super(TemporalAttention, self).__init__()
        self.conv = layers.Conv1D(1, kernel_size=1, activation='sigmoid')

    def call(self, inputs):
        # Shape: (batch, time, features)
        avg_pool = tf.reduce_mean(inputs, axis=1, keepdims=True)  # (batch, 1, features)
        max_pool = tf.reduce_max(inputs, axis=1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=1)  # (batch, 2, features)
        attention = self.conv(concat)  # (batch, 2, 1)
        attention = tf.reduce_mean(attention, axis=1, keepdims=True)  # (batch, 1, 1)
        return inputs * attention


def sparse_loss_fn(sparsity_level, lambda_sparse, encoder_output):
    mean_activation = tf.reduce_mean(encoder_output, axis=0)
    kl_div = tf.reduce_sum(
        sparsity_level * tf.math.log(sparsity_level / (mean_activation + 1e-10)) +
        (1 - sparsity_level) * tf.math.log((1 - sparsity_level) / (1 - mean_activation + 1e-10))
    )
    return lambda_sparse * kl_div


def Model_STA_SA(Data, Targets, EP=20):
    if len(Data.shape) == 2:
        Data = np.expand_dims(Data, axis=1)

    # input_shape = Data.shape[1:]  # (timesteps, features)
    sparsity_level = 0.05
    lambda_sparse = 0.1
    timesteps = 5

    input_shape = (timesteps, Data.shape[-1])

    Datas = np.zeros((Data.shape[0], input_shape[0], input_shape[1]))
    for i in range(Data.shape[0]):
        Datas[i, :] = np.resize(Data[i], (input_shape[0], input_shape[1]))
    X = Datas.reshape(Datas.shape[0], input_shape[0], input_shape[1])

    x_train, x_test, _, _ = train_test_split(X, Targets, test_size=0.2, random_state=42)

    inputs = layers.Input(shape=input_shape)

    # Attention
    x = SpatialAttention()(inputs)
    x = TemporalAttention()(x)

    # Encoder
    x = layers.LayerNormalization()(x)
    encoded = layers.Flatten()(x)
    encoded = layers.Dense(64, activation='relu')(encoded)

    # Decoder
    decoded = layers.Dense(np.prod(input_shape), activation='sigmoid')(encoded)
    decoded = layers.Reshape(input_shape)(decoded)

    model = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    def combined_loss(y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        encoder_out = encoder(y_true)
        sparse_loss = sparse_loss_fn(sparsity_level, lambda_sparse, encoder_out)
        return mse_loss + sparse_loss

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=combined_loss, metrics=['accuracy'])
    model.summary()
    model.fit(x_train, x_train, epochs=EP, batch_size=8, validation_data=(x_test, x_test), shuffle=True)
    layerNo = -2
    intermediate_model = Model(inputs=model.input, outputs=model.layers[layerNo].output)
    Feats = intermediate_model.predict(np.concatenate((x_train, x_test)))
    Feats = np.asarray(Feats)
    features = np.resize(Feats, (Feats.shape[0], Feats.shape[-1]))
    return features

