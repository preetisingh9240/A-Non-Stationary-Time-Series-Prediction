import numpy as np
import tensorflow as tf
from keras.src import optimizers
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from evaluate_error import evaluate_error


class ESNLayer(tf.keras.layers.Layer):
    def __init__(self, units, spectral_radius=0.95, **kwargs):
        super(ESNLayer, self).__init__(**kwargs)
        self.units = units
        self.spectral_radius = spectral_radius

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W_in = self.add_weight(shape=(input_dim, self.units),
                                    initializer='glorot_uniform',
                                    trainable=True,
                                    name='W_in')

        # Spectral normalized internal weights (non-trainable)
        W = np.random.uniform(-1, 1, (self.units, self.units))
        eigvals = np.linalg.eigvals(W)
        W *= self.spectral_radius / np.max(np.abs(eigvals))
        self.W = tf.constant(W, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        def step_fn(prev_state, x_t):
            return tf.nn.tanh(tf.matmul(x_t, self.W_in) + tf.matmul(prev_state, self.W))

        batch_size = tf.shape(inputs)[0]
        initial_state = tf.zeros((batch_size, self.units), dtype=tf.float32)

        inputs_trans = tf.transpose(inputs, [1, 0, 2])  # [time, batch, features]
        outputs = tf.scan(step_fn, inputs_trans, initializer=initial_state)
        return tf.transpose(outputs, [1, 0, 2])  # [batch, time, units]


def mar_esn(feature_shapes, Num_classes, sol, esn_units=64):
    inputs = []
    reshaped = []

    for shape in feature_shapes:
        inp = Input(shape=(shape,))
        inputs.append(inp)
        # Reshape to (batch, time=shape, 1) to simulate time series
        reshaped.append(tf.keras.layers.Reshape((shape, 1))(inp))
    merged = Concatenate(axis=1)(reshaped)  # [batch, total_time, 1]
    esn_out = ESNLayer(esn_units)(merged)
    pooled = GlobalAveragePooling1D()(esn_out)
    out_1 = Dense(int(sol[0]), activation='softmax')(pooled)  # 64
    out = Dense(Num_classes, activation='sigmoid')(out_1)
    model = Model(inputs=inputs, outputs=out)
    return model


def Model_MAR_ESN(Feat_1, Feat_2, Feat_3, Target, sol=None, BS=None, Perc=None):

    if sol is None:
        sol = [5, 0.01, 100]
    if BS is None:
        BS = 32
    if Perc is None:
        Perc = 10

    Num_classes = Target.shape[-1]
    feature_shapes = [Feat_1.shape[-1], Feat_2.shape[-1], Feat_3.shape[-1]]
    X_train_1, X_test_1 = Feat_1[Perc:],  Feat_1[-Perc:]
    X_train_2, X_test_2 = Feat_2[Perc:],  Feat_2[-Perc:]
    X_train_3, X_test_3 = Feat_3[Perc:],  Feat_3[-Perc:]
    y_train, y_test = Target[Perc:],  Target[-Perc:]
    model = mar_esn(feature_shapes, Num_classes, sol)
    model.summary()
    opt = optimizers.Nadam(learning_rate=sol[1])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])  # binary_crossentropy
    model.fit(x=[X_train_1, X_train_2, X_train_3], y=y_train, epochs=5, batch_size=BS, steps_per_epoch=int(sol[2]),
              shuffle=True, validation_data=([X_test_1, X_test_2, X_test_3], y_test))
    pred = model.predict([X_test_1, X_test_2, X_test_3])
    Eval = evaluate_error(y_test, pred)
    return Eval, pred


