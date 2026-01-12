from keras.models import Sequential, Model
from keras.layers import Bidirectional, LSTM, Dense, Input
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from evaluate_error import evaluate_error
import numpy as np


def Model_BiLSTM_RF_MPA(trainX, trainY, testX, testY, BS=None, EP=None):
    print('BiLSTM + RF + MPA')

    if BS is None:
        BS = 32
    if EP is None:
        EP = 10

    IMG_SIZE = [1, 100]
    num_classes = testY.shape[-1]

    # Reshape training data
    Train_Temp = np.zeros((trainX.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(trainX.shape[0]):
        Train_Temp[i, :] = np.resize(trainX[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    Test_Temp = np.zeros((testX.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(testX.shape[0]):
        Test_Temp[i, :] = np.resize(testX[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Test_X = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    # BiLSTM model to extract features
    input_layer = Input(shape=(Train_X.shape[1], Train_X.shape[-1]))
    x = Bidirectional(LSTM(50, activation='relu', return_sequences=False))(input_layer)
    feature_layer = Dense(32, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(feature_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_X, trainY, epochs=EP, batch_size=BS, verbose=1, validation_data=(Test_X, testY))

    # Feature extraction from penultimate layer
    feature_model = Model(inputs=input_layer, outputs=feature_layer)
    train_features = feature_model.predict(Train_X)
    test_features = feature_model.predict(Test_X)

    # Random Forest Classifier
    y_train_labels = np.argmax(trainY, axis=1)
    y_test_labels = np.argmax(testY, axis=1)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_features, y_train_labels)
    y_pred = rf.predict(test_features)

    pred_onehot = np.zeros((len(y_pred), num_classes))
    pred_onehot[np.arange(len(y_pred)), y_pred] = 1

    Eval = evaluate_error(testY, pred_onehot)
    return Eval, pred_onehot
