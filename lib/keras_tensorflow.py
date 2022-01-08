import pandas as p
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from six import string_types
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.backend import dtype

def neural_network(X, Y):
    # Basic Neural network
    Y_encoded = []
    for type in Y:
        if type == "BRCA":
            Y_encoded.append(0)
        elif type == "PRAD":
            Y_encoded.append(1)
        elif type == "LUAD":
            Y_encoded.append(2)
        elif type == "KIRC":
            Y_encoded.append(3)
        elif type == "COAD":
            Y_encoded.append(4)
    Y_bis = to_categorical(Y_encoded, num_classes=5, dtype=np.int32)
    x_train, x_test, y_train, y_test = train_test_split(X, Y_bis, test_size=0.33, random_state=42, stratify=Y)
    init = 'random_uniform'
    input_layer = Input(shape=(20531,))
    mid_layer = Dense(15, activation= 'relu', kernel_initializer=init)(input_layer)
    mid_layer2 = Dense(8, activation= 'relu', kernel_initializer=init)(mid_layer)
    ouput_layer = Dense(5, activation= 'softmax', kernel_initializer=init)(mid_layer2)

    model = Model(input_layer, ouput_layer)

    model.compile(optimizer='sgd', loss="binary_crossentropy", metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1)

    y_predict = model.predict(x_test)

    prediction = np.argmax(y_predict, axis=1)
    y_expect = np.argmax(y_test, axis=1)

    crosstable = p.crosstab(np.argmax(y_test, axis=1), prediction)
    print(crosstable)

    accuracy_test = accuracy_score(y_expect, prediction)
    print(accuracy_test)

if __name__ == "__main__":
    # transformation des fichiers CSV en DataFrame permettant le traitement
    df_data = p.read_csv("../data/data.csv", header=1)
    df_label = p.read_csv("../data/labels.csv", header=1)
    Y = df_label.iloc[:,1]
    X = df_data.iloc[:,1:]
    neural_network(X, Y)