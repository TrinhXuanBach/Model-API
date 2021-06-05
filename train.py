import pickle

from sklearn.preprocessing import StandardScaler

from utls import result_url, save_url, number_labels, fake_db, test, number_width_feature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow import keras


class Trainee:
    def __init__(self):
        self.history = None
        self.data = pd.read_csv(save_url)
        self.train_data = pd.read_csv(result_url)
        self.labels = []
        self.feature = []
        self.model = Sequential()
        self.early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

    def config_model(self):
        self.model.add(Dense(number_width_feature, input_shape=(number_width_feature,), activation='relu'))
        self.model.add(Dropout(0.1))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(number_labels, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    def encoder_labels(self):
        i = 0
        old_lables = self.data['labels'][0]
        for row in self.data['labels']:
            if row != old_lables:
                self.labels.append(i)
                old_lables = row
                i = i + 1
            else:
                self.labels.append(i)
        self.labels.remove(0)
        self.labels = np.array(self.labels)

    def read_data(self):
        for i, row in self.train_data.iterrows():
            if i == 0:
                continue
            self.feature.append(np.array(row[1:len(row)]))
        self.feature = np.array(self.feature)

    def split_data(self):
        X_train, X_test_val, y_train, y_test_val = train_test_split(self.feature, self.labels, test_size=0.3,
                                                                    random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.333, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_model(self):
        self.model.save("model")


if __name__ == "__main__":
    trainee = Trainee()
    trainee.read_data()
    trainee.encoder_labels()
    trainee.config_model()
    X_train, X_val, X_test, y_train, y_val, y_test = trainee.split_data()
    y_check = y_test
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_val = ss.transform(X_val)
    X_test = ss.transform(X_test)

    trainee.history = trainee.model.fit(X_train, y_train, batch_size=256, epochs=100,
                                        validation_data=(X_val, y_val),
                                        callbacks=[trainee.early_stop])
    trainee.save_model()

    with open("ss.pkl", "wb") as file:
        pickle.dump(ss, file)
