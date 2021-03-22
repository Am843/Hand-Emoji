import numpy as np
import cv2
from keras import layers
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPool2D, Dropout, GlobalAveragePooling2D, GlobalAvgPool2D
from keras.utils import np_utils
import pandas as pd
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K


data = pd.read_csv('training data.csv')
# df = data.sample(frac=1).reset_index(drop=True)
print(data.head())
print(data.shape)

df = np.array(data)
np.random.shuffle(df)
X = df[:, 1:2501]/255
X = np.asarray(X).astype(np.float32)
Y = df[:, 0]
X_train = X[0:12000, :]
X_test = X[12000:132001, :]
labels = np.unique(Y)
image_x = 50
image_y = 50


binencoder = LabelBinarizer()
Y = binencoder.fit_transform(Y)
print(Y.shape)
Y_train = Y[0:12000, :]
Y_test = Y[12000:132001, :]
train_y = Y_train.reshape(Y_train.shape[0], Y_train.shape[1])
train_X = X_train.reshape(X_train.shape[0], image_x, image_y, 1)
test_X = X_test.reshape(X_test.shape[0], image_x, image_y, 1)
test_y = Y_test.reshape(Y_test.shape[0], Y_test.shape[1])
print(test_y.shape)
print(test_X.shape)
print(train_y.shape)
print(train_X.shape)


def keras_model(image_x, image_y):
    num_of_classes = 11
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), input_shape=(image_x, image_y, 1), activation="sigmoid"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(Conv2D(64,(5,5), activation='sigmoid'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(Conv2D(128,(5,5), activation='sigmoid'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(Flatten())
    model.add(Dense(num_of_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "emoji.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint1]
    return model,  callback_list


model, callbacks_list = keras_model(image_x,image_y)
model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=2, batch_size=64, callbacks=callbacks_list)
score = model.evaluate(test_X, test_y, verbose=0)
print('CNN Error : %.2f%%' %(100- score[1]*100))
print(model.summary())
model.save('emoji.h5')

print(labels)