'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

from keras.callbacks import LambdaCallback
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import resample
from matplotlib import pyplot as plt


batch_size = 128
num_classes = 10
epochs = 200
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


import pickle
filename = 'pred15_finalized_reg_model.sav'
prediction_model = pickle.load(open(filename, 'rb'))


num_callback = 6
l = np.array([])
temp_a = np.array([])

weights_log = np.array([])

weight_hist_length = 5


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train,y_train=resample(x_train,y_train, n_samples=2500, random_state=1)
x_test,y_test=resample(x_test,y_test, n_samples=500, random_state=1)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


def weight_prediciton():
    global l
    if l.shape[0] == 0:
        l = np.append(l, model.get_weights())
    else:
        l = np.vstack([l, model.get_weights()])


    if l.shape[0] == weight_hist_length:
        temp_weight_pool = np.array([])
        for i in range(weight_hist_length):
            temp_sub_weight_pool = np.array([])
            for j in range(l.shape[1]):
                temp_sub_weight_pool=np.append(temp_sub_weight_pool,l[i][j].flatten())

            if i==0:
                temp_weight_pool = np.append(temp_weight_pool, temp_sub_weight_pool)
            else:
                temp_weight_pool = np.vstack([temp_weight_pool, temp_sub_weight_pool])

        temp_weight_pool = np.transpose(temp_weight_pool)
        #w = np.sqrt(sum(temp_weight_pool ** 2))
        #x_norm2 = temp_weight_pool / w
        #x_norm2 = np.transpose(x_norm2)

        tempstd = np.std(temp_weight_pool, 1)
        #idx = np.where(tempstd > np.mean(tempstd))
        idx = np.where(tempstd > 2*np.mean(tempstd))
        #print(np.mean(tempstd))
        predicted_weight = temp_sub_weight_pool
        predicted_weight[idx] = prediction_model.predict(temp_weight_pool[idx])

        temp_n = 0
        for j in range(l.shape[1]):
            tmp_shape=l[0][j].shape
            tmp_length=l[0][j].flatten().shape[0]
            l[0][j]=np.reshape(predicted_weight[temp_n:temp_n+tmp_length],tmp_shape)
            temp_n=temp_n+tmp_length

        model.set_weights(l[0])
        l = np.array([])





update_weights = LambdaCallback(on_epoch_end=lambda batch, logs: weight_prediciton())

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    hist = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[update_weights])

else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    hist = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4,
                        callbacks = [update_weights])

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

import pickle

# save:
f = open('epochend_epoch200_resample2500_cifar_doc_agu_pred15_std01_history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()

l = np.array(l)