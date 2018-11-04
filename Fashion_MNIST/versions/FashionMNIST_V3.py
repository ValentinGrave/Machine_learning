### IMPORTATION OF LIBRARIES
from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import backend as K
import keras.models as mod
import keras.callbacks as call
import matplotlib.pyplot as plt
import os.path as path
#from keras_tqdm import TQDMCallback

### CONSIGNES
# Nombre EPOCHS désiré = minimum possible
# Nombre PARAMETERS désiré = minimum possible (x < 500.000)


### CNN CREATION
# name of the model to save/load
modelName = ".\Models saved\FashionMNIST_model.h5"
# name of the picture of the architecture of the model to save
pictureName = ".\Models saved\FashionMNIST_model.png"
# name of the log directory
logName = ".\Models saved\log"


### CREATIONS OF VARIABLES
# size of a batch
batch_size = 256
# number of classes in the dataset
num_classes = 10
# number of epochs
epochs = 25
# dimensions of every "piece of data" in the dataset
img_rows, img_cols = 28, 28
# definition of the training/validation data/labels
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


### PREPARATION OF THE DATA
# definition of the channels order
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# converting the values type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# "standardisation" of the values
x_train /= 255
x_test /= 255
# printing of the shapes
print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'train_samples')
print(x_test.shape[0], 'test_samples')
# conversion of the classes vectors in binary classes matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


### DEFINITION OF THE MODEL ARCHITECTURE
modelExist = path.exists(modelName)
# If the model doesn't exist : Creation of a model
if (modelExist == False):
    print("DOESN'T EXIST")
    model = Sequential()
# 1st convolutional layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
# 2nd convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # reduction of the size of each piece of data
    model.add(MaxPooling2D(pool_size=(2, 2)))
# 3rd convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# Serialization of the data
    model.add(Flatten())
# 1st Dense layer
    model.add(Dense(64, activation='relu'))
    # reduction of "over-fitting"
    model.add(Dropout(0.20))
# Output Dense layer
    model.add(Dense(num_classes, activation='softmax'))
# Compilation of all the layers (= creation of the model)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# If the model exists: Loading of the existing model
else:
    print("EXIST")
    # loading of the model contained in 'modelName'
    model = mod.load_model(modelName)
# list of the layers in the model
model.summary()


### CREATION OF CALLBACKS
# callback saving the model after each epoch ('period' parameter = 1)
checkpoint = call.ModelCheckpoint(modelName, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# callback changing the learning rate (LR) when the learning is on a "plateau" (learning not evolving)
reduceLR = call.ReduceLROnPlateau(monitor='val_acc', factor=0.25, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.001)
# callback allowing to visualize dynamics graphs of training metrics, testing metrics, activation histograms, ...
tensorBoard = call.TensorBoard(log_dir=logName, histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=True)


### FITS THE MODEL ON BATCHES MODIFIED WITH REAL-TIME DATA AUGMENTATION
# creation of a ImageDataGenerator object 'dataGenerator' (its parameters represent the possible variation of the data it will modify)
dataGenerator = ImageDataGenerator(rotation_range=0, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
# modification of 'x_train' data through 'dataGenerator'
dataGenerator.fit(x_train)
# fitting of the model to the augmented data
model.fit_generator(dataGenerator.flow(x_train, y_train, batch_size=batch_size, shuffle=True), steps_per_epoch=len(x_train) / 256, epochs=epochs, callbacks=[checkpoint, reduceLR, tensorBoard], validation_data=(x_test, y_test))


### VALIDATION OF THE MODEL
# predictions
x_test_results = model.predict(x_test)
# evaluation of the model
score = model.evaluate(x_test, y_test, verbose=1)
# printing of the results
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])


### SAVINGS OF THE MODEL, THE WEIGHTS (AND A PICTURE OF THE MODEL ARCHITECTURE)
# Save the model (weights)
model.save(modelName)
# Save the model (picture)
plot_model(model, to_file=pictureName)


### PLOTTING OF SOME SAMPLES OF THE DATASET
# number of pictures to plot
pic2show = 25
# plotting
for i in range(pic2show):
    tableau = plt.subplot(5, 5, i + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    tableau.get_xaxis().set_visible(False)
    tableau.get_yaxis().set_visible(False)
# display
plt.show()