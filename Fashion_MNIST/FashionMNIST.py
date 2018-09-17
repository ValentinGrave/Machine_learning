from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras.models as mod
import keras.callbacks as call
import matplotlib.pyplot as plt
import os.path as path


# DEVELOPPEMENT DU CNN
# nom du modèle à charger/sauvegarder
modelName = "Models saved\FashionMNIST_model.h5"

batch_size = 128
num_classes = 10
epochs = 3

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'train_samples')
print(x_test.shape[0], 'test_samples')


# conversion des vecteurs de classes en matrices de classes binaires
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

modelExist = path.exists(modelName)
if(modelExist == False):
    print("DON T EXIST")
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

else:
    print("EXIST")
    # chargement du modèle contenu 'modelName'
    model = mod.load_model(modelName)

model.summary()
# Creation of a callback 
checkpoint = call.ModelCheckpoint(modelName, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# save the model after each epoch ('period' parameter)
#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpoint])

x_test_results = model.predict(x_test)

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

# sauvegarde du modèle
model.save(modelName)


# AFFICHAGE DES RESULTATS
pic2show = 25

for i in range(pic2show):
    tableau = plt.subplot(5, 5 , i+1)
    plt.imshow(x_train[i].reshape(28,28))
    plt.gray()
    tableau.get_xaxis().set_visible(False)
    tableau.get_yaxis().set_visible(False)
    
    plt.show()