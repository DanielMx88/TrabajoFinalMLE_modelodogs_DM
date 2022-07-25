# Código de Evaluación - Modelo de reconocimiento de imagenes de perros
############################################################################

import cv2
import os
import random
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import re
import pandas as pd

#to work Convolutional Network
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers, Sequential

#Input of the network
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', 
                 input_shape=(64,64,3)))
#padding='same|valid->zero padding, no padding'
model.add(Activation('relu'))

#Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))

#Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#Output layer, this depends on the number of classes,
#para este caso se uso 10 clases
model.add(Dense(10, activation='sigmoid'))
model.summary()
#Several hyper-parameters
model.compile(optimizers.RMSprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy", metrics=["accuracy"])

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    #df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df

# Realizamos la eleccion de imagenes de datos
def eval_model(data_train, data_valid, data_test):
    columns=["chihuahua", "shihtzu", "beagle", "basset", "lakeland", "bostonbull", "tibetan", "labrador", "german", "siberian"]
    #Rescale pixel values (0-1 values)
    train_datagen=ImageDataGenerator(rescale=1./255.)
    test_datagen=ImageDataGenerator(rescale=1./255.)
    #A generator creates an optimal structure in terms of memory
    #to load the images.
    #The generator also allows us to change the size of the images,
    #in this case our  images will be resized to 64X64.
    #Train
    train_generator=train_datagen.flow_from_dataframe(
    dataframe=data_train,
    directory="../data/raw/images",
    x_col="Filenames",
    y_col=columns, #array previously created
    batch_size=16,
    seed=42,
    shuffle=True,
    class_mode="raw", #numpy array of values in y_col column(s)
    target_size=(64,64))
    #Valid
    valid_generator=test_datagen.flow_from_dataframe(
    dataframe=data_valid,
    directory="../data/raw/images",
    x_col="Filenames",
    y_col=columns,
    batch_size=8,
    seed=42,
    shuffle=True,
    class_mode="raw", #Clases 
    target_size=(64,64))
    #Test
    test_generator=test_datagen.flow_from_dataframe(
    dataframe=data_test,
    directory="../data/raw/images",
    x_col="Filenames",
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(64,64)) 
    print('Seleccion de imagenes completa')    
    
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size #1500/16=93.75
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size #168/8=21
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size    #168/1=168
    
    n_epochs=5
    print("Evaluacion del modelo con : " + str(n_epochs))
    history=model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=n_epochs
    )
    
# Validación desde el inicio
def main():
    data_train = read_file_csv('dog_train.csv')
    data_valid = read_file_csv('dog_valid.csv')
    data_test = read_file_csv('dog_test.csv')
    df = eval_model(data_train, data_valid, data_test)
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()