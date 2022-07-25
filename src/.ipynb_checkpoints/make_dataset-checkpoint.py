
# Script de Preparación de Datos
###################################

import cv2
import os
import random
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import re
import pandas as pd


# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename))#.set_index('ID')
    print(filename, ' cargado correctamente')
    return df


# Realizamos la eleccion de imagenes de datos
def data_generation(df):
    columns=["chihuahua", "shihtzu", "beagle", "basset", "lakeland", "bostonbull", "tibetan", "labrador", "german", "siberian"]
    #Rescale pixel values (0-1 values)
    train_datagen=ImageDataGenerator(rescale=1./255.)
    test_datagen=ImageDataGenerator(rescale=1./255.)
    #A generator creates an optimal structure in terms of memory
    #to load the images.
    #The generator also allows us to change the size of the images,
    #in this case our  images will be resized to 64X64.
    train_generator=train_datagen.flow_from_dataframe(
    dataframe=df[:1500],
    directory="images",
    x_col="Filenames",
    y_col=columns, #array previously created
    batch_size=16,
    seed=42,
    shuffle=True,
    class_mode="raw", #numpy array of values in y_col column(s)
    target_size=(64,64))
    data_train=df[:1500]
    print(data_train)    
    print(train_generator)
    valid_generator=test_datagen.flow_from_dataframe(
    dataframe=df[1500:1668],
    directory="images",
    x_col="Filenames",
    y_col=columns,
    batch_size=8,
    seed=42,
    shuffle=True,
    class_mode="raw", #Clases 
    target_size=(64,64))
    data_valid=df[1500:1668]
    print(data_valid)    
    print(valid_generator)
    test_generator=test_datagen.flow_from_dataframe(
    dataframe=df[1668:],
    directory="images",
    x_col="Filenames",
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(64,64))
    data_test=df[1668:]
    print(data_test)    
    print(test_generator)    
    print('Generación de datos completa')    
    return data_train, data_valid, data_test


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('labels_1.csv')
    tdf_train, tdf_valid, tdf_test = data_generation(df1)
    #print(tdf_train)   
    #print(tdf_valid)   
    #print(tdf_test)   
    data_exporting(tdf_train, ['Filenames','chihuahua', 'shihtzu', 'beagle', 'basset', 'lakeland', 'bostonbull', 'tibetan', 'labrador', 'german', 'siberian'],'dog_train.csv')
    # Matriz de Validación
    data_exporting(tdf_valid, ['Filenames','chihuahua', 'shihtzu', 'beagle', 'basset', 'lakeland', 'bostonbull', 'tibetan', 'labrador', 'german', 'siberian'],'dog_valid.csv')
    # Matriz de Test
    data_exporting(tdf_test, ['Filenames','chihuahua', 'shihtzu', 'beagle', 'basset', 'lakeland', 'bostonbull', 'tibetan', 'labrador', 'german', 'siberian'],'dog_test.csv')
    
if __name__ == "__main__":
    main()
