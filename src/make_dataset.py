
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
    #Train
    data_train=df[:1500]
    print(data_train)    
    #Valid
    data_valid=df[1500:1668]
    print(data_valid)
    #Test
    data_test=df[1668:]
    print(data_test)
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
