# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:24:35 2018

@author: busca
"""
import skimage.io
import skimage.transform
from sklearn import svm
from sklearn.externals import joblib
from skimage.feature import hog
import os
import numpy as np
import h5py

workspace, fl = os.path.split(os.path.realpath(__file__))
trainFalse = os.path.join(workspace, 'train\\neg\\')
trainPos = os.path.join(workspace, 'train\\pos\\')
testFalse = os.path.join(workspace, 'test\\neg\\')
testPos = os.path.join(workspace, 'test\\pos\\')

hdf5Train = os.path.join(workspace, 'train.h5')  # Path donde se guarda los hogs en HDF5
checkpoint = os.path.join(workspace, 'svmCheckpoint.pkl') # Path donde se guarda el SVM ya entrenado
testPath = os.path.join(workspace, 'Test\\') # Path de la carpeta de donde sacara imagenes propias para predecir

finalSize = [96, 48]
TRAIN = False  # Setear en False cuando se quiera usar el checkpoint y ahorrarse el training
LOAD_FROM_IMGS = False  # Setear en False si se quiere levantar x, y desde HDF5
TEST = False # Setear en False para testear sobre el dataset de entrenamiento


def grayscaled_img(img):
    return np.mean(img, axis=2)

def getHogs(path, grayscale=False):
    """Genera el HOG de todas las imagenes que se encuentran
    dentro de la carpeta pasada por parametro"""
    hogs = []
    size = 0
    for dirpath, dirnames, filenames in os.walk(path):  # Obtiene los nombres de los archivos
        size += len(filenames)  # Cuenta la cantidad de archivos que voy a generar el HOG
        for filename in filenames:
            img_path = os.path.join(dirpath, filename)
            img = skimage.io.imread(img_path)  # Carga la imagen
            if grayscale:
                img = grayscaled_img(img)
            img_hog = hog(img,block_norm='L2-Hys',transform_sqrt=True,visualise=False)
            #img_hog = hog(img, pixels_per_cell=(5, 5))
            hogs.append(img_hog)

    return hogs, size  # Devuelve lista de hogs y el tamaño total de elementos generados
	
def loadHogsHDF5(path):
	"""Carga los hogs del archivo hdf5"""
	h5f = h5py.File(path, 'r')
	x = h5f['dataset_x'][:]
	y = h5f['dataset_y'][:]
	h5f.close()  # Cierra el archivo HDF5
	return x,y

def main():
    if not hdf5Train or not checkpoint:
        print('No se ha seteado el path de HDF5 o del checkpoint!')
        return

    if TRAIN:
        if LOAD_FROM_IMGS:
            # Calcula los hogs del dataset Train negativos
            negHogs, size = getHogs(trainFalse, grayscale=True)
            xTrain = negHogs
            yTrain = np.zeros(size)

            # Calcula los hogs del dataset Train positivos
            posHogs, size = getHogs(trainPos, grayscale=True)
            xTrain += posHogs
            yTrain = np.append(yTrain, np.ones(size))
            
            # Guarda xTrain, yTrain en HDF5!
            h5f = h5py.File(hdf5Train, 'w')
            h5f.create_dataset('dataset_x', data=xTrain)
            h5f.create_dataset('dataset_y', data=yTrain)
            h5f.close()  # Cierra el archivo HDF5
        else:
            xTrain,yTrain = loadHogsHDF5(hdf5Train)

        # Entrena el SVM
        classifierSvm = svm.SVC()
        classifierSvm.fit(xTrain, yTrain)

        joblib.dump(classifierSvm, checkpoint)
    else:
        classifierSvm = joblib.load(checkpoint)
        xTrain,yTrain = loadHogsHDF5(hdf5Train)
		
    if TEST:
        # Calcula los hogs del dataset Test negativos
        negHogs, size = getHogs(testFalse, grayscale=True)
        xTest = negHogs
        yTest = np.zeros(size)
        
        # Calcula los hogs del dataset Test positivos
        posHogs, size = getHogs(testPos, grayscale=True)
        xTest += posHogs
        yTest = np.append(yTest, np.ones(size))
    else:
        xTest = xTrain
        yTest = yTrain
        
    #Carga los hogs de testing si no fueron cargados
    predictions = classifierSvm.predict(xTest)
    
    positives = 0
    negatives = 0
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    i = 0
    fn = []
    fp = []

    for prediction, expected_value in zip(predictions, yTest):
        i += 1
        if expected_value == 1:
            positives += 1
            if prediction == 1:
                truePositive += 1
            else:
                falseNegative += 1
                fn.append(i)
        else:
            negatives += 1
            if prediction == 0:
                trueNegative += 1
            else:
                falsePositive += 1
                fp.append(i)
	
    # Imprime los resultados de la prediccion
    print("Accuracy: ", truePositive, '/', truePositive+falsePositive,' = ', truePositive/(truePositive+falsePositive))
    print("Recall: ", truePositive, '/', positives,' = ', truePositive/positives)
    print("Falsos Positivos: ", falsePositive)
    print("Falsos Negativos: ", falseNegative)
	
    print("Falsos Positivos: ")
    for img in fp:
        print(img)
	
    print("Falsos Negativos: ")
    for img in fn:
        print(img)
	

if __name__ == '__main__':
    main()