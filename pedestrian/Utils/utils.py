CONFIG = {
    "HOG": {
        "BLOCK_NORM": "L2-Hys",
        "VISUALIZE": True
    },
    "IMAGE": {
        "FINAL_SIZE": (96, 48)
    },
    "SVM": {
        "C": 250
    }
}
import skimage.io
import skimage.transform
from sklearn import svm
from sklearn.externals import joblib
from skimage.feature import hog
from skimage.color import rgb2gray
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random


# ---------------TRATAMIENTO DE IMAGENES--------------------
def Resize(image, finalSize):  # Resize común y silvestre
    return skimage.transform.resize(img, finalSize)


def CropImage(image, posX, posY, width, height):  # Corta la imagen dado un punto, ancho y aleatorio
    return image[posY:posY + height, posX:posX + width]


def ToGrayscale(image):  # Devuelve la imagen en escala de grises
    return rgb2gray(image)


def LoadImageFromPath(path):  # Carga una imagen dado un PATH
    return skimage.io.imread(path)


def NormalizeImage(image, maxValue=False):  # Normaliza la imagen entre 0 y maxValue o 255
    isFloat = type(image[0][0]) is float and image[0][
        0] < 1  # Me fijo si la imagen esta normalizada entre 0 y 1, tomando el primer pixel
    return image / (1 if isFloat else 255)


def PrintImage(image):  # Imprime una imagen usando PyPlot
    plt.figure()
    plt.imshow(image)
    plt.show()


def PrintImages(images, length=-1):  # Imprime varias(todas o N) imagen usando PyPlot.
    if (length == -1):
        length = len(images)
    for i in images:
        plt.figure()
        plt.imshow(i)
        length -= 1
        if (length == 0):
            break
    plt.show()


# ------------------TRATAMIENTO DE PATHS---------------------
def JoinPaths(folder, fil):
    return os.path.join(folder, fil)


# ------------------TRATAMIENTO DE HOGS----------------------
def HogFromImage(image, grayscale=False, resize=False, finalSize=None, normalize=True, maxValue=False,
                 printHogs=False):  # Devuelve el HOG de una imagen
    # Si resize es True, prueba usar finalSize, si finalSize es None, usa la configuración por default definida arriba
    if (grayscale):
        image = ToGrayscale(image)
    if (resize):
        image = Resize(image, finalSize if (finalSize != None) else CONFIG["IMAGE"]["FINAL_SIZE"])
    if (normalize):
        image = NormalizeImage(image, maxValue)
    h, i = hog(image, block_norm=CONFIG["HOG"]["BLOCK_NORM"], transform_sqrt=True, visualise=CONFIG["HOG"]["VISUALIZE"])
    if (printHogs):
        PrintImage(i)
    return h


def GetHogsFromPath(pathToFolder, grayscale=False, resize=False, finalSize=None, subset=-1, normalize=True,
                    maxValue=False, printImages=False, printHogs=False):
    hogs = []
    i = 0
    for dirPath, dirName, fileNames in os.walk(pathToFolder):
        random.shuffle(fileNames)
        for f in fileNames:
            image = LoadImageFromPath(JoinPaths(dirPath, f))
            if (printImages):
                PrintImage(image)
            image_hog = HogFromImage(image, grayscale, resize, finalSize, normalize, maxValue, printHogs)
            hogs.append(image_hog)
            i += 1
            if (i == subset):
                return hogs

    return hogs


def GetHogsFromList(images, grayscale=False, resize=False, finalSize=None, subset=-1, normalize=True, maxValue=False,
                    printImages=False, printHogs=False):
    hogs = []
    i = 0
    random.shuffle(images)
    for image in images:
        if (printImages):
            PrintImage(image)
        image_hog = HogFromImage(image, grayscale, resize, finalSize, normalize, maxValue, printHogs)
        hogs.append(image_hog)
        i += 1
        if (i == subset):
            return hogs
    return hogs


def GetHogsFromPathWithWindow(pathToFolder, window, grayscale=False, resize=False, finalSize=None, subset=-1,
                              normalize=True, maxValue=False, printImages=False, printHogs=False, printSlices=False):
    # Window es una ventana única, con el formato (y,x). i.e.:(96,48)
    return GetHogsFromPathWithWindows(pathToFolder, (window), grayscale, resize, finalSize, subset, normalize, maxValue,
                                      printImages, printHogs, printSlices)


def GetHogsFromPathWithWindows(pathToFolder, windows, grayscale=False, resize=False, finalSize=None, subset=-1,
                               normalize=True, maxValue=False, printImages=False, printHogs=False, printSlices=False):
    # Windows es una lista de ventanas, cada ventana tiene el formato (y,x). i.e.: (96,48)
    hogs = []
    i = 0
    for dirPath, dirName, fileNames in os.walk(pathToFolder):
        random.shuffle(fileNames)
        for f in fileNames:
            image = LoadImageFromPath(JoinPaths(dirPath, f))
            for w in windows:
                height = w[0]
                width = w[1]
                y = 0
                while (y + height < image.shape(0)):
                    x = 0
                    while (x + width < imagen.shape(1)):
                        img_cropped = CropImage(image, x, y, width, height)
                        if (printSlices):
                            PrintImage(img_cropped)
                        image_hog = HogFromImage(img_cropped, grayscale, resize, finalSize, normalize, maxValue,
                                                 printHogs)
                        hogs.append(image_hog)
                        x += width
                    y += height
            if (printImages):
                PrintImage(image)
            i += 1
            if (i == subset):
                return hogs
    return hogs


# -----------------------------TRATAMIENTO DE H5PY ----------------------------
def LoadH5PY(path):
    return h5py.File(path, 'rw')


def CreateDataset(h5pyFile, datasetName, dataset):
    h5pyFile.create_dataset(datasetName, data=dataset)


def GetDataset(h5pyFile, datasetName):
    return h5pyFile[datasetName][:]


# -----------------------------TRATAMIENTO DE SVM -----------------------------
def LoadCheckpoint(path):
    return joblib.load(path)


def SaveCheckpoint(classifier_svm, path):
    joblib.dump(classifier_svm, path)


def CreateLinearSVM():
    return svm.LinearSVC(C=CONFIG["SVM"]["C"])
