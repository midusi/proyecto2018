import skimage.io
from sklearn import svm
from sklearn.externals import joblib
from skimage.feature import hog
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py

DAIMLER_NON_PEDESTRIAN_PATH = '/home/genaro/Descargas/training/Daimler/NonPedestrians_final'
DAIMLER_PEDESTRIAN_PATH = '/home/genaro/Descargas/training/Daimler/Pedestrians'
INRIA_NON_PEDESTRIAN_PATH = '/home/genaro/Descargas/training/INRIA/neg'
INRIA_PEDESTRIAN_PATH = '/home/genaro/Descargas/training/INRIA/pos'

HDF5_PATH = '/home/beto0607/Facu/Pedestrians/Datasets/datasets.h5'  # Path donde se guarda los hogs en HDF5
CHECKPOINT_PATH = '/home/beto0607/Facu/Pedestrians/Datasets/svmCheckpoint.pkl'  # Path donde se guarda el SVM ya entrenado
PREDICT_IMGS_PATH = './imgs/'  # Path de la carpeta de donde sacara imagenes propias para predecir

FINAL_SIZE = [96, 48]
TRAIN = False  # Setear en False cuando se quiera usar el checkpoint y ahorrarse el training
LOAD_FROM_IMGS = False  # Setear en False si se quiere levantar x, y desde HDF5
SUBSET_SIZE = 3000  # Tamaño del dataset a parsear, si se setea en 0 se carga el dataset completo

# Datos de test
TEST_DATA = True  # Testear las imagenes de los path de abajo
TEST_DATA_POS_PATH = '/home/beto0607/Facu/Pedestrians/Datasets/Temp/INRIA/pos'
TEST_DATA_NEG_PATH = '/home/beto0607/Facu/Pedestrians/Datasets/Temp/INRIA/neg'


def print_mulitple(list_of_images):
    """Imprime multiples imagenes en pantalla"""
    for img in list_of_images:
        plt.figure()
        plt.imshow(img)
    plt.show()


def get_hog_from_path(path, grayscale=False):
    """Genera el HOG de todas las imagenes que se encuentran
    dentro de la carpeta pasada por parametro"""
    hogs = []
    size = 0
    for dirpath, dirnames, filenames in os.walk(path):  # Obtengo los nombres de los archivos
        if SUBSET_SIZE:
            random.shuffle(filenames)  # Los pongo en orden aleatorio cuando genero subset
            filenames = filenames[0:SUBSET_SIZE]  # Si fue especificado un tamaño de subset recorto el dataset
        size += len(filenames)  # Cuento la cantidad de archivos que voy a generar el HOG
        for filename in filenames:
            img_path = os.path.join(dirpath, filename)
            img = skimage.io.imread(img_path)  # Cargo la imagen
            if grayscale:
                img = grayscaled_img(img)
            img_hog = hog(img, block_norm='L2-Hys', transform_sqrt=True)
            hogs.append(img_hog)

    return hogs, size  # Devuelvo lista de hogs y el tamaño total de elementos generados


def load_training_data():
    """Carga los datos de training desde los 4 paths de training
    seteados arriba"""
    # Leo los de Daimler negativos
    daimler_neg_hogs, size = get_hog_from_path(DAIMLER_NON_PEDESTRIAN_PATH)
    x = daimler_neg_hogs  # Arreglo que almacenara los HOGS de cada imagen
    y = np.zeros(size)

    # Leo los de Daimler positivos
    daimler_pos_hogs, size = get_hog_from_path(DAIMLER_PEDESTRIAN_PATH)
    x += daimler_pos_hogs
    y = np.append(y, np.ones(size))

    # Leo los de INRIA negativos
    # NOTA: escalo a gris estos negativos y positivos porque INRIA viene en RGB
    inria_neg_hogs, size = get_hog_from_path(INRIA_NON_PEDESTRIAN_PATH, grayscale=True)
    x += inria_neg_hogs
    y = np.append(y, np.zeros(size))

    # Leo los de INRIA positivos
    inria_pos_hogs, size = get_hog_from_path(INRIA_PEDESTRIAN_PATH, grayscale=True)
    x += inria_pos_hogs
    y = np.append(y, np.ones(size))

    return x, y


def resize(img):
    """Devuelve la imagen con el tamaño modificado"""
    return skimage.transform.resize(img, FINAL_SIZE)


def print_image(img):
    """No va a funcionar correctamente hasta que no se normalice la imagen a una
    escala aceptable, ya que el formato pgm va de 0 a 4096"""
    plt.imshow(img)
    plt.show()


def grayscaled_img(img):
    """Devuelve la imagen en escala de grises"""
    return np.mean(img, axis=2)


def load_predict_img(img_name):
    """Solo carga una imagen de prediccion personalizada"""
    img_path = os.path.join(PREDICT_IMGS_PATH, img_name)
    img = skimage.io.imread(img_path)  # Cargo la imagen
    img = grayscaled_img(img)
    img = resize(img)
    return hog(img, block_norm='L2-Hys', transform_sqrt=True)


def get_predict_data():
    """Obtiene las muestras de la carpeta imgs para ver si el SVM predice bien
    La data esta fixeada para probar rapidamente"""
    hogs = []  # Lista de hogs
    expected = [1, 0, 0, 1, 1, 1]  # Resultado que se espera

    hogs.append(load_predict_img('dicaprio.jpg'))
    hogs.append(load_predict_img('flor.jpg'))
    hogs.append(load_predict_img('paisaje.jpg'))
    hogs.append(load_predict_img('peaton1.jpg'))
    hogs.append(load_predict_img('peaton2.jpg'))
    hogs.append(load_predict_img('peaton3.jpg'))

    return hogs, expected


def load_test_data():
    """Toma los datos de test usando los paths seteados anteriormente"""
    # Leo los test de INRIA negativos
    inria_neg_hogs, size = get_hog_from_path(TEST_DATA_NEG_PATH, grayscale=True)
    x = inria_neg_hogs
    y = np.zeros(size)

    # Leo los test de INRIA positivos
    inria_pos_hogs, size = get_hog_from_path(TEST_DATA_POS_PATH, grayscale=True)
    x += inria_pos_hogs
    y = np.append(y, np.ones(size))

    return x, y


def main():
    if not HDF5_PATH or not CHECKPOINT_PATH:
        print('No se ha seteado el path de HDF5 o del checkpoint!')
        return

    if TRAIN:
        if LOAD_FROM_IMGS:
            x, y = load_training_data()  # Obtengo la data de entrenamiento (previamente corri los scripts de carga)

            # Guardo x, y en HDF5!
            h5f = h5py.File(HDF5_PATH, 'w')
            h5f.create_dataset('dataset_x', data=x)
            h5f.create_dataset('dataset_y', data=y)
        else:
            h5f = h5py.File(HDF5_PATH, 'r')
            x = h5f['dataset_x'][:]
            y = h5f['dataset_y'][:]

        h5f.close()  # Cierro el archivo HDF5

        # Genero y entreno el SVM
        classifier_svm = svm.LinearSVC(C=200)
        classifier_svm.fit(x, y)

        joblib.dump(classifier_svm, CHECKPOINT_PATH)
    else:
        classifier_svm = joblib.load(CHECKPOINT_PATH)

    # Cargo el set de prediccion
    if TEST_DATA:
        predict_data, expected = load_test_data()  # Si se quiere usar el dataset de tests seteados...
    else:
        predict_data, expected = get_predict_data()  # Si se quiere utilizar la data de test fija

    predictions = classifier_svm.predict(predict_data)

    success = 0
    # Imprimo de forma amigable los resultados de la prediccion
    for prediction, expected_value in zip(predictions, expected):
        if prediction == 1:
            value = 'Es peaton.'
        else:
            value = 'No es peaton.'

        expected_str = 'Se esperaba '
        if expected_value == 1:
            expected_str += 'peaton'
        else:
            expected_str += 'no peaton'

        if prediction == expected_value:
            success += 1
            correct = '✔'
        else:
            correct = '✘'

        print(value, expected_str, correct)

    total_predictions = len(predictions)

    print("------------------------")
    print("{} / {} correctos. {}% de precision".format(success, total_predictions, (100 * success) / total_predictions))


if __name__ == '__main__':
    main()
