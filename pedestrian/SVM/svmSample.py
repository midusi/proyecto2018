import skimage.io
import skimage.transform
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.externals import joblib
from skimage.feature import hog
from skimage.color import rgb2gray
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py

DAIMLER_NON_PEDESTRIAN_PATH = '/home/genaro/Descargas/training/Daimler/NonPedestrians_final'
DAIMLER_PEDESTRIAN_PATH = '/home/genaro/Descargas/training/Daimler/Pedestrians'
INRIA_NON_PEDESTRIAN_PATH = '/home/genaro/Descargas/training/INRIA/neg'
INRIA_PEDESTRIAN_PATH = '/home/genaro/Descargas/training/INRIA/pos'

HDF5_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/datasets.h5'  # Path donde se guarda los hogs en HDF5
CHECKPOINT_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl' # Path donde se guarda el SVM ya entrenado
PREDICT_IMGS_PATH = './imgs/'  # Path de la carpeta de donde sacara imagenes propias para predecir

# Tamanios a los que se va a hacer resize de las imagenes
FINAL_SIZES = [
    [128, 64],
    [96, 48]
]
TRAIN = False  # Setear en False cuando se quiera usar el checkpoint y ahorrarse el training
LOAD_FROM_IMGS = False  # Setear en False si se quiere levantar x, y desde HDF5
SUBSET_SIZE = 4500  # Tamaño del dataset a parsear, si se setea en 0 se carga el dataset completo
VISUALIZE_IMG = False  # Mostrar las imagenes que van a entrar al HOG()

# Datos de test
TEST_DATA = True  # Testear las imagenes de los path de abajo
USE_TRAINING_AS_TEST_DATA = False  # Con True usa los datos de Training como test. False para usar las rutas de abajo

# Si USE_TRAINING_AS_TEST_DATA esta en True estos parametros se ignoran
TEST_DATA_POS_PATH = '/home/genaro/Descargas/PedCut2013_SegmentationDataset/data/testData/left_images'
TEST_DATA_NEG_PATH = '/home/genaro/Descargas/PedCut2013_SegmentationDataset/data/testData/vacia'


def print_mulitple(list_of_images):
    """Imprime multiples imagenes en pantalla"""
    for img in list_of_images:
        plt.figure()
        plt.imshow(img)
    plt.show()


def get_hog_from_path(path, must_grayscale=False, must_resize=True, must_normalize=True):
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
            if VISUALIZE_IMG:
                print("Antes del preprocesamiento")
                print(img_path)
                print(img)
                print("Shape", img.shape)
                flatten = img.flatten()
                print("Min", min(flatten), "Max", max(flatten))
                print_image(img)

            # Si pidieron hacer resize...
            if must_resize:
                # is_max_size = True
                img_hog = []
                for img_size in FINAL_SIZES:

                    img = resize(img, img_size)
                    img_hog_aux = get_img_hog(img, must_grayscale=must_grayscale, must_normalize=must_normalize)
                    img_hog = np.concatenate([img_hog, img_hog_aux])
                    # if is_max_size:
                    #     is_max_size = False
                    # else:
                    #     img_hog = np.pad(img_hog, (0, 3564), 'constant', constant_values=0)
                    # hogs.append(img_hog)
            else:
                img_hog = get_img_hog(img, must_grayscale=must_grayscale, must_normalize=must_normalize)
            hogs.append(img_hog)

    # print("hogs --> ")
    # hogs = np.array(hogs)
    # print(hogs[0].shape)
    # print(hogs.shape)
    # exit(0)
    return hogs, size  # Devuelvo lista de hogs y el tamaño total de elementos generados


def get_img_hog(img, must_grayscale=True, must_normalize=True):
    """Obtiene el HOG de una imagen con algun preprocesamiento
    solicitado"""
    if must_grayscale:
        img = grayscaled_img(img)

    # Normalizo la imagen
    if must_normalize:
        img = normalize_img(img)

    if VISUALIZE_IMG:
        print("Despues del preprocesamiento")
        print(img)
        print("Shape", img.shape)
        flatten = img.flatten()
        print("Min", min(flatten), "Max", max(flatten))
        print_image(img)

    return hog(img, block_norm='L2-Hys', transform_sqrt=True)


def load_training_data():
    """Carga los datos de training desde los 4 paths de training
    seteados arriba"""
    positives = negatives = 0  # Para dar informacion de cantidad de ejemplos positivos y negativos
    # Leo los de Daimler negativos
    daimler_neg_hogs, size = get_hog_from_path(DAIMLER_NON_PEDESTRIAN_PATH, must_grayscale=True)
    x = daimler_neg_hogs  # Arreglo que almacenara los HOGS de cada imagen
    y = np.zeros(size)

    print("Daimler negativos --> {}".format(size))
    negatives += size

    # Leo los de Daimler positivos
    daimler_pos_hogs, size = get_hog_from_path(DAIMLER_PEDESTRIAN_PATH, must_grayscale=True)
    x += daimler_pos_hogs
    y = np.append(y, np.ones(size))

    print("Daimler positivos --> {}".format(size))
    positives += size

    # Leo los de INRIA negativos
    # NOTA: escalo a gris estos negativos y positivos porque INRIA viene en RGB
    inria_neg_hogs, size = get_hog_from_path(INRIA_NON_PEDESTRIAN_PATH, must_grayscale=True)
    x += inria_neg_hogs
    y = np.append(y, np.zeros(size))

    print("INRIA negativos --> {}".format(size))
    negatives += size

    # Leo los de INRIA positivos
    inria_pos_hogs, size = get_hog_from_path(INRIA_PEDESTRIAN_PATH, must_grayscale=True)
    x += inria_pos_hogs
    y = np.append(y, np.ones(size))

    print("INRIA positivos --> {}".format(size))
    positives += size

    print("Entrenando en total con {} ejemplos positivos y {} ejemplos negativos".format(positives, negatives))

    # print(x)
    x = np.array(x)
    # print(x)
    print("------------------------")
    print("x shape --> ", x.shape)
    print("------------------------")
    print("y shape", y.shape, "| y len --> ", len(y))
    print("------------------------")
    # exit(0)
    return x, y


def resize(img, size):
    """Devuelve la imagen con el tamaño modificado"""
    return skimage.transform.resize(img, size)


def print_image(img):
    """No va a funcionar correctamente hasta que no se normalice la imagen a una
    escala aceptable, ya que el formato pgm va de 0 a 4096"""
    plt.imshow(img, cmap="gray")
    plt.show()


def normalize_img(img):
    """Normaliza la imagen con el maximo valor usando sklearn"""
    return normalize(img, 'max')


def grayscaled_img(img):
    """Devuelve la imagen en escala de grises"""
    return rgb2gray(img)


def load_predict_img(img_name):
    """Solo carga una imagen de prediccion personalizada"""
    img_path = os.path.join(PREDICT_IMGS_PATH, img_name)
    img = skimage.io.imread(img_path)  # Cargo la imagen
    img = grayscaled_img(img)
    img = resize(img, FINAL_SIZES[0])
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


def get_hdf5_datasets():
    """Devuelve los datos de training guardados en disco"""
    h5f = h5py.File(HDF5_PATH, 'r')
    x, y = h5f['dataset_x'][:], h5f['dataset_y'][:]
    h5f.close()  # Cierro el archivo HDF5
    return x, y


def load_test_data():
    """Toma los datos de test usando los paths seteados anteriormente"""
    # Leo los test de INRIA negativos
    inria_neg_hogs, size = get_hog_from_path(TEST_DATA_NEG_PATH, must_grayscale=True)
    x = inria_neg_hogs
    y = np.zeros(size)

    # Leo los test de INRIA positivos
    inria_pos_hogs, size = get_hog_from_path(TEST_DATA_POS_PATH, must_grayscale=True)
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
            h5f.close()  # Cierro el archivo HDF5
        else:
            x, y = get_hdf5_datasets()

        # Genero y entreno el SVM
        classifier_svm = svm.LinearSVC(C=250)
        classifier_svm.fit(x, y)

        joblib.dump(classifier_svm, CHECKPOINT_PATH)
    else:
        classifier_svm = joblib.load(CHECKPOINT_PATH)

    # Cargo el set de prediccion
    if TEST_DATA:
        if USE_TRAINING_AS_TEST_DATA:
            predict_data, expected = get_hdf5_datasets()
        else:
            predict_data, expected = load_test_data()  # Si se quiere usar el dataset de tests seteados...
    else:
        predict_data, expected = get_predict_data()  # Si se quiere utilizar la data de test fija

    # Imprimo los datos del clasificador para los experimentos
    print(classifier_svm)

    # Genero la prediccion
    predictions = classifier_svm.predict(predict_data)

    success = error = 0
    total_pedestrian = pedrestrian_predected = pedrestrian_success = 0
    false_positives = false_negatives = 0
    # Imprimo de forma amigable los resultados de la prediccion
    # y saco precision y recall
    i = 0
    for prediction, expected_value in zip(predictions, expected):
        # Para sacar la precision y exhaustividad (recall)
        if expected_value == 1:  # Si es un peaton realmente...
            total_pedestrian += 1

        if prediction == 1:  # Si el SVM dijo que era un peaton...
            pedrestrian_predected += 1
            if expected_value == 1:  # Si era un peaton y fue bien reconocido...
                pedrestrian_success += 1
            else:
                false_positives += 1
        else:
            if expected_value == 1:
                false_negatives += 1

        if prediction == expected_value:
            success += 1
        else:
            error += 1

        i += 1

    total_predictions = len(predictions)

    print("------------------------")
    print("Positivos y negativos acertados --> {} / {} correctos. {}% de aciertos".format(
        success, total_predictions, (100 * success) / total_predictions
    ))
    print("Precision --> {} / {} = {}".format(pedrestrian_success, pedrestrian_predected, pedrestrian_success / pedrestrian_predected))
    print("Recall --> {} / {} = {}".format(pedrestrian_success, total_pedestrian, pedrestrian_success / total_pedestrian))
    print("Falsos positivos --> {} | Falsos negativos --> {}".format(false_positives, false_negatives))


if __name__ == '__main__':
    main()
