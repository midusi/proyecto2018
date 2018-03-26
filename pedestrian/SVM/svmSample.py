import skimage.io
from sklearn import svm
from sklearn.externals import joblib
from skimage.feature import hog
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py

daimler_non_pedestrian_URL = '/home/genaro/Descargas/training/Daimler/NonPedestrians_final'
daimler_pedestrian_URL = '/home/genaro/Descargas/training/Daimler/Pedestrians'
INRIA_non_pedestrian_URL = '/home/genaro/Descargas/training/INRIA/neg'
INRIA_pedestrian_URL = '/home/genaro/Descargas/training/INRIA/pos'

hdf5_URL = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/datasets.h5'  # Path donde se guarda los hogs en HDF5
checkpoint_URL = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl'  # Path donde se guarda el SVM ya entrenado
predict_imgs_path = './imgs/'  # Path de la carpeta de donde sacara imagenes propias para predecir

final_size = [96, 48]
TRAIN = True  # Setear en False cuando se quiera usar el checkpoint y ahorrarse el training
LOAD_FROM_IMGS = True  # Setear en False si se quiere levantar x, y desde HDF5
subset_size = 300  # Tamaño del dataset a parsear, si se setea en 0 se carga el dataset completo


def print_mulitple(list_of_images):
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
        if subset_size:
            random.shuffle(filenames)  # Los pongo en orden aleatorio cuando genero subset
            filenames = filenames[0:subset_size]  # Si fue especificado un tamaño de subset recorto el dataset
        size += len(filenames)  # Cuento la cantidad de archivos que voy a generar el HOG
        for filename in filenames:
            img_path = os.path.join(dirpath, filename)
            img = skimage.io.imread(img_path)  # Cargo la imagen
            if grayscale:
                img = grayscaled_img(img)
            img_hog = hog(img, pixels_per_cell=(2, 2))
            hogs.append(img_hog)

    return hogs, size  # Devuelvo lista de hogs y el tamaño total de elementos generados


def load_training_data():

    # Leo los de Daimler negativos
    daimler_neg_hogs, size = get_hog_from_path(daimler_non_pedestrian_URL)
    x = daimler_neg_hogs  # Arreglo que almacenara los HOGS de cada imagen
    y = np.zeros(size)

    # Leo los de Daimler positivos
    daimler_pos_hogs, size = get_hog_from_path(daimler_pedestrian_URL)
    x += daimler_pos_hogs
    y = np.append(y, np.ones(size))

    # Leo los de INRIA negativos
    # NOTA: escalo a gris estos negativos y positivos porque INRIA viene en RGB
    inria_neg_hogs, size = get_hog_from_path(INRIA_non_pedestrian_URL, grayscale=True)
    x += inria_neg_hogs
    y = np.append(y, np.zeros(size))

    # Leo los de INRIA positivos
    inria_pos_hogs, size = get_hog_from_path(INRIA_pedestrian_URL, grayscale=True)
    x += inria_pos_hogs
    y = np.append(y, np.ones(size))

    return x, y


def resize(img):
    """Devuelve la imagen con el tamaño modificado"""
    return skimage.transform.resize(img, final_size)


def print_image(img):
    """No va a funcionar correctamente hasta que no se normalice la imagen a una
    escala aceptable, ya que el formato pgm va de 0 a 4096"""
    plt.imshow(img)
    plt.show()


def grayscaled_img(img):
    return np.mean(img, axis=2)


def load_predict_img(img_name):
    img_path = os.path.join(predict_imgs_path, img_name)
    img = skimage.io.imread(img_path)  # Cargo la imagen
    img = grayscaled_img(img)
    img = resize(img)
    return hog(img, pixels_per_cell=(2, 2))


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


def main():
    if not hdf5_URL or not checkpoint_URL:
        print('No se ha seteado el path de HDF5 o del checkpoint!')
        return

    if TRAIN:
        if LOAD_FROM_IMGS:
            x, y = load_training_data()  # Obtengo la data de entrenamiento (previamente corri los scripts de carga)

            # Guardo x, y en HDF5!
            h5f = h5py.File(hdf5_URL, 'w')
            h5f.create_dataset('dataset_x', data=x)
            h5f.create_dataset('dataset_y', data=y)
        else:
            h5f = h5py.File(hdf5_URL, 'r')
            x = h5f['dataset_x'][:]
            y = h5f['dataset_y'][:]

        h5f.close()  # Cierro el archivo HDF5

        # Genero y entreno el SVM
        classifier_svm = svm.SVC()
        classifier_svm.fit(x, y)

        joblib.dump(classifier_svm, checkpoint_URL)
    else:
        classifier_svm = joblib.load(checkpoint_URL)

    predict_data, expected = get_predict_data()
    predictions = classifier_svm.predict(predict_data)

    # Imprimo de forma amigable los resultados de la prediccion
    for prediction, expected_value in zip(predictions, expected):
        value = 'Es peaton.' if prediction == 1 else 'No es peaton.'
        expected_str = 'Se esperaba ' + ('peaton' if expected_value == 1 else 'no peaton')
        correct = '✔' if prediction == expected_value else '✘'
        print(value, expected_str, correct)


if __name__ == '__main__':
    main()
