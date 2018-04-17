import skimage.io
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
import os
import time

CHECKPOINT_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl'  # Path donde se guarda el SVM ya entrenado
IMG_PATH = './imgs/street3.jpg'
SAVE_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/imgs_result'
FINAL_SIZE = (96, 48)  # Tamaño de la imagen que vamos a manejar
BLOCK_SIZE = (118, 60)


def print_multiple(list_of_images):
    """Imprime multiples imagenes en pantalla"""
    for img in list_of_images:
        plt.figure()
        plt.imshow(img)
    plt.show()


def get_nanoseconds():
    """Devuelve los nanosegundos actuales"""
    return "%.20f" % time.time()


def generate_sub_samples(img):
    """A partir de la imagen pasada por parametro se generan sub imagenes"""
    height, width = len(img), len(img[1])
    block_heigth, block_width = BLOCK_SIZE  # int(height / BLOCK_SIZE[0]), int(width / BLOCK_SIZE[1])
    y = 0
    sub_samples_imgs = []
    sub_samples_hogs = []
    while y < height:
        x = 0
        while x < width:
            sub_img = img[y:y + block_heigth, x:x + block_width, :]  # Obtengo una subregion/subimagen
            sub_img = resize(sub_img)
            sub_samples_imgs.append(sub_img)
            sub_img = grayscaled_img(sub_img)  # Los hogs solo se pueden calcular sobre escala de grises
            sub_img_hog = hog(sub_img, block_norm='L2-Hys', transform_sqrt=True)
            sub_samples_hogs.append(sub_img_hog)
            x += block_width
        y += block_heigth
    return sub_samples_hogs, sub_samples_imgs


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


def save_img(img, folder, img_filename):
    """Guarda la imagen en el directorio final"""
    img_path = os.path.join(folder, img_filename)
    skimage.io.imsave(img_path, img)


def main():
    if not CHECKPOINT_PATH or not IMG_PATH or not SAVE_PATH:
        print('No se ha seteado algunos parametros!')
        return

    classifier_svm = joblib.load(CHECKPOINT_PATH)  # Obtengo el modelo a partir de un checkpoint
    img = skimage.io.imread(IMG_PATH)  # Cargo la imagen
    sub_samples_hogs, sub_samples_imgs = generate_sub_samples(img)  # Obtengo las submuestras

    # Hacemos la prediccion
    positives = []
    predictions = classifier_svm.predict(sub_samples_hogs)
    for prediction, sub_img in zip(predictions, sub_samples_imgs):
        if prediction == 1:
            positives.append(sub_img)
            save_img(sub_img, SAVE_PATH, get_nanoseconds() + '.jpg')  # Guardo la imagen en la carpeta destino

    print("Cantidad de muestras generadas --> {} | Cantidad de positivos --> {}".format(len(sub_samples_imgs), len(positives)))
    print("Terminado")


if __name__ == '__main__':
    main()
