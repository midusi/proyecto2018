import threading


import skimage.io
import skimage.transform
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.externals import joblib
from skimage.feature import hog
from skimage.color import rgb2gray
import os
import matplotlib.pyplot as plt
import h5py
import random
import numpy as np
import cv2

CONFIG = {
    "HOG": {
        "BLOCK_NORM": "L2-Hys"
    },
    "IMAGE": {
        "FINAL_SIZE": (96, 48)
    },
    "SVM": {
        "C": 250
    }
}


# ---------------TRATAMIENTO DE ARCHIVOS--------------------


def get_filename(path):
    """Devuelve el nombre del archivo, INCLUIDA la extension"""
    return path.split('/')[-1]


def get_basename(path):
    """Devuelve el nombre del archivo SIN extension y la extension por separado"""
    filename = get_filename(path).split('.')
    return filename[0], filename[1]


# ---------------TRATAMIENTO DE IMAGENES--------------------


def resize(image, finalSize):  # Resize común y silvestre
    return skimage.transform.resize(image, finalSize)


def crop_image(image, posX, posY, width, height):  # Corta la imagen dado un punto, ancho y aleatorio
    return image[posY:posY + height, posX:posX + width]


def to_grayscale(image):  # Devuelve la imagen en escala de grises
    return rgb2gray(image)


def load_image_from_path(path):  # Carga una imagen dado un PATH
    return skimage.io.imread(path)


def normalize_image(image, maxValue=False):  # Normaliza la imagen entre 0 y maxValue o 255
    isFloat = type(image[0][0]) is float and image[0][
        0] < 1  # Me fijo si la imagen esta normalizada entre 0 y 1, tomando el primer pixel
    return image / (1 if isFloat else 255)


def normalize_image_max(image):
    """Normaliza la imagen con el maximo valor usando sklearn"""
    return normalize(image, 'max')


def print_image(image):  # Imprime una imagen usando PyPlot
    plt.figure()
    plt.imshow(image)
    plt.show()


def print_images(images, length=-1):  # Imprime varias(todas o N) imagen usando PyPlot.
    if length == -1:
        length = len(images)
    for i in images:
        plt.figure()
        plt.imshow(i)
        length -= 1
        if length == 0:
            break
    plt.show()


def save_img(img, dest_folder, filename):
    """Guarda la imagen en el directorio final"""
    img_path = os.path.join(dest_folder, filename)
    skimage.io.imsave(img_path, img)


def generate_sub_samples(img, original_img_path, folder_dest):
    """A partir de la imagen pasada por parametro se generan sub imagenes"""
    height, width = len(img), len(img[1])
    block_heigth, block_width = int(height / 5), int(width / 5)
    original_filename, extension = get_basename(original_img_path)
    i = 0  # Contador de subimagenes
    y = 0
    while y < height:
        x = 0
        while x < width:
            sub_img = img[y:y + block_heigth, x:x + block_width, :]  # Obtengo una subregion/subimagen
            sub_img = resize(sub_img)
            # Genero el nombre de la imagen a partir del nombre original
            sub_img_filename = original_filename + '_' + str(i) + '.' + extension
            save_img(sub_img, folder_dest, sub_img_filename)
            x += block_width
            i += 1
        y += block_heigth


def get_pyramid(image, scale=1.5, minSize=(30, 30)):
    """Devuelve una lista de escalas diferentes para la imagen pasada, fuente:
    https://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/"""
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = resize(image, [h, w])

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def detect_pedestrian(image, win_w, win_h, epsilon, predict_function):
    image = to_grayscale(image)
    # image = normalize_image_max(image)
    final_bounding_boxes = []
    for i, resized_image in enumerate(get_pyramid(image, scale=epsilon)):
        coefficient = epsilon ** i

        # Loop sobre la ventana deslizante en diferentes posiciones
        for (x, y, window) in get_sliding_window(resized_image, stepSize=(32, 64), windowSize=(win_w, win_h)):
            # Si la ventana no coincide con nuestro tamaño de ventana, se ignora
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != win_h or window.shape[1] != win_w:
                continue

            cropped_image = resize(window, [96, 48])  # Escalo
            cropped_image_hog = get_hog_from_image(cropped_image, normalize=False, )  # Obtengo el HOG

            # Comienzo a predecir
            # prediction = svm.predict([cropped_image_hog])[0]
            prediction = predict_function(cropped_image_hog)
            if prediction:

                # Si es un peaton guardo el bounding box
                bounding_box = (
                    x * coefficient,
                    y * coefficient,
                    (x + win_w) * coefficient,
                    (y + win_h) * coefficient
                )

                final_bounding_boxes.append(bounding_box)

    return non_max_suppression_fast(final_bounding_boxes, 0.25)


def get_sliding_window(image, stepSize, windowSize):
    """Devuelve las ventanas deslizantes de la imagen, fuente:
    https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/"""
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize[0]):
        for x in range(0, image.shape[1], stepSize[1]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def non_max_suppression_fast(boxes, overlapThresh):
    """Non max suppression para bounding boxes solapados, fuente:
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/"""
    # Convierto en arreglo de numpy
    boxes = np.array(boxes)

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return np.array([])

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


# ------------------TRATAMIENTO DE PATHS---------------------


def join_paths(folder, fil):
    return os.path.join(folder, fil)


# ------------------TRATAMIENTO DE HOGS----------------------


def get_hog_from_image(image, grayscale=False, resize=False, finalSize=None, normalize=True, maxValue=False):  # Devuelve el HOG de una imagen
    # Si resize es True, prueba usar finalSize, si finalSize es None, usa la configuración por default definida arriba
    if grayscale:
        image = to_grayscale(image)
    if resize:
        image = resize(image, finalSize if (finalSize != None) else CONFIG["IMAGE"]["FINAL_SIZE"])
    if normalize:
        image = normalize(image, maxValue)
    h = hog(image, block_norm=CONFIG["HOG"]["BLOCK_NORM"], transform_sqrt=True)
    return h


def get_hogs_from_path(pathToFolder, grayscale=False, resize=False, finalSize=None, subset=-1, normalize=True,
                       maxValue=False, printImages=False, printHogs=False):
    hogs = []
    i = 0
    for dirPath, dirName, fileNames in os.walk(pathToFolder):
        random.shuffle(fileNames)
        for f in fileNames:
            image = load_image_from_path(join_paths(dirPath, f))
            if printImages:
                print_image(image)
            image_hog = get_hog_from_image(image, grayscale, resize, finalSize, normalize, maxValue, printHogs)
            hogs.append(image_hog)
            i += 1
            if i == subset:
                return hogs

    return hogs


def get_hogs_from_list(images, grayscale=False, resize=False, finalSize=None, subset=-1, normalize=True, maxValue=False,
                       printImages=False, printHogs=False):
    hogs = []
    i = 0
    for image in images:
        if printImages:
            print_image(image)
        image_hog = get_hog_from_image(image, grayscale, resize, finalSize, normalize, maxValue, printHogs)
        hogs.append(image_hog)
        i += 1
        if i == subset:
            return hogs
    return hogs


def get_hogs_from_path_with_window(pathToFolder, window, grayscale=False, resize=False, finalSize=None, subset=-1,
                                   normalize=True, maxValue=False, printImages=False, printHogs=False,
                                   printSlices=False):
    # Window es una ventana única, con el formato (y,x). i.e.:(96,48)
    return get_hogs_from_path_with_windows(pathToFolder, window, grayscale, resize, finalSize, subset, normalize,
                                           maxValue,
                                           printImages, printHogs, printSlices)


def get_hogs_from_path_with_windows(pathToFolder, windows, grayscale=False, resize=False, finalSize=None, subset=-1,
                                    normalize=True, maxValue=False, printImages=False, printHogs=False,
                                    printSlices=False):
    # Windows es una lista de ventanas, cada ventana tiene el formato (y,x). i.e.: (96,48)
    hogs = []
    i = 0
    for dirPath, dirName, fileNames in os.walk(pathToFolder):
        random.shuffle(fileNames)
        for f in fileNames:
            image = load_image_from_path(join_paths(dirPath, f))
            for w in windows:
                height = w[0]
                width = w[1]
                y = 0
                while y + height < image.shape(0):
                    x = 0
                    while x + width < image.shape(1):
                        img_cropped = crop_image(image, x, y, width, height)
                        if printSlices:
                            print_image(img_cropped)
                        image_hog = get_hog_from_image(img_cropped, grayscale, resize, finalSize, normalize, maxValue,
                                                   printHogs)
                        hogs.append(image_hog)
                        x += width
                    y += height
            if printImages:
                print_image(image)
            i += 1
            if i == subset:
                return hogs
    return hogs


# -----------------------------TRATAMIENTO DE H5PY ----------------------------


def load_h5py(path):
    return h5py.File(path, 'rw')


def create_dataset(h5pyFile, datasetName, dataset):
    h5pyFile.create_dataset(datasetName, data=dataset)


def get_dataset(h5pyFile, datasetName):
    return h5pyFile[datasetName][:]


# -----------------------------TRATAMIENTO DE SVM -----------------------------


def load_checkpoint(path):
    return joblib.load(path)


def save_checkpoint(classifier_svm, path):
    joblib.dump(classifier_svm, path)


def create_linear_svm():
    return svm.LinearSVC(C=CONFIG["SVM"]["C"])
