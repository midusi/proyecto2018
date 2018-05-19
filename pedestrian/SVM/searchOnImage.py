import h5py
import skimage.io
from matplotlib import patches
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
import os
import time

CHECKPOINT_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl'  # Path donde se guarda el SVM ya entrenado
IMG_PATH = './imgs/street4.jpg'
SAVE_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/imgs_result'
FINAL_SIZE = (96, 48)  # Tamaño de la imagen que vamos a manejar
SLIDING_WINDOW_SIZE = (400, 150)  # Tamaño de la ventana deslizante
SLIDING_WINDOW_STRIDE = (400, 150)

# Defino los centros de bounding boxes de los peatones
PEDESTRIAN_BOUNDING_BOXES_CENTER = [
    (219, 379),
    (311, 456),
    (580, 396),
    (940, 384)
]
HDF5_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/datasets.h5'  # Path donde se guarda los hogs en HDF5
CHECKPOINT_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl'  # Path donde se guarda el SVM ya entrena
IOU_limit = 0.6  # IOU Limite maximo



def print_multiple(list_of_images):
    """Imprime multiples imagenes en pantalla"""
    for img in list_of_images:
        plt.figure()
        plt.imshow(img)
    plt.show()


def get_nanoseconds():
    """Devuelve los nanosegundos actuales"""
    return "%.20f" % time.time()


def draw_rectangle(x, y, width, height, color='0'):
    canvas = plt.gca()
    rectangle = patches.Rectangle((x, y), width, height, fill=False, color=color)
    canvas.add_patch(rectangle)


def generate_bounding_boxes():
    """Genera un arreglo de bounding boxes para poder sacar el IOU"""
    bounding_boxes = []
    for center in PEDESTRIAN_BOUNDING_BOXES_CENTER:
        width = SLIDING_WINDOW_SIZE[1]
        height = SLIDING_WINDOW_SIZE[0]
        x = center[0] - width / 2
        y = center[1] - height / 2
        bounding_boxes.append([x, y, x + width, y + height])
        draw_rectangle(x, y, width, height)
    return bounding_boxes


def detect_pedrestrian(img, classifier_svm):
    """A partir de la imagen pasada por parametro se realiza una
    ventana deslizante y se dibujan las areas donde fue detectada una persona"""
    height, width = len(img), len(img[1])
    block_heigth, block_width = SLIDING_WINDOW_SIZE  # int(height / BLOCK_SIZE[0]), int(width / BLOCK_SIZE[1])
    y = 0
    plt.imshow(img)
    bounding_boxes = generate_bounding_boxes()
    hogs_to_hard_mining = []
    stride_x = SLIDING_WINDOW_STRIDE[0]
    stride_y = SLIDING_WINDOW_STRIDE[1]
    while y < height:
        x = 0
        while x < width:
            try:
                sub_img = img[y:y + block_heigth, x:x + block_width, :]  # Obtengo una subregion/subimagen
            except IndexError:
                # Puede ser posible que algunas imagenes sin RGB arroje este error
                sub_img = img[y:y + block_heigth, x:x + block_width]
            finally:
                sub_img = resize(sub_img)
                sub_img = grayscaled_img(sub_img)  # Los hogs solo se pueden calcular sobre escala de grises

            sub_img_hog = hog(sub_img, block_norm='L2-Hys', transform_sqrt=True)
            predictions = classifier_svm.predict([sub_img_hog])

            # Busco los falsos positivos!
            if predictions[0] == 1:
                img_box = [x, y, x + block_width, y + block_heigth]
                # Veo si tiene algun IOU que valga la pena con algun bounding box
                for bounding_box in bounding_boxes:
                    iou = get_iou(img_box, bounding_box)
                    # Si es menor que el limite de IOU seteado, lo grafico
                    if iou < IOU_limit:
                        hogs_to_hard_mining.append(sub_img_hog)  # Almaceno para hacer hard negative mining
                        color = '#d62d20'  # Rojo
                    else:
                        color = '#008744'  # Verde
                    draw_rectangle(x, y, block_width, block_heigth, color)
            # x += block_width
            x += stride_x
        # y += block_heigth
        y += stride_y
    plt.show()  # Muestro la imagen

    # Si hay datos para hacer Hard Negative Mining pregunto
    if hogs_to_hard_mining:
        should_do_hnm = input("Desea hacer Hard Negative Mining con las imagenes remarcadas en rojo?[y/N]")
        if should_do_hnm == 'y':
            do_hard_negative_mining(hogs_to_hard_mining, classifier_svm)
    else:
        print("No hay falsos positivos para hacer Hard Negative Mining")


def do_hard_negative_mining(hogs_to_hard_mining, classifier_svm):
    """Hace hard negative mining con los nuevos HOGS. Recupera desde memoria
    los que ya teniamos, se los agrega y vuelvee a entrenar el SVM"""

    # Obtengo los arreglos
    print("Extrayendo datos guardados")
    h5f = h5py.File(HDF5_PATH, 'r')
    x, y = h5f['dataset_x'][:], h5f['dataset_y'][:]
    h5f.close()  # Cierro el archivo HDF5

    # Concateno los nuevos valores
    print("Almacenando nuevos cambios")
    hogs_to_hard_mining = np.array(hogs_to_hard_mining)
    x = np.concatenate([x, hogs_to_hard_mining])
    y = np.append(y, np.zeros(len(hogs_to_hard_mining)))

    # Guardo los nuevos valores
    print("Guardando cambios modificados")
    h5f = h5py.File(HDF5_PATH, 'w')
    h5f.create_dataset('dataset_x', data=x)
    h5f.create_dataset('dataset_y', data=y)
    h5f.close()  # Cierro el archivo HDF5

    # Entreno al SVM nuevamente
    print("Reentrenando al SVM")
    classifier_svm.fit(x, y)
    joblib.dump(classifier_svm, CHECKPOINT_PATH)  # Guardo los cambios

    print("Hard Negative Mining terminado")


def resize(img):
    """Devuelve la imagen con el tamaño modificado"""
    return skimage.transform.resize(img, FINAL_SIZE)


def print_image(img):
    """No va a funcionar correctamente hasta que no se normalice la imagen a una
    escala aceptable, ya que el formato pgm va de 0 a 4096"""
    plt.imshow(img)
    plt.show()


def get_iou(box_a, box_b):
    """Codigo sacado de https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    porque me la re banco"""
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # Valor final
    return iou


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
    detect_pedrestrian(img, classifier_svm)


if __name__ == '__main__':
    main()
