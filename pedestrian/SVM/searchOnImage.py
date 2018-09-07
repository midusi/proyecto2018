import h5py
import skimage.io
from matplotlib import patches
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.preprocessing import normalize
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os
import random
import re
import sys

CHECKPOINT_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl'  # Path donde se guarda el SVM ya entrenado
IMG_PATH = './imgs/street4.jpg'
FINAL_SIZE = (96, 48)  # Tamaño de la imagen que vamos a manejar
SLIDING_WINDOW_SIZE = [500, 400]  # Tamaño de la ventana deslizante (heigth, width)
SLIDING_WINDOW_STRIDE = (300, 150)  # Stride de la ventana deslizante (heigth, width)
DRAW_SLIDING_WINDOW = True  # En True si se quiere graficar en la imagen la ventrana deslizante
DRAW_PEDRESTRIAN_BOUNDING_BOX = True  # En True si se quiere graficar los bounding boxes de los peatones
SHOW_IMG = False  # Mostar la imagen con los rectangulos dibujados
TEST_SUBSET_SIZE = 0  # Cantidad de imagenes a procesar para la deteccion. 0 si se quieren procesar todas

# HNM
HDF5_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/datasets.h5'  # Path donde se guarda los hogs en HDF5
CHECKPOINT_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl'  # Path donde se guarda el SVM ya entrena
IOU_limit = 0.4  # IOU Limite maximo
DO_HNM = True  # Si esta en True hace HNM, caso no se hace HNM y se ignora el parametro de abajo
HNM_CICLE_COUNT = 600  # Cantidad de imagenes que se quieren procesar antes de hacer HNM. 0 Para esperar a todas (mejor performance pero quizas arroja problemas de memoria)

# INRIA
INRIA_ROOT_FOLDER = '/home/genaro/Descargas/'
INRIA_ANNOTATIONS_FOLDER = '/home/genaro/Descargas/Test/'

# Daimler
DAIMLER_DB_FILE_PATH = '/home/genaro/Descargas/GroundTruth2D.db'
DAIMLER_IMGS_FOLDER_PATH = '/home/genaro/Descargas/DaimlerBenchmark/Data/TestData/'
TYPES_TO_EVALUATE = [  # Tipos de peatones a detectar (los tipos estan en la documentacion de Daimler)
    '0',  # Peatones claramente visibles
]


def draw_rectangle(x, y, width, height, color='0'):
    """Dibuja un rectangulo en la imagen"""
    canvas = plt.gca()
    rectangle = patches.Rectangle((x, y), width, height, fill=False, color=color)
    canvas.add_patch(rectangle)


def get_inria_test_pedestrian_bounding_boxes(file_path):
    """"Devuelve un listado de bounding boxes (Xmin, Ymin, Xmax, Ymax), esta informacion se extrae de los archivos
    dentro de la carpeta 'annotations' correspondientes a cada imagen"""
    bounding_boxes_list = []
    bounding_boxes_file = open(os.path.join(INRIA_ROOT_FOLDER, file_path), encoding="ISO-8859-1")
    min_width = min_height = sys.maxsize
    max_width = max_height = -1
    # Busco por cada linea si son datos de bounding boxes
    for line in bounding_boxes_file.readlines():
        if re.search('Bounding box', line):
            matches = re.findall('(\d+)', line)
            matches = [int(num) for num in matches]  # Paso todas las coordenadas a int
            x, y, width, height = matches[1], matches[2], matches[3], matches[4]
            bounding_boxes_list.append([x, y, width, height])

            # Obtengo los valores para poder sacar el tamanio
            # de una ventana deslizante
            min_width = min(width, min_width)
            min_height = min(height, min_height)
            max_width = max(width, max_width)
            max_height = max(height, max_height)
    mean_width, mean_height = get_mean_sliding_window_parameters(min_width, min_height, max_width, max_height)

    return bounding_boxes_list, mean_width, mean_height


def get_mean_sliding_window_parameters(min_width, min_height, max_width, max_height):
    """"Devuelve la media del ancho y alto para poder usar una ventana deslizante
    medianamente generica"""
    # return SLIDING_WINDOW_SIZE
    if min_width == max_width:
        mean_width = min_width
    else:
        mean_width = int(min_width + (max_width - min_width) / 2)

    if min_height == max_height:
        mean_height = min_height
    else:
        mean_height = int(min_height + (max_height - min_height) / 2)
    return mean_width, mean_height


def detect_pedrestrian(img, pedestrians_bounding_boxes, sliding_window_parameters, classifier_svm, grayscale=False, must_normalize=True):
    """A partir de la imagen pasada por parametro se realiza una
    ventana deslizante y se dibujan las areas donde fue detectada una persona"""
    height, width = len(img), len(img[1])
    # block_heigth, block_width = SLIDING_WINDOW_SIZE
    block_width, block_heigth = sliding_window_parameters
    y = 0
    plt.imshow(img, cmap='gray')
    hogs_to_hard_mining = []
    # stride_y, stride_x = SLIDING_WINDOW_STRIDE
    stride_y, stride_x = int(SLIDING_WINDOW_STRIDE[0] / 2), int(SLIDING_WINDOW_STRIDE[1] / 2)
    # Datos de precision, recall, etc
    total_pedestrian = len(pedestrians_bounding_boxes)
    pedrestrian_predected = pedrestrian_success = 0

    # Paso a escalas de grises
    if grayscale:
        img = grayscaled_img(img)  # Los hogs solo se pueden calcular sobre escala de grises

    # Normalizo
    if must_normalize:
        img = normalize_img(img)

    # Comienzo a correr la ventana deslizante
    while y < height:
        x = 0
        while x < width:
            try:
                # sub_img = img[y:y + block_heigth, x:x + block_width, :]  # Obtengo una subregion/subimagen
                sub_img = img[y:y + block_heigth, x:x + block_width, :]  # Obtengo una subregion/subimagen
            except IndexError:
                # Puede ser posible que algunas imagenes sin RGB arroje este error
                # sub_img = img[y:y + block_heigth, x:x + block_width]
                sub_img = img[y:y + block_heigth, x:x + block_width]
            finally:
                sub_img = resize(sub_img)

            sub_img_hog = hog(sub_img, block_norm='L2-Hys', transform_sqrt=True)
            predictions = classifier_svm.predict([sub_img_hog])

            if SHOW_IMG and DRAW_SLIDING_WINDOW:
                draw_rectangle(x, y, block_width, block_heigth, 'yellow')

            # Busco los falsos positivos!
            if predictions[0] == 1:
                pedrestrian_predected += 1

                img_box = [x, y, x + block_width, y + block_heigth]
                # Veo si tiene algun IOU que valga la pena con algun bounding box de peatones declarados
                must_be_added = False
                intersects_with_pedestrian = False
                for pedestrian_bounding_box in pedestrians_bounding_boxes:
                    # Grafico el bounding boxs si asi se quiere
                    if SHOW_IMG and DRAW_PEDRESTRIAN_BOUNDING_BOX:
                        draw_rectangle(pedestrian_bounding_box[0], pedestrian_bounding_box[1],
                                       pedestrian_bounding_box[2], pedestrian_bounding_box[3])

                    box_to_iou = [
                        pedestrian_bounding_box[0],
                        pedestrian_bounding_box[1],
                        pedestrian_bounding_box[0] + pedestrian_bounding_box[2],
                        pedestrian_bounding_box[1] + pedestrian_bounding_box[3]
                    ]
                    # Calculo el IOU
                    iou = get_iou(img_box, box_to_iou)
                    # print("iou", iou)

                    # Si es menor que el limite de IOU seteado, lo considero para agregar al HNM
                    if iou < IOU_limit:
                        must_be_added = True
                    else:
                        # Grafico en verde los peatones correctamente detectados
                        # draw_rectangle(x, y, block_width, block_heigth, '#008744')
                        draw_rectangle(x, y, block_width, block_heigth, 'blue')
                        intersects_with_pedestrian = True
                if not intersects_with_pedestrian and must_be_added:
                    # Si no esta arriba de una persona y es un falso
                    # positivo lo grafico en rojo ('#d62d20') en la imagen
                    # y lo considero para hacer hard negative mining
                    draw_rectangle(x, y, block_width, block_heigth, '#d62d20')
                    hogs_to_hard_mining.append(sub_img_hog)  # Almaceno para hacer hard negative mining
                else:
                    pedrestrian_success += 1
            x += stride_x
        y += stride_y

    if SHOW_IMG:
        plt.title(
            "Bounding boxes en negro. Ventana deslizante en amarillo. Falsos positivos en Rojo. Positivos en Verde")
        plt.show()  # Muestro la imagen

    return hogs_to_hard_mining, total_pedestrian, pedrestrian_predected, pedrestrian_success


def get_inria_test_data():
    """Se encarga de obtener las imagenes con los bounding boxes
    de los peatones del dataset de test de INRIA"""
    lst_pos_file = os.path.join(INRIA_ANNOTATIONS_FOLDER, 'pos.lst')
    lst_annotations_file = os.path.join(INRIA_ANNOTATIONS_FOLDER, 'annotations.lst')
    content_pos = open(os.path.join(INRIA_ANNOTATIONS_FOLDER, lst_pos_file))  # Abro el listado de imagenes positivas
    content_annotations = open(
        os.path.join(INRIA_ANNOTATIONS_FOLDER, lst_annotations_file))  # Abro el listado de imagenes positivas
    content_pos_lines = content_pos.readlines()
    content_annotations_lines = content_annotations.readlines()
    # Si fue especificado un tamaño de subset recorto la lista de lineas
    ans = []  # Genero una lista de (imagen, [lista de bounding boxes de la imagen])
    if TEST_SUBSET_SIZE:
        combined = list(zip(content_pos_lines, content_annotations_lines))
        # Los pongo en orden aleatorio cuando genero subset
        random.shuffle(combined)
        content_pos_lines, content_annotations_lines = zip(*combined)
        # Recorto el numero de resultados
        content_pos_lines = content_pos_lines[0:TEST_SUBSET_SIZE]
        content_annotations_lines = content_annotations_lines[0:TEST_SUBSET_SIZE]
    for img_path, bounding_boxes_path in zip(content_pos_lines, content_annotations_lines):
        # Elimino el caracter de nueva linea
        img_path = img_path.rstrip('\n')
        bounding_boxes_path = bounding_boxes_path.rstrip('\n')
        img_original = skimage.io.imread(os.path.join(INRIA_ROOT_FOLDER, img_path))  # Cargo la imagen

        # Obtenglo los bounding boxes y los datos de la ventana deslizante conveniente
        bounding_boxes_list, sliding_window_width, sliding_window_height = get_inria_test_pedestrian_bounding_boxes(
            bounding_boxes_path)  # Obtengo los bounding boxes de personas a recortar
        ans.append([img_original, bounding_boxes_list, [sliding_window_width, sliding_window_height]])
    return ans


def get_daimler_test_data():
    """Se encarga de obtener las imagenes con los bounding boxes
    de los peatones del dataset de test de Daimler"""
    db_file = open(DAIMLER_DB_FILE_PATH)  # Abro el listado de imagenes positivas
    db_file_lines = db_file.read()
    # Si fue especificado un tamaño de subset recorto la lista de lineas
    ans = []  # Lista de (imagen, [lista de bounding boxes de la imagen], [width de ventana, heigth de ventana])

    frames = db_file_lines.split(';')  # Separo los frames
    frames = frames[1:]  # Ignoro el primer frame porque es de muestra

    for frame in frames:
        bounding_boxes_list = []
        frame_lines = frame.splitlines()
        count_lines = len(frame_lines) - 1
        img_path = frame_lines[1]  # Obtengo el nombre de la imagen

        max_width = max_height = -1

        i = 4  # Posicion donde esta la descripcion del tipo de peaton
        while i < count_lines:
            # Me fijo si hay algun peaton en el frame
            pedestrian_type_line = (frame_lines[i]).split(' ')
            pedestrian_type = pedestrian_type_line[1] if pedestrian_type_line[0] == '#' else None
            if pedestrian_type in TYPES_TO_EVALUATE:
                coordinates_line = frame_lines[i + 3].split(' ')

                # Agrego el bounding box del peaton a la lista de
                # bounding boxes referentes a la imagen actual
                min_x = int(coordinates_line[0])
                min_y = int(coordinates_line[1])
                max_x = int(coordinates_line[2])
                max_y = int(coordinates_line[3])
                width = max_x - min_x
                height = max_y - min_y
                bounding_boxes_list.append([
                    min_x,  # x
                    min_y,  # y
                    width,
                    height,
                ])

                # Obtengo los valores para poder sacar el tamanio
                # de una ventana deslizante
                max_width = max(width, max_width)
                max_height = max(height, max_height)
            i += 5  # Salto al siguiente tipo de peaton

        # Solo agrego las imagenes que tienen algun peaton
        if bounding_boxes_list:
            # print("La imagen {} tiene {} peatones de tipo 0".format(img_path, len(bounding_boxes_list)))
            img = skimage.io.imread(os.path.join(DAIMLER_IMGS_FOLDER_PATH, img_path))  # Cargo la imagen
            sliding_window_parameters = [max_width, max_height]
            ans.append([img, bounding_boxes_list, sliding_window_parameters])

    # Si se especifica un tamanio de subset recorto el dataset
    # No lo hago al principio porque no todos los frames tienen peatones.
    # Si hago el recorte antes del analisis quizas no veo ninguna imagen
    if TEST_SUBSET_SIZE:
        random.shuffle(ans)  # Randomizo
        ans = ans[0:TEST_SUBSET_SIZE]  # Me quedo solo con subset

    return ans


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
    plt.imshow(img, cmap="gray")
    plt.show()


def print_image2(img):
    """No va a funcionar correctamente hasta que no se normalice la imagen a una
    escala aceptable, ya que el formato pgm va de 0 a 4096"""
    skimage.io.imshow(img)
    # plt.show()


def get_iou(box_a, box_b):
    """Codigo sacado de https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    porque me la re banco. box = [min_x, min_y, max_x, max_y]"""
    if not overlap(box_a, box_b):
        return 0.0

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


def overlap(r1, r2):
    """Devuelve True si los rectangulos tienen interseccion"""
    # return range_overlap(r1.left, r1.right, r2.left, r2.right) and range_overlap(r1.bottom, r1.top, r2.bottom, r2.top)
    return range_overlap(r1[0], r1[2], r2[0], r2[2]) and range_overlap(r1[1], r1[3], r2[1], r2[3])


def range_overlap(a_min, a_max, b_min, b_max):
    """Funcion usada para el calculo de overlapping"""
    return (a_min <= b_max) and (b_min <= a_max)


def grayscaled_img(img):
    """Devuelve la imagen en escala de grises"""
    return rgb2gray(img)


def save_img(img, folder, img_filename):
    """Guarda la imagen en el directorio final"""
    img_path = os.path.join(folder, img_filename)
    skimage.io.imsave(img_path, img)


def normalize_img(img):
    """Normaliza la imagen con el maximo valor usando sklearn"""
    return normalize(img, 'max')


def main():
    if not CHECKPOINT_PATH or not IMG_PATH:
        print('No se ha seteado algunos parametros!')
        return

    # Cargo el SVM y obtengo los datos de testeo
    classifier_svm = joblib.load(CHECKPOINT_PATH)  # Obtengo el modelo a partir de un checkpoint
    # test_imgs = get_inria_test_data()
    test_imgs = get_daimler_test_data()

    i = 1  # Contador para el break

    # Datos finales
    total_pedestrians = pedestrian_predected = pedestrian_success = 0
    hogs_to_hnm = []
    hnm_count = 0
    for img, bounding_boxes, sliding_window_parameters in test_imgs:
        if TEST_SUBSET_SIZE and i > TEST_SUBSET_SIZE:
            break

        hogs_to_hnm_aux, cant_pedestrian, cant_pedrestrian_predected, cant_pedrestrian_success = detect_pedrestrian(img, bounding_boxes,
                                                                                                   sliding_window_parameters,
                                                                                                   classifier_svm)
        hnm_count += 1
        hogs_to_hnm += hogs_to_hnm_aux
        if DO_HNM and HNM_CICLE_COUNT and hnm_count == HNM_CICLE_COUNT:
            do_hard_negative_mining(hogs_to_hnm, classifier_svm)

            # Reseteo los datos
            hnm_count = 0
            hogs_to_hnm = []

        total_pedestrians += cant_pedestrian
        pedestrian_predected += cant_pedrestrian_predected
        pedestrian_success += cant_pedrestrian_success

        i += 1

    # Si no se especifico a la variable HNM_CICLE_COUNT hago HNM con todos
    # los hogs almacenados
    if DO_HNM and not HNM_CICLE_COUNT:
        do_hard_negative_mining(hogs_to_hnm, classifier_svm)

    if pedestrian_predected:
        print("Precision --> {} / {} = {}".format(pedestrian_success, pedestrian_predected,
                                                  pedestrian_success / pedestrian_predected))
        print("Recall --> {} / {} = {}"
              .format(pedestrian_success, total_pedestrians, pedestrian_success / total_pedestrians))
    else:
        print("No se detecto ningun peaton!")


if __name__ == '__main__':
    main()
