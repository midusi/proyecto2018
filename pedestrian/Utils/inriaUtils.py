import skimage.io
import skimage.transform
import os
import re
import random
import utils


def get_inria_bounding_boxes(root_folder, file_path):
    """"Devuelve un listado de bounding boxes (Xmin, Ymin, Xmax, Ymax), esta informacion se extrae de los archivos
    dentro de la carpeta 'annotations' correspondientes a cada imagen"""
    bounding_boxes_list = []
    bounding_boxes_file = open(os.path.join(root_folder, file_path), encoding="ISO-8859-1")
    # Busco por cada linea si son datos de bounding boxes
    for line in bounding_boxes_file.readlines():
        if re.search('Bounding box', line):
            matches = re.findall('(\d+)', line)
            matches = [int(num) for num in matches]  # Paso todas las coordenadas a int
            bounding_boxes_list.append([matches[1], matches[2], matches[3], matches[4]])
    return bounding_boxes_list


def get_inria_bounding_box_cropped(img, bounding_box):
    """"Recorta el bounding box y retorna dicha sub matriz"""
    return img[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]


def save_pos_samples(root_folder, folder_to, train=True, subset_size=None):
    """Se encarga de cargar todos los samples positivos"""
    lst_file_folder = 'Train' if train else 'Test'
    lst_pos_file = os.path.join(lst_file_folder, 'pos.lst')
    lst_annotations_file = os.path.join(lst_file_folder, 'annotations.lst')
    content_pos = open(os.path.join(root_folder, lst_pos_file))  # Abro el listado de imagenes positivas
    content_annotations = open(os.path.join(root_folder, lst_annotations_file))  # Abro el listado de imagenes positivas
    content_pos_lines = content_pos.readlines()
    content_annotations_lines = content_annotations.readlines()
    # Si fue especificado un tamaño de subset recorto la lista de lineas
    if subset_size:
        combined = list(zip(content_pos_lines, content_annotations_lines))
        # Los pongo en orden aleatorio cuando genero subset
        random.shuffle(combined)
        content_pos_lines, content_annotations_lines = zip(*combined)
        # Recorto el numero de resultados
        content_pos_lines = content_pos_lines[0:subset_size]
        content_annotations_lines = content_annotations_lines[0:subset_size]
    for img_path, bounding_boxes_path in zip(content_pos_lines, content_annotations_lines):
        # Elimino el caracter de nueva linea
        img_path = img_path.rstrip('\n')
        bounding_boxes_path = bounding_boxes_path.rstrip('\n')
        img_original = skimage.io.imread(os.path.join(root_folder, img_path))  # Cargo la imagen

        # Obtengo los bounding boxes de personas a recortar
        bounding_boxes_list = get_inria_bounding_boxes(root_folder, bounding_boxes_path)
        for bounding_box in bounding_boxes_list:
            persona = get_inria_bounding_box_cropped(img_original, bounding_box)  # Recorto a la persona
            persona = utils.resize(persona)  # Re escalo la imagen
            img_filename = utils.get_filename(img_path)  # Genero el nombre que tendra la imagen guardada
            utils.save_img(persona, folder_to, img_filename)  # Guardo la imagen en la carpeta de positivos


def save_neg_samples(root_folder, folder_to, train=True, subset_size=None, generate_subset=False):
    """Se encarga de cargar todos los samples negativos"""
    lst_file_folder = 'Train' if train else 'Test'
    lst_neg_file = os.path.join(lst_file_folder, 'neg.lst')
    content_neg = open(os.path.join(root_folder, lst_neg_file))  # Abro el listado de imagenes positivas
    content_neg_lines = content_neg.readlines()
    if subset_size:
        random.shuffle(content_neg_lines)  # Los pongo en orden aleatorio cuando genero subset
        content_neg_lines = content_neg_lines[0:subset_size]  # Si fue especificado un tamaño de subset recorto el dataset
    for img_path in content_neg_lines:
        img_path = img_path.rstrip('\n')  # Cuando lee la linea queda el \n en el final, lo eliminamos
        img = skimage.io.imread(os.path.join(root_folder, img_path))  # Cargo la imagen
        if generate_subset and not subset_size:
            utils.generate_sub_samples(img, img_path, folder_to)  # Genero nuevas muestras a partir de la imagen
        img = utils.resize(img)  # Re escalo la imagen original
        filename = utils.get_filename(img_path)  # Genero el nombre que tendra la imagen guardada
        utils.save_img(img, folder_to, filename)  # Guardo la imagen en la carpeta de negativos