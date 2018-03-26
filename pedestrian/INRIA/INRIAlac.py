import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import re
import os

root_folder = '/home/genaro/Descargas/INRIAPerson/'
folder_pos_to = '/home/genaro/Descargas/INRIAPerson/train_final/pos/'
folder_neg_to = '/home/genaro/Descargas/INRIAPerson/train_final/neg/'
final_size = [96, 48]


def get_filename(path):
    """Devuelve el nombre del archivo, INCLUIDA la extension"""
    return path.split('/')[-1]


def get_basename(path):
    """Devuelve el nombre del archivo SIN extension y la extension por separado"""
    filename = get_filename(path).split('.')
    return filename[0], filename[1]


def get_bounding_boxes(file_path):
    """"Devulve un listado de bounding boxes (Xmin, Ymin, Xmax, Ymax), esta informacion se extrae de los archivos
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


def print_image(img):
    plt.imshow(img, vmin=0, vmax=1)
    plt.show()


def print_mulitple(list_of_images):
    for img in list_of_images:
        plt.figure()
        plt.imshow(img, vmin=0, vmax=1)
    plt.show()


def resize(img):
    """Devuelve la imagen con el tamaño modificado"""
    return skimage.transform.resize(img, final_size)


def get_bounding_box_cropped(img, bounding_box):
    """"Recorta el bounding box y retorna dicha sub matriz"""
    return img[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]


def save_img(img, folder, img_filename):
    """Guarda la imagen en el directorio final"""
    img_path = os.path.join(folder, img_filename)
    skimage.io.imsave(img_path, img)
    # print('Imagen guardada en ' + folder + img_filename)


def generate_sub_samples(img, original_img_path):
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
            save_img(sub_img, folder_neg_to, sub_img_filename)
            x += block_width
            i += 1
        y += block_heigth


def load_pos():
    """Se encarga de cargar todos los samples positivos"""
    content_pos = open(os.path.join(root_folder, 'Train/pos.lst'))  # Abro el listado de imagenes positivas
    content_annotations = open(os.path.join(root_folder, 'Train/annotations.lst'))  # Abro el listado de imagenes positivas
    for img_path, bounding_boxes_path in zip(content_pos.readlines(), content_annotations.readlines()):
        # Elimino el caracter de nueva linea
        img_path = img_path.rstrip('\n')
        bounding_boxes_path = bounding_boxes_path.rstrip('\n')
        img_original = skimage.io.imread(os.path.join(root_folder, img_path))  # Cargo la imagen
        bounding_boxes_list = get_bounding_boxes(bounding_boxes_path)  # Obtengo los bounding boxes de personas a recortar
        for bounding_box in bounding_boxes_list:
            persona = get_bounding_box_cropped(img_original, bounding_box)  # Recorto a la persona
            persona = resize(persona)  # Re escalo la imagen
            img_filename = get_filename(img_path)  # Genero el nombre que tendra la imagen guardada
            save_img(persona, folder_pos_to, img_filename)  # Guardo la imagen en la carpeta de positivos


def load_neg():
    """Se encarga de cargar todos los samples negativos"""
    content_neg = open(os.path.join(root_folder, 'Train/neg.lst'))  # Abro el listado de imagenes positivas
    for img_path in content_neg.readlines():
        img_path = img_path.rstrip('\n')  # Cuando lee la linea queda el \n en el final, lo eliminamos
        img = skimage.io.imread(os.path.join(root_folder, img_path))  # Cargo la imagen
        generate_sub_samples(img, img_path)  # Genero nuevas muestras a partir de la imagen
        img = resize(img)  # Re escalo la imagen original
        filename = get_filename(img_path)  # Genero el nombre que tendra la imagen guardada
        save_img(img, folder_neg_to, filename)  # Guardo la imagen en la carpeta de negativos


def main():
    load_pos()  # Cargo muestras positivas
    load_neg()  # Cargo muestras negativas


if __name__ == '__main__':
    """INRIA Like A Champion"""
    main()