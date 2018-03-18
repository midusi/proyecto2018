import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import re
import os
import numpy as np

root_folder = 'DataSetAutos/'
folder_pos_to = 'DataSetAutos/Resize/pos/'
folder_neg_to = 'DataSetAutos/Resize/neg/'
final_size = [50, 50]

def get_filename(path):
    """Devuelve el nombre del archivo, INCLUIDA la extension"""
    return path.split('/')[-1]


def get_basename(path):
    """Devuelve el nombre del archivo SIN extension y la extension por separado"""
    filename = get_filename(path).split('.')
    return filename[0], filename[1]

def resize(img):
    """Devuelve la imagen con el tamaño modificado"""
    return skimage.transform.resize(img, final_size)

def save_img(img, folder, img_filename):
    """Guarda la imagen en el directorio final"""
    skimage.io.imsave(folder + img_filename, img)
    # print('Imagen guardada en ' + folder + img_filename)

def generate_sub_samples(img, original_img_path):
    """A partir de la imagen pasada por parametro se generan sub imagenes"""
    #height, width = len(img), len(img[1])
    height = np.size(img, 0)
    width = np.size(img, 1)
    print(height)
    block_heigth, block_width = int(height / 5), int(width / 5)
    original_filename, extension = get_basename(original_img_path)
    i = 0  # Contador de subimagenes
    y = 0
    while y < height:
        x = 0
        while x < width:
            try: # Puede ser que la subdivision de la imagen no sea siempre igual. Lo que sobra no lo uso
                sub_img = img[y:y + block_heigth, x:x + block_width, :]  # Obtengo una subregion/subimagen
                sub_img = resize(sub_img)
                # Genero el nombre de la imagen a partir del nombre original
                sub_img_filename = original_filename + '_' + str(i) + '.' + extension
                save_img(sub_img, folder_neg_to, sub_img_filename)
                x += block_width
                i += 1
            except:
                break
        y += block_heigth

def load_neg():
    """Se encarga de cargar todos los samples negativos"""

    files = os.listdir(root_folder + "/negative") # Lista todos los archivos de ese directorio
    for img_path in files:
        img_path = img_path.rstrip('\n')
        img = skimage.io.imread(root_folder + "/negative/" + img_path)  # Cargo la imagen
        generate_sub_samples(img, img_path)  # Genero nuevas muestras a partir de la imagen
        #print(img_path)
        img = resize(img)  # Re escalo la imagen original
        filename = get_filename(img_path)  # Genero el nombre que tendra la imagen guardada
        save_img(img, folder_neg_to, filename)  # Guardo la imagen en la carpeta de negativos

def main():
    load_neg() # Cargo muestras negativas

if __name__ == '__main__':
    main()