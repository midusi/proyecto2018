import os
import random
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt

folder_neg_from = '/home/genaro/Descargas/DaimlerTrainingData/DaimlerBenchmark/Data/TrainingData/NonPedestrians/'
folder_neg_to = '/home/genaro/Descargas/DaimlerTrainingData/DaimlerBenchmark/Data/TrainingData/NonPedestrians_final/'
final_size = [96, 48]
subset_size = 500


def get_filename(path):
    """Devuelve el nombre del archivo, INCLUIDA la extension"""
    return path.split('/')[-1]


def get_basename(path):
    """Devuelve el nombre del archivo SIN extension y la extension por separado"""
    filename = get_filename(path).split('.')
    return filename[0], filename[1]


def print_image(img):
    """No va a funcionar correctamente hasta que no se normalice la imagen a una
    escala aceptable, ya que el formato pgm va de 0 a 4096"""
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


def save_img(img, folder, img_filename):
    """Guarda la imagen en el directorio final"""
    img_path = os.path.join(folder, img_filename)
    skimage.io.imsave(img_path, img)
    # print('Imagen guardada en ' + folder + img_filename)


# No se usa, pero la dejo por las dudas en caso de que se quieran generar mas muestras negativas
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


def load_neg():
    """Se encarga de cargar todos los samples negativos"""
    for dirpath, dirnames, filenames in os.walk(folder_neg_from):  # Obtengo los nombres de los archivos
        if subset_size:
            random.shuffle(filenames)  # Los pongo en orden aleatorio cuando genero subset
            filenames = filenames[0:subset_size]  # Si fue especificado un tamaño de subset recorto el dataset
        for filename in filenames:
            img_path = os.path.join(dirpath, filename)
            img = skimage.io.imread(img_path)  # Cargo la imagen
            # generate_sub_samples(img, img_path)  # Genero nuevas muestras a partir de la imagen
            img = resize(img)  # Re escalo la imagen original
            basename, extension = get_basename(filename)
            filename_final = basename + '.png'
            save_img(img, folder_neg_to, filename_final)  # Guardo la imagen en la carpeta de negativos


def main():
    load_neg()  # Cargo muestras negativas


if __name__ == '__main__':
    """Daimler Loader"""
    main()
