import os
import utils
import random
import skimage.io
import skimage.transform

DAIMLER_CONFIG = {
    "BASE_PATH": "/home/beto0607/Facu/Pedestrians/Datasets/Daimler/DaimlerBenchmark/",#PATH A DaimlerBenchmark
    "TRAINING":{
        "NEG": DAIMLER_CONFIG["BASE_PATH"] + "Data/TrainingData/NonPedestrians/",
        "POS": DAIMLER_CONFIG["BASE_PATH"] + "Data/TrainingData/Pedestrians/48x96/"
    },
    "TEST": DAIMLER_CONFIG["BASE_PATH"] + "Data/TestData/",
    "GROUND_TRUTH": DAIMLER_CONFIG["BASE_PATH"] + "GroundTruth/GroundTruth2D.db"
}


def save_neg_samples(folder_from, folder_to, subset_size=None):
    """Se encarga de cargar todos los samples negativos"""
    for dirpath, dirnames, filenames in os.walk(folder_from):  # Obtengo los nombres de los archivos
        if subset_size:
            random.shuffle(filenames)  # Los pongo en orden aleatorio cuando genero subset
            filenames = filenames[0:subset_size]  # Si fue especificado un tamaño de subset recorto el dataset
        for filename in filenames:
            img_path = os.path.join(dirpath, filename)
            img = skimage.io.imread(img_path)  # Cargo la imagen
            # generate_sub_samples(img, img_path)  # Genero nuevas muestras a partir de la imagen
            img = utils.resize(img)  # Re escalo la imagen original
            basename, extension = utils.get_basename(filename)
            filename_final = basename + '.png'
            utils.save_img(img, folder_to, filename_final)  # Guardo la imagen en la carpeta de negativos