import utils
import numpy as np

# Parametros del SVM
HDF5_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/datasets.h5'
CHECKPOINT_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl'

# Path de los samples negativos
NEGATIVES_SAMPLES_PATH = '/home/genaro/PycharmProjects/samples_negativos/'


def main():
    svm = utils.load_checkpoint(CHECKPOINT_PATH)  # Cargo el SVM que voy a re entrenar
    hogs = utils.get_hog_from_path(
        NEGATIVES_SAMPLES_PATH,
        must_grayscale=False,
        must_normalize=False,
        must_resize=False
    )  # Obtengo los HOGs de los samples negativos
    # print(np.array(hogs).shape)
    utils.do_hard_negative_mining(hogs, svm, HDF5_PATH, CHECKPOINT_PATH)  # Hago HNM


if __name__ == '__main__':
    main()