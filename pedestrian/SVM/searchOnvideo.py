import skimage
import cv2
import numpy as np
from sklearn.preprocessing import normalize
from skimage.feature import hog
from sklearn.externals import joblib

CHECKPOINT_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl'  # Path donde se guarda el SVM ya entrenado
SLIDING_WINDOW_SIZE = [150, 80]  # Tamaño de la ventana deslizante (heigth, width)
SLIDING_WINDOW_STRIDE = (100, 100)  # Stride de la ventana deslizante (heigth, width)
FINAL_SIZE = (96, 48)  # Tamaño de la imagen que vamos a manejar
DRAW_SLIDING_WINDOW = True


def draw_rectangle(img, x, y, width, height):
    """Dibuja un rectangulo en la imagen"""
    return cv2.rectangle(img, (y, x), (y + height, x + width), color=(0, 255, 0))


def detect_pedrestrian(img_original, classifier_svm, grayscale=True, must_normalize=True):
    """A partir de la imagen pasada por parametro se realiza una
    ventana deslizante y se dibujan las areas donde fue detectada una persona"""
    print(img_original.shape)
    height, width = len(img_original), len(img_original[1])
    block_width, block_heigth = SLIDING_WINDOW_SIZE
    stride_y, stride_x = SLIDING_WINDOW_STRIDE

    print("Width -->", width)
    print("Height -->", height)

    img = np.array(img_original)
    if grayscale:
        # print("Escalando a grises")
        img = grayscaled_img(img)  # Los hogs solo se pueden calcular sobre escala de grises

    if must_normalize:
        # print("Normalizando")
        img = normalize_img(img)

    # Comienzo a correr la ventana deslizante
    y = 0
    while y < height:
        x = 0
        while x < width:
            try:
                # sub_img = img[y:y + block_heigth, x:x + block_width, :]  # Obtengo una subregion/subimagen
                sub_img = img[y:y + block_heigth, x:x + block_width, :]  # Obtengo una subregion/subimagen
            except IndexError:
                sub_img = img[y:y + block_heigth, x:x + block_width]
            finally:
                sub_img = resize(sub_img)

            sub_img_hog = hog(sub_img, block_norm='L2-Hys', transform_sqrt=True)
            predictions = classifier_svm.predict([sub_img_hog])

            if DRAW_SLIDING_WINDOW:
                img_original = draw_rectangle(img_original, x, y, block_width, block_heigth)

            # Si detecto un peaton lo dibujo en azul
            if predictions[0] == 1:
                img_original = draw_rectangle(img_original, x, y, block_width, block_heigth)

            print("x = ", x, " Stride = ", stride_x, "Suma", x + stride_x)
            x += stride_x
        print("Sale de x", x)
        y += stride_y
    return img_original


def normalize_img(img):
    """Normaliza la imagen con el maximo valor usando sklearn"""
    return normalize(img, 'max')


def resize(img):
    """Devuelve la imagen con el tamaño modificado"""
    return skimage.transform.resize(img, FINAL_SIZE)


def grayscaled_img(img):
    """Devuelve la imagen en escala de grises"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    if not CHECKPOINT_PATH:
        print('No se ha seteado algunos parametros!')
        exit()

    # Instancio el SVM
    classifier_svm = joblib.load(CHECKPOINT_PATH)  # Obtengo el modelo a partir de un checkpoint

    # Abro el video
    cap = cv2.VideoCapture('./imgs/video.ogv')

    while cap.isOpened():
        ret, frame = cap.read()
        frame = detect_pedrestrian(frame, classifier_svm)

        cv2.imshow('frame', frame)  # Muestro el frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
