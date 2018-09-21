import utils
import cv2
import numpy as np
from datetime import datetime as dt
from paramsModifyWithoutDialogs import bind_keypress_event
import settings

# Parametros del SVM
CHECKPOINT_PATH = '/home/genaro/PycharmProjects/checkpoints_proyecto2018/svmCheckpoint.pkl'

# Parametros de video
# SAVE_POSITIVE_PATH = '/home/genaro/PycharmProjects/predicciones_positivas/'
SAVE_POSITIVE_PATH = None

# Instancio el SVM de forma global para facil acceso
svm = utils.load_checkpoint(CHECKPOINT_PATH)


def predict_funcion(x):
    """Devuelve true si fue evaluado como peaton"""
    return svm.predict([x])[0] == 1


def predict_proba_funcion(x):
    """Devuelve true si fue evaluado como peaton
    utilizando probabilidades"""
    res = svm.decision_function([x])
    return res[0] > settings.scoreThreshold


def main():
    old_bounding_boxes = np.array([])  # Para le tracking

    # Abro el video
    cap = cv2.VideoCapture(0)
    count_frames = 0

    # Bindeo el evento para la presion de una tecla
    bind_keypress_event()

    while cap.isOpened():
        # Ignora la cantidad de frames seteados
        if count_frames < settings.countIgnoredFrames:
            count_frames += 1
            cap.grab()
            continue

        # Reinicio el contador y obtengo el nuevo frame
        count_frames = 0
        ret, frame = cap.retrieve()

        # Calculo el tiempo de procesamiento de un frame
        begin_time_frame = dt.now()

        # Obtengo los bounding boxes de los peatones del frame
        bounding_boxes = utils.detect_pedestrian(
            frame,
            settings.winWidth,
            settings.winHeight,
            settings.scaleDetection,
            predict_proba_funcion,
            SAVE_POSITIVE_PATH
        )

        # Obtengo los bounding boxes con parametros de tracking
        old_bounding_boxes = utils.tracking_bounding_boxes_ms(
            old_bounding_boxes,
            bounding_boxes,
            settings.trackThreshold,
            begin_time_frame,
            settings.boundBoxLife
        )

        # Dibujo los bounding boxes de los peatones
        for (startX, startY, endX, endY, bounding_box_lifetime) in old_bounding_boxes:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)  # Rectangulo verde

        # Muestro el frame
        cv2.imshow('frame', frame)

        # Si presiona 'q' sale
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libero los recursos del video
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
